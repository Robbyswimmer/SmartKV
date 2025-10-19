/*
 * SmartKV CUDA Kernels - Implementation
 *
 * Fused quantized attention with on-the-fly dequantization.
 * Phase 2.2: Minimal kernel implementation with numerically stable softmax
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>

namespace {

// Warp-level reduction for max/sum
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for max
__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
    val = warp_reduce_max(val);

    // Write to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Final reduction across warps
    if (wid == 0) {
        val = (lane < (blockDim.x / 32)) ? shared[lane] : -FLT_MAX;
        val = warp_reduce_max(val);
    }

    return val;
}

// Block-level reduction for sum
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Final reduction across warps
    if (wid == 0) {
        val = (lane < (blockDim.x / 32)) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }

    return val;
}

// Minimal quantized attention kernel (Phase 2.2)
// Implements: output = softmax(Q @ K^T / sqrt(d)) @ V
// With on-the-fly dequantization of K and V

__global__ void quantized_attention_kernel_minimal(
    const float* __restrict__ query,
    const int8_t* __restrict__ key_int8,
    const float* __restrict__ key_scale,
    const int8_t* __restrict__ value_int8,
    const float* __restrict__ value_scale,
    const float* __restrict__ attention_mask,
    float* __restrict__ output,
    int B, int H, int q_len, int k_len, int d
) {
    // Grid: (B, H, q_len)
    int b = blockIdx.x;
    int h = blockIdx.y;
    int q_pos = blockIdx.z;

    if (b >= B || h >= H || q_pos >= q_len) return;

    // Shared memory layout:
    // [0..d-1]: query vector
    // [d..d+k_len-1]: attention scores
    // [d+k_len..d+k_len+31]: reduction scratch space
    extern __shared__ float shared_mem[];
    float* shared_query = shared_mem;
    float* shared_scores = shared_mem + d;
    float* shared_scratch = shared_mem + d + k_len;

    // Load query to shared memory
    for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) {
        int q_offset = ((b * H + h) * q_len + q_pos) * d + d_idx;
        shared_query[d_idx] = query[q_offset];
    }
    __syncthreads();

    float inv_sqrt_d = 1.0f / sqrtf((float)d);

    // Phase 1: Compute attention scores (Q @ K^T)
    for (int k_pos = threadIdx.x; k_pos < k_len; k_pos += blockDim.x) {
        float score = 0.0f;

        // Get scale for this key
        int k_scale_offset = (b * H + h) * k_len + k_pos;
        float k_scale_val = key_scale[k_scale_offset];

        // Dot product with on-the-fly dequantization
        for (int d_idx = 0; d_idx < d; d_idx++) {
            float q_val = shared_query[d_idx];

            int k_offset = ((b * H + h) * k_len + k_pos) * d + d_idx;
            int8_t k_q_val = key_int8[k_offset];
            float k_val = (float)k_q_val * k_scale_val;  // Fused dequantization

            score += q_val * k_val;
        }

        score *= inv_sqrt_d;

        // Apply attention mask if provided
        if (attention_mask != nullptr) {
            int mask_offset = ((b * 1 + 0) * q_len + q_pos) * k_len + k_pos;
            score += attention_mask[mask_offset];
        }

        shared_scores[k_pos] = score;
    }
    __syncthreads();

    // Phase 2: Numerically stable softmax
    // Find max score for numerical stability
    float max_score = -FLT_MAX;
    for (int k_pos = threadIdx.x; k_pos < k_len; k_pos += blockDim.x) {
        max_score = fmaxf(max_score, shared_scores[k_pos]);
    }
    max_score = block_reduce_max(max_score, shared_scratch);

    // Broadcast max to all threads
    if (threadIdx.x == 0) {
        shared_scratch[0] = max_score;
    }
    __syncthreads();
    max_score = shared_scratch[0];

    // Compute exp(score - max) and sum
    float exp_sum = 0.0f;
    for (int k_pos = threadIdx.x; k_pos < k_len; k_pos += blockDim.x) {
        float exp_score = expf(shared_scores[k_pos] - max_score);
        shared_scores[k_pos] = exp_score;
        exp_sum += exp_score;
    }
    exp_sum = block_reduce_sum(exp_sum, shared_scratch);

    // Broadcast sum to all threads
    if (threadIdx.x == 0) {
        shared_scratch[0] = exp_sum;
    }
    __syncthreads();
    exp_sum = shared_scratch[0];

    // Normalize to get probabilities
    for (int k_pos = threadIdx.x; k_pos < k_len; k_pos += blockDim.x) {
        shared_scores[k_pos] /= exp_sum;
    }
    __syncthreads();

    // Phase 3: Weighted sum of values (probs @ V)
    for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) {
        float output_val = 0.0f;

        for (int k_pos = 0; k_pos < k_len; k_pos++) {
            float prob = shared_scores[k_pos];

            // Get scale for this value
            int v_scale_offset = (b * H + h) * k_len + k_pos;
            float v_scale_val = value_scale[v_scale_offset];

            // Dequantize value and accumulate
            int v_offset = ((b * H + h) * k_len + k_pos) * d + d_idx;
            int8_t v_q_val = value_int8[v_offset];
            float v_val = (float)v_q_val * v_scale_val;

            output_val += prob * v_val;
        }

        // Write output
        int out_offset = ((b * H + h) * q_len + q_pos) * d + d_idx;
        output[out_offset] = output_val;
    }
}

} // namespace

// PyTorch wrapper
torch::Tensor quantized_attention_forward(
    torch::Tensor query,
    torch::Tensor key_int8,
    torch::Tensor key_scale,
    torch::Tensor value_int8,
    torch::Tensor value_scale,
    torch::optional<torch::Tensor> attention_mask
) {
    // Input validation
    TORCH_CHECK(query.is_cuda(), "query must be on CUDA");
    TORCH_CHECK(key_int8.is_cuda(), "key_int8 must be on CUDA");
    TORCH_CHECK(key_int8.dtype() == torch::kInt8, "key_int8 must be int8");
    TORCH_CHECK(value_int8.dtype() == torch::kInt8, "value_int8 must be int8");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");

    const auto B = query.size(0);
    const auto H = query.size(1);
    const auto q_len = query.size(2);
    const auto d = query.size(3);
    const auto k_len = key_int8.size(2);

    // Allocate output
    auto output = torch::zeros_like(query);

    // Kernel launch configuration
    dim3 grid(B, H, q_len);
    int block_size = std::min(static_cast<int>(k_len), 256);
    dim3 block(block_size);

    // Shared memory: query + scores + scratch
    size_t shared_mem_size = (d + k_len + 32) * sizeof(float);

    // Get pointers
    const float* query_ptr = query.data_ptr<float>();
    const int8_t* key_int8_ptr = key_int8.data_ptr<int8_t>();
    const float* key_scale_ptr = key_scale.data_ptr<float>();
    const int8_t* value_int8_ptr = value_int8.data_ptr<int8_t>();
    const float* value_scale_ptr = value_scale.data_ptr<float>();
    const float* mask_ptr = attention_mask.has_value() ? attention_mask.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // Launch kernel
    quantized_attention_kernel_minimal<<<grid, block, shared_mem_size>>>(
        query_ptr,
        key_int8_ptr,
        key_scale_ptr,
        value_int8_ptr,
        value_scale_ptr,
        mask_ptr,
        output_ptr,
        B, H, q_len, k_len, d
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("quantized_attention_kernel_minimal failed: ", cudaGetErrorString(err));
    }

    return output;
}

std::tuple<torch::Tensor, torch::Tensor> quantize_per_head_forward(
    torch::Tensor input,
    int bits
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dim() == 3, "input must have shape [N, H, D]");

    auto input_f32 = input.to(torch::kFloat32);
    const int64_t N = input_f32.size(0);
    const int64_t H = input_f32.size(1);

    int max_val = (bits == 1) ? 0 : (1 << (bits - 1)) - 1;
    int min_val = (bits == 1) ? 0 : -(1 << (bits - 1));
    float denom = static_cast<float>(std::max(max_val, 1));

    // Per-head absolute max
    auto abs_max = input_f32.abs().amax(/*dim=*/2, /*keepdim=*/true);
    auto scale = abs_max / denom;
    auto ones = torch::ones_like(scale, scale.options());
    scale = torch::where(scale == 0, ones, scale);

    auto quantized = torch::round(input_f32 / scale)
                         .clamp(min_val, max_val)
                         .to(torch::kInt8);

    auto scale_out = scale.view({N, H});

    return std::make_tuple(quantized, scale_out);
}
