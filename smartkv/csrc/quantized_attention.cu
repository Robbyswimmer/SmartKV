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
#include <cstdint>

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

// ============================================================================
// On-the-fly unpacking helpers for bucket-aware kernel
// ============================================================================

// Packed bit-stream reader that amortizes bit extraction cost across elements.
template<int BITS>
struct PackedBitReader {
    static_assert(BITS > 0 && BITS <= 7, "PackedBitReader only supports sub-byte widths");

    const uint8_t* ptr;
    uint32_t buffer;
    int bits_in_buffer;

    __device__ __forceinline__ explicit PackedBitReader(const uint8_t* base)
        : ptr(base), buffer(0), bits_in_buffer(0) {}

    __device__ __forceinline__ int8_t next() {
        constexpr int MASK = (1 << BITS) - 1;
        constexpr int ZERO_POINT = 1 << (BITS - 1);

        while (bits_in_buffer < BITS) {
            buffer |= static_cast<uint32_t>(*ptr++) << bits_in_buffer;
            bits_in_buffer += 8;
        }

        int raw = static_cast<int>(buffer & MASK);
        buffer >>= BITS;
        bits_in_buffer -= BITS;
        return static_cast<int8_t>(raw - ZERO_POINT);
    }
};

// Generic unpacking dispatcher (compile-time template)
template<int BITS>
__device__ __forceinline__ int8_t unpack_value(const void* packed_ptr, int idx) {
    const uint8_t* packed = reinterpret_cast<const uint8_t*>(packed_ptr);
    if (BITS == 2 || BITS == 3 || BITS == 4) {
        // Fallback to index-based extraction for random access (used rarely after introducing
        // PackedBitReader for the streaming code-paths).
        if (BITS == 2) {
            int byte_idx = idx / 4;
            int bit_offset = (idx % 4) * 2;
            uint8_t packed_byte = packed[byte_idx];
            uint8_t unsigned_val = (packed_byte >> bit_offset) & 0x3;
            return static_cast<int8_t>(unsigned_val - 2);
        } else if (BITS == 3) {
            int bit_offset = idx * 3;
            int byte_idx = bit_offset / 8;
            int bit_in_byte = bit_offset % 8;

            uint32_t window = packed[byte_idx];
            if (bit_in_byte + 3 > 8) {
                window |= static_cast<uint32_t>(packed[byte_idx + 1]) << 8;
            }

            uint8_t unsigned_val = (window >> bit_in_byte) & 0x7;
            return static_cast<int8_t>(unsigned_val - 4);
        } else {
            int byte_idx = idx / 2;
            int bit_offset = (idx % 2) * 4;
            uint8_t packed_byte = packed[byte_idx];
            uint8_t unsigned_val = (packed_byte >> bit_offset) & 0xF;
            return static_cast<int8_t>(unsigned_val - 8);
        }
    } else {
        // 8-bit: direct load as int8
        return reinterpret_cast<const int8_t*>(packed)[idx];
    }
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

__global__ void quantize_per_head_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int N,
    int H,
    int D,
    int bits,
    int max_q,
    int min_q
) {
    int n = blockIdx.x;
    int h = blockIdx.y;
    int idx = n * H + h;
    const float* row = input + idx * D;

    extern __shared__ float shared[];
    float* scratch = shared;  // size = blockDim.x / 32

    float local_max = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(row[d]));
    }
    local_max = block_reduce_max(local_max, scratch);

    __shared__ float scale;
    if (threadIdx.x == 0) {
        scale = local_max / static_cast<float>(max_q > 0 ? max_q : 1);
        if (scale == 0.0f) {
            scale = 1.0f;
        }
        scales[idx] = scale;
    }
    __syncthreads();

    float s = scale;
    int8_t* out_row = output + idx * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float q = roundf(row[d] / s);
        q = fmaxf(static_cast<float>(min_q), fminf(static_cast<float>(max_q), q));
        out_row[d] = static_cast<int8_t>(q);
    }
}

std::tuple<torch::Tensor, torch::Tensor> quantize_per_head_forward(
    torch::Tensor input,
    int bits
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dim() == 3, "input must have shape [N, H, D]");

    auto input_f32 = input.to(torch::kFloat32).contiguous();
    const int64_t N = input_f32.size(0);
    const int64_t H = input_f32.size(1);
    const int64_t D = input_f32.size(2);

    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 8,
                "Unsupported bit-width for CUDA quantization");

    auto options_quant = torch::TensorOptions()
                             .dtype(torch::kInt8)
                             .device(input.device());
    auto options_scale = torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .device(input.device());

    auto quantized = torch::empty({N, H, D}, options_quant);
    auto scales = torch::empty({N, H}, options_scale);

    int max_q = (bits == 1) ? 0 : (1 << (bits - 1)) - 1;
    int min_q = (bits == 1) ? 0 : -(1 << (bits - 1));

    dim3 grid(N, H, 1);
    int threads = std::min<int64_t>(256, D);
    threads = std::max<int>(32, ((threads + 31) / 32) * 32);
    int num_warps = threads / 32;
    size_t shared_mem = static_cast<size_t>(num_warps) * sizeof(float);

    quantize_per_head_kernel<<<grid, threads, shared_mem>>>(
        input_f32.data_ptr<float>(),
        quantized.data_ptr<int8_t>(),
        scales.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(H),
        static_cast<int>(D),
        bits,
        max_q,
        min_q
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("quantize_per_head_kernel failed: ", cudaGetErrorString(err));
    }

    return std::make_tuple(quantized, scales);
}

// ============================================================================
// Bucket-aware tiled attention kernel with streaming softmax
// ============================================================================

template<int BITS, int TILE_SIZE = 64>
__global__ void quantized_attention_bucket_tiled_kernel(
    const float* __restrict__ query,           // [B, H, q_len, d]
    const void* __restrict__ key_qx,           // [num_tokens, H, packed_dim] (uint8 if packed, int8 if not)
    const float* __restrict__ key_scale,       // [num_tokens, H]
    const void* __restrict__ value_qx,         // [num_tokens, H, packed_dim]
    const float* __restrict__ value_scale,     // [num_tokens, H]
    const float* __restrict__ attention_mask,  // [B, 1, q_len, full_k_len] or nullptr
    const int64_t* __restrict__ global_slots,  // [num_tokens] slot indices into full context
    float* __restrict__ output,                // [B, H, q_len, d] UNNORMALIZED
    float* __restrict__ m_out,                 // [B, H, q_len] max logit
    float* __restrict__ s_out,                 // [B, H, q_len] exp sum
    int B, int H, int q_len, int k_len, int d,
    int packed_dim,  // actual dimension of packed data (d for int8, packed size for 2/3/4-bit)
    bool is_packed,  // true if data is bit-packed
    int full_k_len,  // full context length for mask stride (may differ from k_len in bucket)
    int64_t key_stride_tokens,
    int64_t key_stride_heads,
    int64_t key_stride_dim,
    int64_t value_stride_tokens,
    int64_t value_stride_heads,
    int64_t value_stride_dim
) {
    // Grid: (B, H, q_len)
    int b = blockIdx.x;
    int h = blockIdx.y;
    int q_pos = blockIdx.z;

    if (b >= B || h >= H || q_pos >= q_len) return;

    // Shared memory layout (float unless noted):
    // [0..d-1]                     : query vector
    // [d..2d-1]                    : output accumulator (per-thread)
    // [2d..2d+TILE_SIZE-1]         : tile attention scores
    // [2d+TILE_SIZE..2d+2*TILE_SIZE-1]   : key scales for active tile
    // [2d+2*TILE_SIZE..2d+3*TILE_SIZE-1] : value scales for active tile
    // [2d+3*TILE_SIZE..2d+3*TILE_SIZE+31]: reduction scratch space
    // [ints next TILE_SIZE]              : cached global slot indices
    // [following d floats]               : dequantized value tile
    extern __shared__ unsigned char shared_raw[];
    float* shared_query = reinterpret_cast<float*>(shared_raw);
    float* output_acc = shared_query + d;
    float* shared_scores = output_acc + d;
    float* shared_key_scales = shared_scores + TILE_SIZE;
    float* shared_value_scales = shared_key_scales + TILE_SIZE;
    float* shared_scratch = shared_value_scales + TILE_SIZE;
    int32_t* shared_slots = reinterpret_cast<int32_t*>(shared_scratch + 32);
    float* shared_value_tile = reinterpret_cast<float*>(shared_slots + TILE_SIZE);

    // Load query to shared memory
    for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) {
        int q_offset = ((b * H + h) * q_len + q_pos) * d + d_idx;
        shared_query[d_idx] = query[q_offset];
    }
    __syncthreads();

    float inv_sqrt_d = 1.0f / sqrtf((float)d);

    // Streaming softmax accumulators
    float m = -FLT_MAX;  // running max
    float s = 0.0f;      // running sum

    // Initialize output accumulator in shared memory
    for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) {
        output_acc[d_idx] = 0.0f;
    }

    // Number of tiles
    int num_tiles = (k_len + TILE_SIZE - 1) / TILE_SIZE;

    int64_t mask_row_offset = 0;
    if (attention_mask != nullptr) {
        mask_row_offset = static_cast<int64_t>((b * 1 + 0) * q_len + q_pos) * full_k_len;
    }

    (void)packed_dim;  // kept for API compatibility

    const uint8_t* key_base_u8 = reinterpret_cast<const uint8_t*>(key_qx);
    const int8_t* key_base_i8 = reinterpret_cast<const int8_t*>(key_qx);
    const uint8_t* value_base_u8 = reinterpret_cast<const uint8_t*>(value_qx);
    const int8_t* value_base_i8 = reinterpret_cast<const int8_t*>(value_qx);

    // Process each tile
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * TILE_SIZE;
        int tile_end = min(tile_start + TILE_SIZE, k_len);
        int tile_size = tile_end - tile_start;

        __syncthreads();

        // Stage scales (and slots if needed) for this tile into shared memory so that they can
        // be reused without extra global reads in later phases.
        for (int local_k = threadIdx.x; local_k < tile_size; local_k += blockDim.x) {
            int k_pos = tile_start + local_k;
            shared_key_scales[local_k] = key_scale[k_pos * H + h];
            shared_value_scales[local_k] = value_scale[k_pos * H + h];
            if (global_slots != nullptr) {
                shared_slots[local_k] = static_cast<int32_t>(global_slots[k_pos]);
            }
        }
        __syncthreads();

        // Phase 1: Compute attention scores for this tile (Q @ K^T)
        for (int local_k = threadIdx.x; local_k < tile_size; local_k += blockDim.x) {
            int k_pos = tile_start + local_k;
            float score = 0.0f;

            // Get scale for this key
            float k_scale_val = shared_key_scales[local_k];

            // Compute dot product with on-the-fly unpacking
            int64_t key_row_offset = static_cast<int64_t>(k_pos) * key_stride_tokens +
                                     static_cast<int64_t>(h) * key_stride_heads;
            bool used_packed_reader = false;
            if constexpr (BITS < 8) {
                if (is_packed) {
                    const uint8_t* k_packed_row = key_base_u8 + key_row_offset;
                    PackedBitReader<BITS> reader(k_packed_row);
                    for (int d_idx = 0; d_idx < d; d_idx++) {
                        float q_val = shared_query[d_idx];
                        int8_t k_q_val = reader.next();
                        float k_val = static_cast<float>(k_q_val) * k_scale_val;
                        score += q_val * k_val;
                    }
                    used_packed_reader = true;
                }
            }

            if (!used_packed_reader) {
                const int8_t* k_row_ptr = key_base_i8 + key_row_offset;
                if (key_stride_dim == 1) {
                    for (int d_idx = 0; d_idx < d; d_idx++) {
                        float q_val = shared_query[d_idx];
                        int8_t k_q_val = k_row_ptr[d_idx];
                        float k_val = static_cast<float>(k_q_val) * k_scale_val;
                        score += q_val * k_val;
                    }
                } else {
                    for (int d_idx = 0; d_idx < d; d_idx++) {
                        float q_val = shared_query[d_idx];
                        int8_t k_q_val = k_row_ptr[d_idx * key_stride_dim];
                        float k_val = static_cast<float>(k_q_val) * k_scale_val;
                        score += q_val * k_val;
                    }
                }
            }

            score *= inv_sqrt_d;

            // Apply attention mask if provided
            if (attention_mask != nullptr) {
                int slot = global_slots != nullptr ? shared_slots[local_k] : k_pos;
                score += attention_mask[mask_row_offset + static_cast<int64_t>(slot)];
            }

            shared_scores[local_k] = score;
        }
        __syncthreads();

        // Phase 2: Streaming softmax update
        // Find max score in this tile
        float tile_max = -FLT_MAX;
        for (int local_k = threadIdx.x; local_k < tile_size; local_k += blockDim.x) {
            tile_max = fmaxf(tile_max, shared_scores[local_k]);
        }
        tile_max = block_reduce_max(tile_max, shared_scratch);
        __syncthreads();

        // Broadcast tile_max
        if (threadIdx.x == 0) {
            shared_scratch[0] = tile_max;
        }
        __syncthreads();
        tile_max = shared_scratch[0];

        // Update global max and rescale previous accumulator
        float m_prev = m;
        m = fmaxf(m, tile_max);
        float rescale_factor = expf(m_prev - m);

        // Rescale output accumulator from previous tiles
        for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) {
            output_acc[d_idx] *= rescale_factor;
        }

        // Compute exp(score - m) and sum for this tile
        float tile_sum = 0.0f;
        for (int local_k = threadIdx.x; local_k < tile_size; local_k += blockDim.x) {
            float exp_score = expf(shared_scores[local_k] - m);
            shared_scores[local_k] = exp_score;
            tile_sum += exp_score;
        }
        tile_sum = block_reduce_sum(tile_sum, shared_scratch);
        __syncthreads();

        // Broadcast tile_sum
        if (threadIdx.x == 0) {
            shared_scratch[0] = tile_sum;
        }
        __syncthreads();
        tile_sum = shared_scratch[0];

        // Update global sum
        s = s * rescale_factor + tile_sum;

        // Phase 3: Accumulate attention @ V for this tile
        __syncthreads();
        for (int local_k = 0; local_k < tile_size; local_k++) {
            int k_pos = tile_start + local_k;
            float prob = shared_scores[local_k];
            float v_scale_val = shared_value_scales[local_k];

            int64_t value_row_base = static_cast<int64_t>(k_pos) * value_stride_tokens +
                                     static_cast<int64_t>(h) * value_stride_heads;

            bool used_packed_value = false;
            if constexpr (BITS < 8) {
                if (is_packed) {
                    if (threadIdx.x == 0) {
                        PackedBitReader<BITS> reader(value_base_u8 + value_row_base);
                        for (int d_idx = 0; d_idx < d; d_idx++) {
                            int8_t v_q_val = reader.next();
                            shared_value_tile[d_idx] = static_cast<float>(v_q_val) * v_scale_val;
                        }
                    }
                    used_packed_value = true;
                }
            }

            if (!used_packed_value) {
                const int8_t* v_row_ptr = value_base_i8 + value_row_base;
                if (value_stride_dim == 1) {
                    for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) {
                        int8_t v_q_val = v_row_ptr[d_idx];
                        shared_value_tile[d_idx] = static_cast<float>(v_q_val) * v_scale_val;
                    }
                } else {
                    for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) {
                        int8_t v_q_val = v_row_ptr[d_idx * value_stride_dim];
                        shared_value_tile[d_idx] = static_cast<float>(v_q_val) * v_scale_val;
                    }
                }
            }
            __syncthreads();

            for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) {
                output_acc[d_idx] += prob * shared_value_tile[d_idx];
            }
            __syncthreads();
        }
    }

    // Write unnormalized output and softmax statistics
    __syncthreads();
    for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) {
        int out_offset = ((b * H + h) * q_len + q_pos) * d + d_idx;
        output[out_offset] = output_acc[d_idx];  // UNNORMALIZED for cross-bucket accumulation
    }

    // Thread 0 writes softmax statistics
    if (threadIdx.x == 0) {
        int stats_offset = (b * H + h) * q_len + q_pos;
        m_out[stats_offset] = m;
        s_out[stats_offset] = s;
    }
}

// Helper to launch bucket kernel with appropriate template instantiation
template<int TILE_SIZE = 64>
void launch_bucket_kernel(
    int bits,
    torch::Tensor query,
    torch::Tensor key_qx,
    torch::Tensor key_scale,
    torch::Tensor value_qx,
    torch::Tensor value_scale,
    torch::optional<torch::Tensor> attention_mask,
    torch::Tensor global_slots,
    torch::Tensor output,
    torch::Tensor m_out,
    torch::Tensor s_out,
    int packed_dim,
    bool is_packed,
    int full_k_len
) {
    const auto B = query.size(0);
    const auto H = query.size(1);
    const auto q_len = query.size(2);
    const auto d = query.size(3);
    const auto k_len = key_qx.size(0);

    // Kernel launch configuration
    dim3 grid(B, H, q_len);
    int block_size = std::min(256, std::max(32, static_cast<int>(k_len)));
    block_size = ((block_size + 31) / 32) * 32;
    dim3 block(block_size);

    // Shared memory: query + output accumulator + scores + scales + scratch + value tile + slots
    size_t shared_mem_floats = static_cast<size_t>(3 * d + 3 * TILE_SIZE + 32);
    size_t shared_mem_ints = static_cast<size_t>(TILE_SIZE);
    size_t shared_mem_size = shared_mem_floats * sizeof(float) + shared_mem_ints * sizeof(int32_t);

    const float* query_ptr = query.data_ptr<float>();
    const void* key_qx_ptr = key_qx.data_ptr();
    const float* key_scale_ptr = key_scale.data_ptr<float>();
    const void* value_qx_ptr = value_qx.data_ptr();
    const float* value_scale_ptr = value_scale.data_ptr<float>();
    const float* mask_ptr = attention_mask.has_value() ? attention_mask.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();
    const int64_t* slots_ptr = global_slots.defined() ? global_slots.data_ptr<int64_t>() : nullptr;
    float* m_ptr = m_out.data_ptr<float>();
    float* s_ptr = s_out.data_ptr<float>();

    // Extract strides (in elements) for packed/int8 layouts
    auto key_strides = key_qx.strides();
    auto value_strides = value_qx.strides();

    int64_t key_elem_size = key_qx.element_size();
    int64_t value_elem_size = value_qx.element_size();

    int64_t key_stride_tokens = key_strides.size() > 0 ? key_strides[0] * key_elem_size : 0;
    int64_t key_stride_heads = key_strides.size() > 1 ? key_strides[1] * key_elem_size : 0;
    int64_t key_stride_dim = key_strides.size() > 2 ? key_strides[2] * key_elem_size : key_elem_size;

    int64_t value_stride_tokens = value_strides.size() > 0 ? value_strides[0] * value_elem_size : 0;
    int64_t value_stride_heads = value_strides.size() > 1 ? value_strides[1] * value_elem_size : 0;
    int64_t value_stride_dim = value_strides.size() > 2 ? value_strides[2] * value_elem_size : value_elem_size;

    // Dispatch to appropriate template based on bits
    if (bits == 2) {
        quantized_attention_bucket_tiled_kernel<2, TILE_SIZE><<<grid, block, shared_mem_size>>>(
            query_ptr, key_qx_ptr, key_scale_ptr, value_qx_ptr, value_scale_ptr,
            mask_ptr, slots_ptr, output_ptr, m_ptr, s_ptr, B, H, q_len, k_len, d,
            packed_dim, is_packed, full_k_len,
            key_stride_tokens, key_stride_heads, key_stride_dim,
            value_stride_tokens, value_stride_heads, value_stride_dim
        );
    } else if (bits == 3) {
        quantized_attention_bucket_tiled_kernel<3, TILE_SIZE><<<grid, block, shared_mem_size>>>(
            query_ptr, key_qx_ptr, key_scale_ptr, value_qx_ptr, value_scale_ptr,
            mask_ptr, slots_ptr, output_ptr, m_ptr, s_ptr, B, H, q_len, k_len, d,
            packed_dim, is_packed, full_k_len,
            key_stride_tokens, key_stride_heads, key_stride_dim,
            value_stride_tokens, value_stride_heads, value_stride_dim
        );
    } else if (bits == 4) {
        quantized_attention_bucket_tiled_kernel<4, TILE_SIZE><<<grid, block, shared_mem_size>>>(
            query_ptr, key_qx_ptr, key_scale_ptr, value_qx_ptr, value_scale_ptr,
            mask_ptr, slots_ptr, output_ptr, m_ptr, s_ptr, B, H, q_len, k_len, d,
            packed_dim, is_packed, full_k_len,
            key_stride_tokens, key_stride_heads, key_stride_dim,
            value_stride_tokens, value_stride_heads, value_stride_dim
        );
    } else if (bits == 8) {
        quantized_attention_bucket_tiled_kernel<8, TILE_SIZE><<<grid, block, shared_mem_size>>>(
            query_ptr, key_qx_ptr, key_scale_ptr, value_qx_ptr, value_scale_ptr,
            mask_ptr, slots_ptr, output_ptr, m_ptr, s_ptr, B, H, q_len, k_len, d,
            packed_dim, is_packed, full_k_len,
            key_stride_tokens, key_stride_heads, key_stride_dim,
            value_stride_tokens, value_stride_heads, value_stride_dim
        );
    } else {
        AT_ERROR("Unsupported bit-width: ", bits);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("quantized_attention_bucket_tiled_kernel failed: ", cudaGetErrorString(err));
    }
}

// PyTorch wrapper for bucketed attention (returns unnormalized output + statistics)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantized_attention_bucket_forward(
    torch::Tensor query,
    torch::Tensor key_qx,
    torch::Tensor key_scale,
    torch::Tensor value_qx,
    torch::Tensor value_scale,
    torch::Tensor global_slots,
    int bits,
    int packed_dim,
    bool is_packed,
    torch::optional<torch::Tensor> attention_mask,
    int full_k_len
) {
    // Input validation
    TORCH_CHECK(query.is_cuda(), "query must be on CUDA");
    TORCH_CHECK(key_qx.is_cuda(), "key_qx must be on CUDA");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key_qx.is_contiguous(), "key_qx must be contiguous");
    TORCH_CHECK(key_scale.is_contiguous(), "key_scale must be contiguous");
    TORCH_CHECK(value_qx.is_contiguous(), "value_qx must be contiguous");
    TORCH_CHECK(value_scale.is_contiguous(), "value_scale must be contiguous");
    TORCH_CHECK(global_slots.dtype() == torch::kLong, "global_slots must be int64 tensor");
    TORCH_CHECK(global_slots.is_contiguous(), "global_slots must be contiguous");

    const auto B = query.size(0);
    const auto H = query.size(1);
    const auto q_len = query.size(2);
    const auto d = query.size(3);
    const auto k_len = key_qx.size(0);
    TORCH_CHECK(global_slots.numel() == k_len,
                "global_slots must have same length as bucket tokens");

    // If full_k_len not provided, default to bucket size
    if (full_k_len <= 0) {
        full_k_len = k_len;
    }

    // Allocate output (unnormalized) and softmax statistics without redundant zero fill
    auto output = torch::empty_like(query);
    auto m_out = torch::empty({B, H, q_len}, query.options());
    auto s_out = torch::empty({B, H, q_len}, query.options());

    // Launch kernel for this bucket
    launch_bucket_kernel(bits, query, key_qx, key_scale, value_qx, value_scale,
                         attention_mask, global_slots, output, m_out, s_out,
                         packed_dim, is_packed, full_k_len);

    return std::make_tuple(output, m_out, s_out);
}
