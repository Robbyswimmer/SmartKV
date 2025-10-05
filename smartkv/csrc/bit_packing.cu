/*
 * SmartKV Bit-Packing Kernels - Implementation
 *
 * Efficient CUDA kernels for packing/unpacking 2/3/4-bit quantized values.
 * Achieves true memory savings vs storing everything as INT8.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "bit_packing.h"

namespace {

// ============================================================================
// 2-BIT PACKING (4 values per byte)
// Range: [-2, 1] mapped to [0, 3]
// ============================================================================

__global__ void pack_2bit_kernel(
    const int8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t num_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t byte_idx = idx / 4;  // 4 values per byte

    if (byte_idx >= (num_elements + 3) / 4) return;

    // Pack 4 values into 1 byte
    int64_t base_idx = byte_idx * 4;
    uint8_t packed = 0;

    for (int i = 0; i < 4 && base_idx + i < num_elements; i++) {
        int8_t val = input[base_idx + i];
        uint8_t unsigned_val = (uint8_t)(val + 2);  // [-2,1] -> [0,3]
        packed |= (unsigned_val & 0x3) << (i * 2);
    }

    if (threadIdx.x % 4 == 0) {
        output[byte_idx] = packed;
    }
}

__global__ void unpack_2bit_kernel(
    const uint8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t num_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_elements) return;

    int64_t byte_idx = idx / 4;
    int bit_offset = (idx % 4) * 2;

    uint8_t packed = input[byte_idx];
    uint8_t unsigned_val = (packed >> bit_offset) & 0x3;
    int8_t val = (int8_t)(unsigned_val - 2);  // [0,3] -> [-2,1]

    output[idx] = val;
}

// ============================================================================
// 3-BIT PACKING (8 values per 3 bytes)
// Range: [-4, 3] mapped to [0, 7]
// ============================================================================

__global__ void pack_3bit_kernel(
    const int8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t num_elements
) {
    int64_t group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t base_idx = group_idx * 8;  // 8 values per group

    if (base_idx >= num_elements) return;

    // Pack 8x 3-bit values into 3 bytes (24 bits total)
    uint32_t packed = 0;

    for (int i = 0; i < 8 && base_idx + i < num_elements; i++) {
        int8_t val = input[base_idx + i];
        uint8_t unsigned_val = (uint8_t)(val + 4);  // [-4,3] -> [0,7]
        packed |= (uint32_t)(unsigned_val & 0x7) << (i * 3);
    }

    // Write 3 bytes
    int64_t byte_idx = (base_idx / 8) * 3;
    if (byte_idx < ((num_elements + 7) / 8) * 3) {
        output[byte_idx + 0] = (uint8_t)(packed & 0xFF);
        output[byte_idx + 1] = (uint8_t)((packed >> 8) & 0xFF);
        output[byte_idx + 2] = (uint8_t)((packed >> 16) & 0xFF);
    }
}

__global__ void unpack_3bit_kernel(
    const uint8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t num_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_elements) return;

    // Calculate byte position
    int64_t bit_pos = idx * 3;
    int64_t byte_idx = bit_pos / 8;
    int bit_offset = bit_pos % 8;

    // Read value spanning up to 2 bytes
    uint32_t bytes = (uint32_t)input[byte_idx];
    if (bit_offset > 5 && byte_idx + 1 < ((num_elements + 7) / 8) * 3) {
        bytes |= (uint32_t)input[byte_idx + 1] << 8;
    }

    uint8_t unsigned_val = (bytes >> bit_offset) & 0x7;
    int8_t val = (int8_t)(unsigned_val - 4);  // [0,7] -> [-4,3]

    output[idx] = val;
}

// ============================================================================
// 4-BIT PACKING (2 values per byte)
// Range: [-8, 7] mapped to [0, 15]
// ============================================================================

__global__ void pack_4bit_kernel(
    const int8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t num_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t byte_idx = idx / 2;  // 2 values per byte

    if (byte_idx >= (num_elements + 1) / 2) return;

    // Pack 2 values into 1 byte
    int64_t base_idx = byte_idx * 2;
    uint8_t packed = 0;

    for (int i = 0; i < 2 && base_idx + i < num_elements; i++) {
        int8_t val = input[base_idx + i];
        uint8_t unsigned_val = (uint8_t)(val + 8);  // [-8,7] -> [0,15]
        packed |= (unsigned_val & 0xF) << (i * 4);
    }

    if (threadIdx.x % 2 == 0) {
        output[byte_idx] = packed;
    }
}

__global__ void unpack_4bit_kernel(
    const uint8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t num_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_elements) return;

    int64_t byte_idx = idx / 2;
    int bit_offset = (idx % 2) * 4;

    uint8_t packed = input[byte_idx];
    uint8_t unsigned_val = (packed >> bit_offset) & 0xF;
    int8_t val = (int8_t)(unsigned_val - 8);  // [0,15] -> [-8,7]

    output[idx] = val;
}

} // namespace

// ============================================================================
// PyTorch Wrappers
// ============================================================================

torch::Tensor pack_2bit(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt8, "input must be int8");

    int64_t num_elements = input.numel();
    int64_t num_bytes = (num_elements + 3) / 4;  // 4 values per byte

    auto output = torch::empty({num_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    pack_2bit_kernel<<<blocks, threads>>>(
        input.data_ptr<int8_t>(),
        output.data_ptr<uint8_t>(),
        num_elements
    );

    return output;
}

torch::Tensor unpack_2bit(torch::Tensor packed, std::vector<int64_t> shape) {
    TORCH_CHECK(packed.is_cuda(), "packed must be on CUDA");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");

    int64_t num_elements = 1;
    for (auto s : shape) num_elements *= s;

    auto output = torch::empty(shape, torch::TensorOptions().dtype(torch::kInt8).device(packed.device()));

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    unpack_2bit_kernel<<<blocks, threads>>>(
        packed.data_ptr<uint8_t>(),
        output.data_ptr<int8_t>(),
        num_elements
    );

    return output;
}

torch::Tensor pack_3bit(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt8, "input must be int8");

    int64_t num_elements = input.numel();
    int64_t num_bytes = ((num_elements + 7) / 8) * 3;  // 8 values per 3 bytes

    auto output = torch::empty({num_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));

    int threads = 256;
    int blocks = ((num_elements + 7) / 8 + threads - 1) / threads;

    pack_3bit_kernel<<<blocks, threads>>>(
        input.data_ptr<int8_t>(),
        output.data_ptr<uint8_t>(),
        num_elements
    );

    return output;
}

torch::Tensor unpack_3bit(torch::Tensor packed, std::vector<int64_t> shape) {
    TORCH_CHECK(packed.is_cuda(), "packed must be on CUDA");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");

    int64_t num_elements = 1;
    for (auto s : shape) num_elements *= s;

    auto output = torch::empty(shape, torch::TensorOptions().dtype(torch::kInt8).device(packed.device()));

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    unpack_3bit_kernel<<<blocks, threads>>>(
        packed.data_ptr<uint8_t>(),
        output.data_ptr<int8_t>(),
        num_elements
    );

    return output;
}

torch::Tensor pack_4bit(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt8, "input must be int8");

    int64_t num_elements = input.numel();
    int64_t num_bytes = (num_elements + 1) / 2;  // 2 values per byte

    auto output = torch::empty({num_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    pack_4bit_kernel<<<blocks, threads>>>(
        input.data_ptr<int8_t>(),
        output.data_ptr<uint8_t>(),
        num_elements
    );

    return output;
}

torch::Tensor unpack_4bit(torch::Tensor packed, std::vector<int64_t> shape) {
    TORCH_CHECK(packed.is_cuda(), "packed must be on CUDA");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");

    int64_t num_elements = 1;
    for (auto s : shape) num_elements *= s;

    auto output = torch::empty(shape, torch::TensorOptions().dtype(torch::kInt8).device(packed.device()));

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    unpack_4bit_kernel<<<blocks, threads>>>(
        packed.data_ptr<uint8_t>(),
        output.data_ptr<int8_t>(),
        num_elements
    );

    return output;
}

// Generic interface
torch::Tensor pack_values(torch::Tensor input, int bits) {
    switch (bits) {
        case 2: return pack_2bit(input);
        case 3: return pack_3bit(input);
        case 4: return pack_4bit(input);
        case 8: return input;  // No packing needed for 8-bit
        default:
            TORCH_CHECK(false, "Unsupported bits: ", bits, ". Must be 2, 3, 4, or 8.");
    }
}

torch::Tensor unpack_values(torch::Tensor packed, int bits, std::vector<int64_t> shape) {
    switch (bits) {
        case 2: return unpack_2bit(packed, shape);
        case 3: return unpack_3bit(packed, shape);
        case 4: return unpack_4bit(packed, shape);
        case 8: return packed.reshape(shape);  // No unpacking needed
        default:
            TORCH_CHECK(false, "Unsupported bits: ", bits, ". Must be 2, 3, 4, or 8.");
    }
}
