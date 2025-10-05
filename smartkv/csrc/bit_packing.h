/*
 * SmartKV Bit-Packing Kernels - Header
 *
 * Efficient bit-packing for 2/3/4-bit quantized values to achieve
 * true sub-50% memory usage.
 */

#pragma once

#include <torch/extension.h>
#include <vector>

// Pack 2-bit values (4 values per byte)
torch::Tensor pack_2bit(torch::Tensor input);
torch::Tensor unpack_2bit(torch::Tensor packed, std::vector<int64_t> shape);

// Pack 3-bit values (8 values per 3 bytes)
torch::Tensor pack_3bit(torch::Tensor input);
torch::Tensor unpack_3bit(torch::Tensor packed, std::vector<int64_t> shape);

// Pack 4-bit values (2 values per byte)
torch::Tensor pack_4bit(torch::Tensor input);
torch::Tensor unpack_4bit(torch::Tensor packed, std::vector<int64_t> shape);

// Generic pack/unpack interface
torch::Tensor pack_values(torch::Tensor input, int bits);
torch::Tensor unpack_values(torch::Tensor packed, int bits, std::vector<int64_t> shape);
