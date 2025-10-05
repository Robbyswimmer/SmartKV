/*
 * SmartKV CUDA Kernels - Header
 *
 * Fused quantized attention kernels for SmartKV mixed-precision KV-cache.
 */

#pragma once

#include <torch/extension.h>
#include <vector>

// Forward declarations for CUDA kernels

/**
 * Fused quantized attention kernel (minimal version)
 *
 * Computes attention with on-the-fly dequantization of K/V cache.
 *
 * @param query: [B, H, q_len, d] float32
 * @param key_int8: [B, H, k_len, d] int8 (quantized keys)
 * @param key_scale: [B, H, k_len] float32 (per-head scales)
 * @param value_int8: [B, H, k_len, d] int8 (quantized values)
 * @param value_scale: [B, H, k_len] float32 (per-head scales)
 * @param attention_mask: Optional [B, 1, q_len, k_len] float32
 *
 * @return: attention output [B, H, q_len, d] float32
 */
torch::Tensor quantized_attention_forward(
    torch::Tensor query,
    torch::Tensor key_int8,
    torch::Tensor key_scale,
    torch::Tensor value_int8,
    torch::Tensor value_scale,
    torch::optional<torch::Tensor> attention_mask
);

/**
 * Per-head quantization kernel
 *
 * Quantizes K/V tensors with per-head scale factors.
 *
 * @param input: [N, H, D] float32
 * @param bits: target bit-width (2, 3, 4, or 8)
 *
 * @return: tuple of (quantized [N, H, D] int8, scales [N, H] float32)
 */
std::tuple<torch::Tensor, torch::Tensor> quantize_per_head_forward(
    torch::Tensor input,
    int bits
);
