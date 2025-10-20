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

/**
 * Bucket-aware fused attention kernel
 *
 * Computes attention with on-the-fly unpacking and dequantization for a single bucket.
 * Supports tiled execution with streaming softmax for long contexts.
 * Returns unnormalized output + softmax statistics for cross-bucket accumulation.
 *
 * @param query: [B, H, q_len, d] float32
 * @param key_qx: [num_tokens, H, packed_dim] uint8 (if packed) or int8 (if not)
 * @param key_scale: [num_tokens, H] float32 (per-head scales)
 * @param value_qx: [num_tokens, H, packed_dim] uint8 (if packed) or int8 (if not)
 * @param value_scale: [num_tokens, H] float32 (per-head scales)
 * @param bits: bit-width for this bucket (2, 3, 4, or 8)
 * @param packed_dim: dimension of packed data (d for int8, computed size for packed)
 * @param is_packed: true if data is bit-packed
 * @param attention_mask: Optional [B, 1, q_len, k_len] float32
 * @param full_k_len: Full context length for mask stride
 *
 * @return: tuple of (unnormalized_output [B, H, q_len, d], m [B, H, q_len], s [B, H, q_len])
 */
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
);
