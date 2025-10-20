"""
CUDA quantization kernels for SmartKV.

This module provides GPU-accelerated per-head quantization for KV-cache compression.
When the `smartkv_cuda` extension is available it launches a custom CUDA kernel that
computes per-head scales and quantized payloads; otherwise it falls back to PyTorch
ops (slower).
"""

import torch
from typing import Tuple

try:
    import smartkv_cuda
except ImportError:  # pragma: no cover - CUDA extension optional
    smartkv_cuda = None


def quantize_per_head_cuda(
    k_subset: torch.Tensor,
    v_subset: torch.Tensor,
    bits: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA-accelerated per-head quantization.

    Args:
        k_subset: Key tensor [N, H, D] on CUDA device
        v_subset: Value tensor [N, H, D] on CUDA device
        bits: Target bit-width (2, 3, 4, or 8)

    Returns:
        Tuple of (k_quantized, v_quantized, k_scale, v_scale)
        - k_quantized: [N, H, D] int8 on CUDA
        - v_quantized: [N, H, D] int8 on CUDA
        - k_scale: [N, H] float32 on CUDA
        - v_scale: [N, H] float32 on CUDA

    Note:
        Uses the CUDA extension when available; otherwise falls back to PyTorch ops.
    """
    if not k_subset.is_cuda or not v_subset.is_cuda:
        raise ValueError("Input tensors must be on CUDA device")

    if smartkv_cuda is not None:
        k_q, k_scale = smartkv_cuda.quantize_per_head_forward(k_subset.contiguous(), bits)
        v_q, v_scale = smartkv_cuda.quantize_per_head_forward(v_subset.contiguous(), bits)
        return k_q, v_q, k_scale, v_scale

    # Fallback: use PyTorch ops (slower but functional)
    max_val = 2 ** (bits - 1) - 1
    min_val = -2 ** (bits - 1)
    if bits == 1:
        max_val = 0
        min_val = 0

    k_abs_max = k_subset.abs().amax(dim=2, keepdim=True)
    k_scale_subset = k_abs_max / max(max_val, 1)
    k_scale_subset = torch.where(k_scale_subset == 0, torch.ones_like(k_scale_subset), k_scale_subset)
    k_q = torch.clamp(torch.round(k_subset / k_scale_subset), min_val, max_val).to(torch.int8)

    v_abs_max = v_subset.abs().amax(dim=2, keepdim=True)
    v_scale_subset = v_abs_max / max(max_val, 1)
    v_scale_subset = torch.where(v_scale_subset == 0, torch.ones_like(v_scale_subset), v_scale_subset)
    v_q = torch.clamp(torch.round(v_subset / v_scale_subset), min_val, max_val).to(torch.int8)

    k_scale_out = k_scale_subset.view(-1, k_subset.shape[1])
    v_scale_out = v_scale_subset.view(-1, v_subset.shape[1])

    return k_q, v_q, k_scale_out, v_scale_out


def pack_values(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if smartkv_cuda is not None:
        return smartkv_cuda.pack_values(tensor, bits)
    if bits == 8:
        return tensor
    return tensor.to(torch.uint8)


def unpack_values(packed: torch.Tensor, bits: int, shape) -> torch.Tensor:
    if smartkv_cuda is not None:
        return smartkv_cuda.unpack_values(packed, bits, list(shape))
    return packed.view(shape).to(torch.int8)


__all__ = [
    "quantize_per_head_cuda",
    "pack_values",
    "unpack_values",
]
