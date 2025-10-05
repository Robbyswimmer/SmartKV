"""
SmartKV Kernel Interface

High-level Python interface for CUDA/Triton quantized attention kernels.
Automatically selects best kernel based on hardware and context length.
"""

import torch
from typing import Optional, Tuple
import warnings


# Try to import CUDA extension
try:
    import smartkv_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn(
        "SmartKV CUDA kernels not available. Install with: pip install -e '.[gpu]' "
        "Falling back to PyTorch implementation (slower)."
    )


def quantized_attention(
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    use_cuda: bool = True
) -> torch.Tensor:
    """
    Fused quantized attention with on-the-fly dequantization.

    Computes: output = softmax(Q @ K^T / sqrt(d)) @ V
    where K and V are quantized (int8) with per-head scales.

    Args:
        query: [B, H, q_len, d] float32 query tensor
        key_int8: [B, H, k_len, d] int8 quantized keys
        key_scale: [B, H, k_len] float32 per-head key scales
        value_int8: [B, H, k_len, d] int8 quantized values
        value_scale: [B, H, k_len] float32 per-head value scales
        attention_mask: Optional [B, 1, q_len, k_len] float32 mask
        use_cuda: Whether to use CUDA kernel (if available)

    Returns:
        output: [B, H, q_len, d] float32 attention output

    Example:
        >>> q = torch.randn(1, 8, 1, 128, device='cuda')
        >>> k_int8 = torch.randint(-128, 127, (1, 8, 500, 128), dtype=torch.int8, device='cuda')
        >>> k_scale = torch.randn(1, 8, 500, device='cuda')
        >>> v_int8 = torch.randint(-128, 127, (1, 8, 500, 128), dtype=torch.int8, device='cuda')
        >>> v_scale = torch.randn(1, 8, 500, device='cuda')
        >>> output = quantized_attention(q, k_int8, k_scale, v_int8, v_scale)
    """
    # Input validation
    if query.ndim != 4:
        raise ValueError(f"query must be 4D [B, H, q_len, d], got shape {query.shape}")
    if key_int8.ndim != 4:
        raise ValueError(f"key_int8 must be 4D [B, H, k_len, d], got shape {key_int8.shape}")

    # Device check
    device = query.device

    # CUDA kernel path
    if use_cuda and CUDA_AVAILABLE and device.type == 'cuda':
        return smartkv_cuda.quantized_attention_forward(
            query.contiguous(),
            key_int8.contiguous(),
            key_scale.contiguous(),
            value_int8.contiguous(),
            value_scale.contiguous(),
            attention_mask.contiguous() if attention_mask is not None else None
        )

    # PyTorch fallback (dequantize then standard attention)
    return _quantized_attention_pytorch(
        query, key_int8, key_scale, value_int8, value_scale, attention_mask
    )


def _quantized_attention_pytorch(
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    PyTorch fallback implementation of quantized attention.

    Slower than CUDA kernel but works on any device.
    """
    # Dequantize keys and values
    key = key_int8.float() * key_scale.unsqueeze(-1)
    value = value_int8.float() * value_scale.unsqueeze(-1)

    # Standard scaled dot-product attention
    d = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d ** 0.5)

    if attention_mask is not None:
        scores = scores + attention_mask

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output


def check_cuda_availability() -> Tuple[bool, str]:
    """
    Check if CUDA kernels are available.

    Returns:
        Tuple of (available, message)
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available on this system"

    if not CUDA_AVAILABLE:
        return False, "SmartKV CUDA extension not compiled. Install with: pip install -e '.[gpu]'"

    try:
        # Test kernel with small input
        test_q = torch.randn(1, 1, 1, 64, device='cuda')
        test_k = torch.randint(-128, 127, (1, 1, 10, 64), dtype=torch.int8, device='cuda')
        test_k_scale = torch.randn(1, 1, 10, device='cuda')
        test_v = torch.randint(-128, 127, (1, 1, 10, 64), dtype=torch.int8, device='cuda')
        test_v_scale = torch.randn(1, 1, 10, device='cuda')

        _ = smartkv_cuda.quantized_attention_forward(
            test_q, test_k, test_k_scale, test_v, test_v_scale, None
        )

        return True, "SmartKV CUDA kernels are available and functional"
    except Exception as e:
        return False, f"CUDA kernels failed self-test: {str(e)}"


# Public API
__all__ = [
    'quantized_attention',
    'check_cuda_availability',
    'CUDA_AVAILABLE',
]
