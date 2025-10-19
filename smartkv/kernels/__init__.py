"""
SmartKV Kernel Interface

High-level Python interface for CUDA/Triton quantized attention kernels.
Automatically selects best kernel based on hardware and context length.
"""

import torch
from typing import Optional, Tuple
import warnings


def _ensure_query_layout(query: torch.Tensor) -> torch.Tensor:
    """Ensure query has shape [B, H, q_len, d]."""
    if query.ndim == 4:
        return query
    if query.ndim == 3:
        # [B, H, d] -> [B, H, 1, d]
        return query.unsqueeze(2)
    raise ValueError(f"query must be 3D or 4D tensor, got shape {query.shape}")


def _ensure_kv_layout(tensor: torch.Tensor, num_heads: int, name: str) -> torch.Tensor:
    """Ensure KV tensor has shape [B, H, k_len, d]."""
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be 4D tensor after batching, got shape {tensor.shape}")

    # Common layouts: [B, H, k_len, d] (expected) or [B, k_len, H, d]
    if tensor.size(1) == num_heads:
        return tensor
    if tensor.size(2) == num_heads:
        # Permute from [B, k_len, H, d]
        return tensor.permute(0, 2, 1, 3).contiguous()

    raise ValueError(
        f"{name} has incompatible shape {tensor.shape}; cannot infer head dimension"
    )


def _ensure_scale_layout(scale: torch.Tensor, num_heads: int, k_len: int, name: str) -> torch.Tensor:
    """Ensure scale tensor has shape [B, H, k_len]."""
    if scale.ndim == 3 and scale.size(1) == num_heads:
        return scale
    if scale.ndim == 3 and scale.size(2) == num_heads:
        return scale.permute(0, 2, 1).contiguous()
    if scale.ndim == 2 and scale.size(1) == k_len:
        # [B, k_len] -> broadcast Heads
        return scale.unsqueeze(1).expand(-1, num_heads, -1)
    raise ValueError(f"{name} must broadcast to [B, H, k_len]; got shape {scale.shape}")


def _normalize_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch: int,
    q_len: int,
    k_len: int,
) -> Optional[torch.Tensor]:
    """Convert mask to [B, 1, q_len, k_len] if provided."""
    if attention_mask is None:
        return None

    mask = attention_mask
    if mask.ndim == 2 and mask.shape == (batch, k_len):
        mask = mask.unsqueeze(1).unsqueeze(1)
    elif mask.ndim == 3 and mask.shape[0] == batch and mask.shape[1] in (1, q_len) and mask.shape[2] == k_len:
        if mask.shape[1] == q_len:
            mask = mask.unsqueeze(1)
        else:
            mask = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[0] == batch:
        # Already in [B, ?, ?, k_len]; ensure second dim is 1
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        if mask.shape[2] != q_len:
            raise ValueError(
                f"attention_mask last-but-one dim must match q_len={q_len}; got {mask.shape}"
            )
    else:
        raise ValueError(
            "attention_mask must be broadcastable to [B, 1, q_len, k_len]; "
            f"got shape {attention_mask.shape}"
        )

    return mask.contiguous()


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
    # Normalize layouts
    query = _ensure_query_layout(query)
    device = query.device
    batch, num_heads, q_len, d = query.shape

    key_int8 = _ensure_kv_layout(key_int8, num_heads, "key_int8")
    value_int8 = _ensure_kv_layout(value_int8, num_heads, "value_int8")
    _, _, k_len, _ = key_int8.shape

    key_scale = _ensure_scale_layout(key_scale, num_heads, k_len, "key_scale")
    value_scale = _ensure_scale_layout(value_scale, num_heads, k_len, "value_scale")

    attn_mask = _normalize_attention_mask(attention_mask, batch, q_len, k_len)

    use_cuda_kernel = use_cuda and CUDA_AVAILABLE and device.type == 'cuda'

    if use_cuda_kernel:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        shared_mem_bytes = (d + k_len + 32) * 4  # floats in shared memory

        # Check shared memory limit (attribute name varies by PyTorch version)
        max_shared_mem = getattr(props, 'max_shared_memory_per_block',
                                 getattr(props, 'max_shared_memory_per_block_optin',
                                        getattr(props, 'total_shared_memory_per_block', 49152)))

        if shared_mem_bytes > max_shared_mem:
            warnings.warn(
                f"SmartKV CUDA kernel requires {shared_mem_bytes} bytes shared memory "
                f"but GPU only has {max_shared_mem} bytes per block. "
                "Falling back to PyTorch attention.",
                RuntimeWarning
            )
            use_cuda_kernel = False

    if use_cuda_kernel:
        return smartkv_cuda.quantized_attention_forward(
            query.contiguous(),
            key_int8.contiguous(),
            key_scale.contiguous(),
            value_int8.contiguous(),
            value_scale.contiguous(),
            attn_mask
        )

    # PyTorch fallback (dequantize then standard attention)
    return _quantized_attention_pytorch(
        query, key_int8, key_scale, value_int8, value_scale, attn_mask
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
