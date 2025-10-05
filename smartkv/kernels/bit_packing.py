"""
Bit-Packing Utilities for SmartKV

High-level Python interface for efficient bit-packing of 2/3/4-bit quantized values.
Achieves true sub-50% memory usage by packing values tightly.
"""

import torch
from typing import Tuple
import warnings


# Try to import CUDA extension
try:
    import smartkv_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def pack_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack int8 tensor to specified bit-width.

    Args:
        tensor: int8 tensor to pack
        bits: Target bit-width (2, 3, or 4). 8 returns unchanged.

    Returns:
        Packed uint8 tensor (fewer bytes)

    Memory savings:
        - 2-bit: 4× compression (4 values per byte)
        - 3-bit: 2.67× compression (8 values per 3 bytes)
        - 4-bit: 2× compression (2 values per byte)
        - 8-bit: No compression (returns copy)

    Example:
        >>> x = torch.randint(-4, 3, (100,), dtype=torch.int8, device='cuda')
        >>> packed = pack_tensor(x, bits=3)
        >>> print(f"Original: {x.nbytes} bytes, Packed: {packed.nbytes} bytes")
        Original: 100 bytes, Packed: 38 bytes
    """
    if tensor.dtype != torch.int8:
        raise ValueError(f"Input must be int8, got {tensor.dtype}")

    if bits == 8:
        return tensor.contiguous()

    if bits not in [2, 3, 4]:
        raise ValueError(f"bits must be 2, 3, 4, or 8, got {bits}")

    # CUDA path
    if CUDA_AVAILABLE and tensor.device.type == 'cuda':
        return smartkv_cuda.pack_values(tensor.contiguous(), bits)

    # CPU fallback (store as uint8, no actual packing)
    warnings.warn(
        "CUDA not available for bit-packing. Storing as uint8 without compression. "
        "Install CUDA extension for true memory savings."
    )
    return tensor.to(torch.uint8)


def unpack_tensor(packed: torch.Tensor, bits: int, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Unpack bit-packed tensor back to int8.

    Args:
        packed: Packed uint8 tensor
        bits: Bit-width used for packing (2, 3, or 4)
        shape: Original tensor shape to restore

    Returns:
        Unpacked int8 tensor with specified shape

    Example:
        >>> x = torch.randint(-4, 3, (8, 16, 128), dtype=torch.int8, device='cuda')
        >>> packed = pack_tensor(x, bits=3)
        >>> restored = unpack_tensor(packed, bits=3, shape=(8, 16, 128))
        >>> assert torch.equal(x, restored)
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"Packed tensor must be uint8, got {packed.dtype}")

    if bits == 8:
        return packed.view(shape).to(torch.int8)

    if bits not in [2, 3, 4]:
        raise ValueError(f"bits must be 2, 3, 4, or 8, got {bits}")

    # CUDA path
    if CUDA_AVAILABLE and packed.device.type == 'cuda':
        return smartkv_cuda.unpack_values(packed, bits, list(shape))

    # CPU fallback
    return packed.view(shape).to(torch.int8)


def compute_packed_size(num_elements: int, bits: int) -> int:
    """
    Compute packed tensor size in bytes.

    Args:
        num_elements: Number of values to pack
        bits: Bit-width (2, 3, 4, or 8)

    Returns:
        Number of bytes required for packed storage
    """
    if bits == 2:
        return (num_elements + 3) // 4
    elif bits == 3:
        return ((num_elements + 7) // 8) * 3
    elif bits == 4:
        return (num_elements + 1) // 2
    elif bits == 8:
        return num_elements
    else:
        raise ValueError(f"bits must be 2, 3, 4, or 8, got {bits}")


def get_compression_ratio(bits: int) -> float:
    """
    Get theoretical compression ratio vs INT8 storage.

    Args:
        bits: Bit-width (2, 3, 4, or 8)

    Returns:
        Compression ratio (e.g., 4.0 for 2-bit = 4× compression)
    """
    if bits == 2:
        return 4.0
    elif bits == 3:
        return 8.0 / 3.0  # ≈2.67
    elif bits == 4:
        return 2.0
    elif bits == 8:
        return 1.0
    else:
        raise ValueError(f"bits must be 2, 3, 4, or 8, got {bits}")


def get_memory_savings_vs_fp16(bits: int) -> float:
    """
    Get memory savings vs FP16 baseline.

    Args:
        bits: Bit-width (2, 3, 4, or 8)

    Returns:
        Fraction of FP16 memory (e.g., 0.125 for 2-bit = 12.5% of FP16)
    """
    if bits == 2:
        return 0.125  # 2/16
    elif bits == 3:
        return 0.1875  # 3/16
    elif bits == 4:
        return 0.25  # 4/16
    elif bits == 8:
        return 0.5  # 8/16
    else:
        raise ValueError(f"bits must be 2, 3, 4, or 8, got {bits}")


# Public API
__all__ = [
    'pack_tensor',
    'unpack_tensor',
    'compute_packed_size',
    'get_compression_ratio',
    'get_memory_savings_vs_fp16',
    'CUDA_AVAILABLE',
]
