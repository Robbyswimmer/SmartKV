"""
Quantizers for KV-cache compression.

Implements 2-bit, 3-bit, 4-bit, and 8-bit symmetric quantization schemes
for compressing key and value vectors in transformer KV-cache.
"""

import torch
from typing import Dict, Any
from abc import ABC, abstractmethod


class QuantizerBase(ABC):
    """
    Base class for quantizers.
    
    All quantizers implement symmetric quantization where the quantization
    range is centered at zero: [-max_val, max_val] -> [-2^(bits-1), 2^(bits-1)-1]
    """
    
    @abstractmethod
    def quantize(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Quantize a tensor.
        
        Args:
            tensor: Input tensor to quantize (typically FP16 or FP32)
            
        Returns:
            Dictionary containing:
                - 'qx': Quantized tensor (int8)
                - 'scale': Scale factor for dequantization
                - Additional metadata as needed
        """
        raise NotImplementedError
    
    @abstractmethod
    def dequantize(self, qdata: Dict[str, Any]) -> torch.Tensor:
        """
        Dequantize a tensor.
        
        Args:
            qdata: Dictionary from quantize() containing quantized data
            
        Returns:
            Dequantized tensor (FP32)
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def bits(self) -> int:
        """Number of bits used for quantization."""
        raise NotImplementedError


class EightbitQuantizer(QuantizerBase):
    """
    8-bit symmetric quantization (INT8).
    
    Quantization range: [-128, 127]
    This is the standard INT8 quantization commonly used as a baseline.
    """
    
    @property
    def bits(self) -> int:
        return 8
    
    def quantize(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Quantize to INT8.
        
        Formula: q = clamp(round(x / scale), -128, 127)
        where scale = max(|x|) / 127
        """
        # Per-tensor quantization (single scale for entire tensor)
        scale = x.abs().max() / 127.0
        
        # Handle zero tensors
        if scale == 0:
            scale = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        
        # Quantize: scale, round, clamp
        qx = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
        
        return {
            'qx': qx,
            'scale': scale,
            'dtype': x.dtype,
            'shape': x.shape
        }
    
    def dequantize(self, qdata: Dict[str, Any]) -> torch.Tensor:
        """
        Dequantize from INT8.
        
        Formula: x = q * scale
        """
        return qdata['qx'].float() * qdata['scale']


class FourbitQuantizer(QuantizerBase):
    """
    4-bit symmetric quantization (INT4).
    
    Quantization range: [-8, 7]
    Stored as INT8 but only uses 4 bits of precision.
    """
    
    @property
    def bits(self) -> int:
        return 4
    
    def quantize(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Quantize to INT4 (stored as INT8).
        
        Formula: q = clamp(round(x / scale), -8, 7)
        where scale = max(|x|) / 7
        """
        scale = x.abs().max() / 7.0
        
        if scale == 0:
            scale = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        
        qx = torch.clamp(torch.round(x / scale), -8, 7).to(torch.int8)
        
        return {
            'qx': qx,
            'scale': scale,
            'dtype': x.dtype,
            'shape': x.shape
        }
    
    def dequantize(self, qdata: Dict[str, Any]) -> torch.Tensor:
        """Dequantize from INT4."""
        return qdata['qx'].float() * qdata['scale']


class ThreebitQuantizer(QuantizerBase):
    """
    3-bit symmetric quantization.
    
    Quantization range: [-4, 3]
    Stored as INT8 but only uses 3 bits of precision.
    """
    
    @property
    def bits(self) -> int:
        return 3
    
    def quantize(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Quantize to 3-bit (stored as INT8).
        
        Formula: q = clamp(round(x / scale), -4, 3)
        where scale = max(|x|) / 3
        """
        scale = x.abs().max() / 3.0
        
        if scale == 0:
            scale = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        
        qx = torch.clamp(torch.round(x / scale), -4, 3).to(torch.int8)
        
        return {
            'qx': qx,
            'scale': scale,
            'dtype': x.dtype,
            'shape': x.shape
        }
    
    def dequantize(self, qdata: Dict[str, Any]) -> torch.Tensor:
        """Dequantize from 3-bit."""
        return qdata['qx'].float() * qdata['scale']


class TwobitQuantizer(QuantizerBase):
    """
    2-bit symmetric quantization.
    
    Quantization range: [-2, 1]
    Stored as INT8 but only uses 2 bits of precision.
    Most aggressive quantization, used for least important tokens.
    """
    
    @property
    def bits(self) -> int:
        return 2
    
    def quantize(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Quantize to 2-bit (stored as INT8).
        
        Formula: q = clamp(round(x / scale), -2, 1)
        where scale = max(|x|) / 1
        """
        scale = x.abs().max() / 1.0
        
        if scale == 0:
            scale = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        
        qx = torch.clamp(torch.round(x / scale), -2, 1).to(torch.int8)
        
        return {
            'qx': qx,
            'scale': scale,
            'dtype': x.dtype,
            'shape': x.shape
        }
    
    def dequantize(self, qdata: Dict[str, Any]) -> torch.Tensor:
        """Dequantize from 2-bit."""
        return qdata['qx'].float() * qdata['scale']


# Factory function for convenience
def get_quantizer(bits: int) -> QuantizerBase:
    """
    Get quantizer instance by bit-width.
    
    Args:
        bits: Number of bits (2, 3, 4, or 8)
        
    Returns:
        Quantizer instance
        
    Raises:
        ValueError: If bits not in [2, 3, 4, 8]
    """
    quantizer_map = {
        2: TwobitQuantizer,
        3: ThreebitQuantizer,
        4: FourbitQuantizer,
        8: EightbitQuantizer,
    }
    
    if bits not in quantizer_map:
        raise ValueError(f"Unsupported bit-width: {bits}. Must be one of {list(quantizer_map.keys())}")
    
    return quantizer_map[bits]()
