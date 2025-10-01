"""
SmartKV: Attention-Guided Adaptive Precision for KV-Cache Compression

A PyTorch implementation of dynamic precision allocation for LLM KV-cache
compression based on attention patterns.
"""

__version__ = "0.1.0"
__author__ = "SmartKV Team"

from smartkv.core.cache import SmartKVCache
from smartkv.core.quantizers import (
    QuantizerBase,
    TwobitQuantizer,
    ThreebitQuantizer,
    FourbitQuantizer,
    EightbitQuantizer,
)

__all__ = [
    "SmartKVCache",
    "QuantizerBase",
    "TwobitQuantizer",
    "ThreebitQuantizer",
    "FourbitQuantizer",
    "EightbitQuantizer",
]
