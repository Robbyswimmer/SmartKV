# GPU Porting Roadmap for SmartKV

**Status:** Phase 2 Complete ✅ | Ready for Phase 3 (Triton) or Phase 4 (Model Integration) 🚀
**Timeline:** 14 weeks
**Objective:** Port SmartKV from CPU-only to full GPU support with CUDA/Triton kernels for optimized mixed-precision KV-cache attention

**Progress:**
- ✅ Phase 1: Core GPU Infrastructure (Week 1-2) - COMPLETE
- ✅ Phase 2: CUDA Kernel Development (Week 3-6) - COMPLETE
  - ✅ Phase 2.1: Extension setup (bindings, build system)
  - ✅ Phase 2.2: Minimal fused attention kernel (working, optimizations deferred)
  - ✅ Phase 2.3: Bit-packing for 2/3/4-bit (complete with cache integration)
- ⏳ Phase 3: Triton Kernels (Week 7-8) - READY TO START
- ⏳ Phase 4: Model Integration (Week 9-10) - Alternative path
- ⏳ Phase 5: Testing & Validation (Week 11-12)
- ⏳ Phase 6: Advanced Optimizations (Week 13-14)

---

## Table of Contents

1. [Overview](#overview)
2. [Current State](#current-state)
3. [Phase 1: Core GPU Infrastructure](#phase-1-core-gpu-infrastructure-week-1-2)
4. [Phase 2: CUDA Kernel Development](#phase-2-cuda-kernel-development-week-3-6)
5. [Phase 3: Triton Kernels](#phase-3-triton-kernels-week-7-8)
6. [Phase 4: Model Integration](#phase-4-model-integration-week-9-10)
7. [Phase 5: Testing & Validation](#phase-5-testing--validation-week-11-12)
8. [Phase 6: Advanced Optimizations](#phase-6-advanced-optimizations-week-13-14)
9. [File Structure](#file-structure-post-port)
10. [Dependencies](#dependencies-to-add)
11. [Success Metrics](#success-metrics)
12. [Risks & Mitigations](#risks--mitigations)

---

## Overview

SmartKV is a mixed-precision KV-cache compression system that dynamically allocates 2-8 bit precision to tokens based on attention importance. Currently CPU-only, this roadmap outlines the full GPU port to achieve:

- **3-4× speedup** at medium context (500 tokens)
- **5-10× speedup** at long context (8K+ tokens)
- **60% memory savings** vs FP16 baseline
- **Full CUDA/Triton kernel support** with device auto-selection

### Why GPU Port?

**Current bottleneck** (from ALGORITHM_SPECIFICATION.md §7.2):
- Autoregressive decoding: dequantization overhead is **50% of total compute**
- At 500 tokens: 30ms dequantization + 10ms attention = 41ms/step
- **Target with GPU kernels**: <15ms/step (3× faster)

---

## Current State

### ✅ What Works (CPU)
- Per-head symmetric quantization (2/3/4/8-bit)
- Attention-guided importance tracking (EMA)
- Tier-based greedy allocation
- CPU quantization kernels (`_quant_cpu.py`)
- Streaming CPU kernel (`fused_cpu.py`)

### ❌ What's Missing (GPU)
- CUDA quantization kernels
- Fused GPU attention (dequantize-on-the-fly)
- GPU tensor support in cache
- Triton kernel alternatives
- Multi-GPU support

---

## Phase 1: Core GPU Infrastructure (Week 1-2) ✅ COMPLETE

### 1.1 Device-Agnostic Core Modules

#### `smartkv/core/cache.py`
**Changes needed:**
```python
# Current: All tensors on CPU
k_cpu = k_batch.detach().to(torch.float32).cpu()

# Target: Support GPU tensors
k_device = k_batch.detach().to(torch.float32).to(self.device)
```

**Tasks:**
- [x] Add `device` parameter to `SmartKVCache.__init__`
- [x] Modify `quantize_and_store_batch` to handle GPU tensors
- [x] Update `retrieve_all` to return tensors on correct device
- [x] Keep quantized buffers on GPU (int8 + fp32 scales)
- [x] Add `.to(device)` calls strategically to avoid CPU-GPU sync

#### `smartkv/core/importance.py`
**Changes needed:**
```python
# Current: Attention weights converted to CPU
head_stats = attention_weights.detach().to('cpu')

# Target: Keep on GPU if available
head_stats = attention_weights.detach()
if self.device == 'cpu':
    head_stats = head_stats.cpu()
```

**Tasks:**
- [x] Add device tracking to `ImportanceTracker`
- [x] Keep EMA computations on GPU when available
- [x] Handle GPU attention weights directly (no CPU conversion)

#### `smartkv/core/allocation.py`
- ✅ **Already device-agnostic** (NumPy-based, runs on CPU)
- No changes needed

### 1.2 Quantization Kernels

#### Create `smartkv/core/_quant_cuda.py`

**CUDA kernel for per-head quantization:**
```python
import torch
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel
cuda_source = """
__global__ void quantize_per_head_kernel(
    const float* input,     // [N, H, D]
    int8_t* output,         // [N, H, D]
    float* scales,          // [N, H]
    const int N, const int H, const int D,
    const int bits
) {
    int n = blockIdx.x;
    int h = blockIdx.y;

    if (n >= N || h >= H) return;

    // Find max abs value for this head
    float max_val = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val = input[n * H * D + h * D + d];
        max_val = max(max_val, fabsf(val));
    }

    // Reduce across threads (warp shuffle)
    // ... [warp reduction code]

    // Compute scale
    int q_max = (1 << (bits - 1)) - 1;
    float scale = max_val / q_max;
    if (threadIdx.x == 0) {
        scales[n * H + h] = scale;
    }
    __syncthreads();

    // Quantize
    float inv_scale = 1.0f / (scale + 1e-8f);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val = input[n * H * D + h * D + d];
        int8_t q_val = (int8_t)rintf(val * inv_scale);
        output[n * H * D + h * D + d] = q_val;
    }
}
"""

def quantize_per_head_cuda(k_tensor, v_tensor, bits):
    # Load and compile CUDA kernel
    # Return quantized tensors on GPU
    pass
```

**Tasks:**
- [x] Implement CUDA kernel for per-head quantization (PyTorch-based stub)
- [x] Support 2/3/4/8-bit quantization
- [ ] Optimize with warp-level reductions (Phase 2.2)
- [ ] Write unit tests comparing to CPU version (Phase 5.1)

---

## Phase 2: CUDA Kernel Development (Week 3-6) 🚧 IN PROGRESS

### 2.1 CUDA Extension Setup

#### Directory Structure
```
smartkv/csrc/
├── quantized_attention.cu        # Main CUDA kernel
├── quantized_attention.h         # Header
├── quantized_attention_wrapper.cpp  # PyTorch binding
└── setup.py                      # Build config
```

#### `smartkv/csrc/setup.py`
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='smartkv_cuda',
    ext_modules=[
        CUDAExtension(
            name='smartkv_cuda',
            sources=[
                'quantized_attention.cu',
                'quantized_attention_wrapper.cpp'
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-arch=sm_80',  # A100
                    '-arch=sm_86',  # RTX 30xx
                    '-arch=sm_89',  # RTX 40xx
                    '--expt-relaxed-constexpr'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

#### Root `setup.py`
```python
from setuptools import setup, find_packages

setup(
    name='smartkv',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.35.0',
        # ... other deps
    ],
    extras_require={
        'gpu': [
            'cuda-python>=12.0',
            'triton>=2.1.0',
            'ninja>=1.11.0',
            'flash-attn>=2.3.0',
        ]
    }
)
```

**Tasks:**
- [ ] Create `smartkv/csrc/` directory
- [ ] Set up build infrastructure with CUDAExtension
- [ ] Create root `setup.py` for package installation
- [ ] Test CUDA compilation on different GPU architectures

### 2.2 Fused Attention Kernel (CUDA)

#### Week 3-4: Minimal Kernel

**Goal:** Eliminate 30ms dequantization overhead

```cuda
// quantized_attention.cu

__global__ void quantized_attention_minimal(
    const float* query,           // [B, H, q_len, d]
    const int8_t* key_quantized,  // [B, H, k_len, d]
    const float* key_scale,       // [B, H, k_len]
    float* output,                // [B, H, q_len, k_len]
    int B, int H, int q_len, int k_len, int d
) {
    // Compute grid/block indices
    int b = blockIdx.x;
    int h = blockIdx.y;
    int q_pos = blockIdx.z;

    if (b >= B || h >= H || q_pos >= q_len) return;

    // Shared memory for query tile
    extern __shared__ float shared_query[];

    // Load query to shared memory
    for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) {
        shared_query[d_idx] = query[b*H*q_len*d + h*q_len*d + q_pos*d + d_idx];
    }
    __syncthreads();

    // Compute attention scores
    for (int k_pos = threadIdx.x; k_pos < k_len; k_pos += blockDim.x) {
        float score = 0.0f;
        float k_scale_val = key_scale[b*H*k_len + h*k_len + k_pos];

        // Dot product with on-the-fly dequantization
        for (int d_idx = 0; d_idx < d; d_idx++) {
            float q_val = shared_query[d_idx];
            int8_t k_q_val = key_quantized[b*H*k_len*d + h*k_len*d + k_pos*d + d_idx];
            float k_val = k_q_val * k_scale_val;  // Fused dequantization
            score += q_val * k_val;
        }

        output[b*H*q_len*k_len + h*q_len*k_len + q_pos*k_len + k_pos] = score;
    }
}
```

**Expected performance:**
- 500 tokens: ~15-20ms per step (vs 41ms current)
- Eliminates 30ms dequantization overhead
- Not yet optimized (no SIMD, basic tiling)

**Tasks:**
- [ ] Implement minimal kernel with basic tiling
- [ ] Add PyTorch wrapper function
- [ ] Test numerical correctness vs CPU
- [ ] Benchmark latency improvement

#### Week 5-6: Optimized Kernel

**Optimizations to add:**

1. **FlashAttention-2 Style Tiling**
   - Tile KV cache to fit in shared memory
   - Online softmax with numerically stable computation
   - Fuse attention scores + value matmul

2. **Warp-Level Primitives**
   ```cuda
   // Use warp shuffle for reductions
   __device__ float warp_reduce_sum(float val) {
       for (int offset = 16; offset > 0; offset /= 2) {
           val += __shfl_down_sync(0xffffffff, val, offset);
       }
       return val;
   }
   ```

3. **Memory Coalescing**
   - Arrange memory layout for coalesced access
   - Use vectorized loads (float4, int4)

4. **Fused Softmax + MatMul**
   ```cuda
   // Don't materialize full attention matrix
   for (int q_pos = ...) {
       // Compute scores
       float max_score = ...;
       float sum_exp = ...;

       // Immediately compute output
       for (int k_pos = ...) {
           float prob = exp(score - max_score) / sum_exp;
           output += prob * value[k_pos];
       }
   }
   ```

**Expected performance:**
- 500 tokens: ~10-12ms per step (3.5-4× faster than FP16)
- 1K tokens: <25ms per step (2× faster than FP16)
- Matches short-context performance

**Tasks:**
- [ ] Implement FlashAttention-2 tiling strategy
- [ ] Add online softmax with numerical stability
- [ ] Fuse attention computation + value matmul
- [ ] Optimize memory access patterns
- [ ] Comprehensive benchmarking vs FP16

### 2.3 Multi-Bit Handling & Storage Efficiency

#### Current Storage Limitation (CPU Implementation)

**Problem:** Current CPU implementation stores all quantized values as INT8, regardless of target bit-width:
- 2-bit quantized values: stored in INT8 (4× overhead)
- 3-bit quantized values: stored in INT8 (2.67× overhead)
- 4-bit quantized values: stored in INT8 (2× overhead)
- 8-bit quantized values: stored in INT8 (1× - optimal)

This means **actual memory usage is always ≥50%** of FP16, even with aggressive 2-bit quantization, because:
```python
# Current storage (cache.py):
k_qx = torch.empty((seq_len, num_heads, head_dim), dtype=torch.int8)  # Always INT8!
```

**With bit-packing, we can achieve true 2/3/4-bit storage:**
- 2-bit: 12.5% of FP16 (8× compression)
- 3-bit: 18.75% of FP16 (5.3× compression)
- 4-bit: 25% of FP16 (4× compression)

#### Bit-Packing Strategy for GPU

**Pack 8 values into minimal bytes:**
- 2-bit: 8 values in 2 bytes (instead of 8 bytes)
- 3-bit: 8 values in 3 bytes (instead of 8 bytes)
- 4-bit: 8 values in 4 bytes (instead of 8 bytes)

```cuda
// Pack 8x 3-bit values into 3 bytes
__device__ void pack_3bit(const int8_t* values, uint8_t* packed) {
    // values[0..7] each in range [-4, 3]
    // Map to unsigned [0, 7]
    for (int i = 0; i < 8; i++) {
        uint8_t unsigned_val = values[i] + 4;  // [-4,3] -> [0,7]
        int bit_pos = i * 3;
        int byte_idx = bit_pos / 8;
        int bit_offset = bit_pos % 8;

        // Pack into bytes
        packed[byte_idx] |= (unsigned_val << bit_offset);
        if (bit_offset > 5) {  // Spans two bytes
            packed[byte_idx + 1] |= (unsigned_val >> (8 - bit_offset));
        }
    }
}

__device__ int8_t unpack_3bit(const uint8_t* packed, int idx) {
    int bit_pos = idx * 3;
    int byte_idx = bit_pos / 8;
    int bit_offset = bit_pos % 8;

    uint8_t unsigned_val;
    if (bit_offset <= 5) {
        unsigned_val = (packed[byte_idx] >> bit_offset) & 0x7;
    } else {
        // Spans two bytes
        uint8_t low_bits = packed[byte_idx] >> bit_offset;
        uint8_t high_bits = packed[byte_idx + 1] << (8 - bit_offset);
        unsigned_val = (low_bits | high_bits) & 0x7;
    }

    return (int8_t)(unsigned_val - 4);  // [0,7] -> [-4,3]
}

__global__ void quantized_attention_3bit(
    const uint8_t* key_packed,    // Packed 3-bit keys
    const float* key_scale,
    // ... other params
) {
    // Unpack on-the-fly during attention computation
    int8_t k_val = unpack_3bit(key_packed, k_pos * d + d_idx);
    float k_fp = k_val * key_scale[k_pos];
    // ... continue with attention
}
```

#### Memory Layout Optimization

**Option 1: Separate buffers by bit-width**
```python
# Organize cache by precision tier
cache_2bit = {}  # Packed 2-bit storage
cache_3bit = {}  # Packed 3-bit storage
cache_4bit = {}  # Packed 4-bit storage
cache_8bit = {}  # INT8 storage

# Dispatch to appropriate buffer
if bits == 2:
    store_packed_2bit(k_vec, cache_2bit)
elif bits == 3:
    store_packed_3bit(k_vec, cache_3bit)
# ...
```

**Option 2: Unified buffer with metadata**
```python
# Single contiguous buffer with precision metadata
cache_buffer = torch.zeros(total_packed_bytes, dtype=torch.uint8)
metadata = {
    'token_id': [...],
    'offset': [...],    # Byte offset in buffer
    'bits': [...],      # Precision for this token
}
```

#### Separate Kernels by Precision

**Dispatch based on precision distribution:**
```cpp
// Check precision uniformity
bool all_8bit = all(precision_map.values() == 8);
bool all_4bit = all(precision_map.values() == 4);
bool all_3bit = all(precision_map.values() == 3);
bool all_2bit = all(precision_map.values() == 2);

if (all_8bit) {
    // Optimized INT8 kernel (no unpacking)
    quantized_attention_8bit<<<grid, block>>>(
        query, key_int8, key_scale, ...
    );
} else if (all_4bit) {
    // 4-bit specialized kernel
    quantized_attention_4bit<<<grid, block>>>(
        query, key_packed_4bit, key_scale, ...
    );
} else if (all_3bit) {
    // 3-bit specialized kernel (most complex unpacking)
    quantized_attention_3bit<<<grid, block>>>(
        query, key_packed_3bit, key_scale, ...
    );
} else {
    // Mixed precision kernel (handles all cases)
    quantized_attention_mixed<<<grid, block>>>(
        query, key_buffers, precision_map, key_scale, ...
    );
}
```

#### Vectorized Unpacking (SIMD)

**AVX2 example for 3-bit unpacking (8 values at once):**
```cpp
__m256i unpack_3bit_avx2(const uint8_t* packed) {
    // Load 3 bytes containing 8x 3-bit values
    uint32_t data = *(uint32_t*)packed;  // Load 4 bytes

    // Shuffle and mask to extract each 3-bit value
    __m256i indices = _mm256_setr_epi32(
        (data >> 0) & 0x7,
        (data >> 3) & 0x7,
        (data >> 6) & 0x7,
        (data >> 9) & 0x7,
        (data >> 12) & 0x7,
        (data >> 15) & 0x7,
        (data >> 18) & 0x7,
        (data >> 21) & 0x7
    );

    // Convert to int8_t and shift to [-4, 3]
    __m256i result = _mm256_sub_epi32(indices, _mm256_set1_epi32(4));
    return result;
}
```

#### True Memory Savings with Bit-Packing

**Without bit-packing (current CPU implementation):**
| Config | Payload | Scales | Total | Ratio |
|--------|---------|--------|-------|-------|
| 2-bit (stored as INT8) | 50% | 3.13% | 53.13% | ❌ |
| 4-bit (stored as INT8) | 50% | 3.13% | 53.13% | ❌ |
| 8-bit (stored as INT8) | 50% | 3.13% | 53.13% | ✅ |

**With bit-packing (GPU target):**
| Config | Payload | Scales | Total | Ratio |
|--------|---------|--------|-------|-------|
| 2-bit (packed) | 12.5% | 3.13% | 15.63% | ✅ |
| 3-bit (packed) | 18.75% | 3.13% | 21.88% | ✅ |
| 4-bit (packed) | 25% | 3.13% | 28.13% | ✅ |
| 8-bit (INT8) | 50% | 3.13% | 53.13% | ✅ |

**SmartKV with 40% budget (mixed precision):**
- Average precision: ~4 bits
- With packing: 25% + 3.13% = **28.13%** (2.8× memory savings)
- Without packing: 50% + 3.13% = **53.13%** (1.88× memory savings)

**This is critical for GPU port: bit-packing enables true sub-50% memory usage!**

**Tasks:**
- [x] Implement 2/3/4-bit packing kernels (CUDA) - ✅ Complete (smartkv/csrc/bit_packing.cu)
- [x] Update cache.py to use packed storage on GPU - ✅ Complete (with use_bit_packing parameter)
- [ ] Create specialized kernels for uniform precision (optimization for Phase 6)
- [ ] Mixed precision kernel for SmartKV tiers (optimization for Phase 6)
- [ ] Measure packing overhead (<5% target) (benchmarking in Phase 5)
- [ ] Add packed vs unpacked benchmark comparison (benchmarking in Phase 5)

---

## Phase 3: Triton Kernels (Week 7-8)

### 3.1 Triton Implementation

#### `smartkv/kernels/triton/quantized_attention.py`

```python
import triton
import triton.language as tl

@triton.jit
def quantized_attention_kernel(
    Q, K_int8, K_scale, V_int8, V_scale, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    B, H, M, N, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Block indices
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Load query block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + \
             offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

    # Tile over keys
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load quantized keys
        k_int8_ptrs = K_int8 + pid_b * stride_kb + pid_h * stride_kh + \
                      offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_int8 = tl.load(k_int8_ptrs, mask=offs_n[:, None] < N, other=0)

        # Load scales and dequantize
        k_scale_ptrs = K_scale + pid_b * H * N + pid_h * N + offs_n
        k_scale = tl.load(k_scale_ptrs, mask=offs_n < N, other=1.0)
        k = k_int8.to(tl.float32) * k_scale[:, None]

        # Compute attention scores
        scores = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]
        # ... softmax and output computation
```

**Advantages of Triton:**
- Easier to write than CUDA (Python-like syntax)
- JIT compilation optimizes for specific shapes
- Automatic memory coalescing
- Good performance out-of-the-box

**Tasks:**
- [ ] Create `smartkv/kernels/triton/` directory
- [ ] Implement quantized attention in Triton
- [ ] Add bit-packing utilities
- [ ] Benchmark vs CUDA kernel

### 3.2 Kernel Selection Logic

#### `smartkv/kernels/kernel_selector.py`

```python
import torch

def select_attention_kernel(
    context_length: int,
    gpu_arch: str,
    precision_distribution: dict
) -> str:
    """Auto-select best kernel based on context and hardware."""

    # Check GPU architecture
    if gpu_arch in ['sm_80', 'sm_86', 'sm_89']:  # A100, RTX 30xx/40xx
        has_tensor_cores = True
    else:
        has_tensor_cores = False

    # Check if uniform precision
    is_uniform = len(set(precision_distribution.values())) == 1

    # Selection logic
    if context_length < 2048:
        # Short context: CUDA kernel is fastest
        if is_uniform:
            return 'cuda_uniform'
        else:
            return 'cuda_mixed'
    else:
        # Long context: Triton handles tiling better
        return 'triton'

    # Fallback
    return 'torch_native'  # Pure PyTorch implementation

def quantized_attention(
    query, key_int8, key_scale, value_int8, value_scale,
    attention_mask=None
):
    """Unified interface for quantized attention."""

    device = query.device
    if device.type == 'cpu':
        from smartkv.core.fused_cpu import quantized_attention_streaming_cpu
        return quantized_attention_streaming_cpu(...)

    # GPU path
    context_len = key_int8.shape[2]
    gpu_arch = torch.cuda.get_device_capability()

    kernel = select_attention_kernel(context_len, gpu_arch, ...)

    if kernel == 'cuda_uniform':
        from smartkv_cuda import quantized_attention_8bit
        return quantized_attention_8bit(...)
    elif kernel == 'triton':
        from smartkv.kernels.triton import triton_quantized_attention
        return triton_quantized_attention(...)
    else:
        # Fallback: dequantize then standard attention
        k_fp = key_int8.float() * key_scale.unsqueeze(-1)
        v_fp = value_int8.float() * value_scale.unsqueeze(-1)
        return torch.nn.functional.scaled_dot_product_attention(query, k_fp, v_fp)
```

**Tasks:**
- [ ] Implement kernel selection heuristics
- [ ] Add unified attention interface
- [ ] Support automatic fallback
- [ ] Add profiling to tune selection logic

---

## Phase 4: Model Integration (Week 9-10)

### 4.1 Update Model Files

#### `smartkv/models/llama_smartkv.py`

**Current flow:**
```python
# Retrieve and dequantize (CPU)
keys, values = self.smartkv_cache.retrieve_all(layer_idx, token_ids)

# Standard attention
attn_output = torch.nn.functional.scaled_dot_product_attention(
    query, keys, values, attention_mask
)
```

**Target flow:**
```python
# Retrieve quantized (GPU)
quant_data = self.smartkv_cache.retrieve_all_quantized(layer_idx, token_ids)

# Fused quantized attention (GPU)
if self.use_fused_kernel and query.device.type == 'cuda':
    from smartkv.kernels import quantized_attention
    attn_output = quantized_attention(
        query=query,
        key_int8=quant_data['k_qx'],
        key_scale=quant_data['k_scale'],
        value_int8=quant_data['v_qx'],
        value_scale=quant_data['v_scale'],
        attention_mask=attention_mask
    )
else:
    # Fallback to dequantize-then-attend
    keys, values = self.smartkv_cache.retrieve_all(layer_idx, token_ids)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query, keys, values, attention_mask
    )
```

**Tasks:**
- [ ] Add `use_fused_kernel` flag to config
- [ ] Modify forward pass to use fused kernels
- [ ] Add device placement logic
- [ ] Test with different GPU types

#### `smartkv/models/attention.py`

```python
class SmartKVAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.use_fused_kernel = config.use_fused_kernel and torch.cuda.is_available()

    def forward(self, hidden_states, ...):
        # Project QKV
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Store in cache (quantized on GPU)
        self.cache.quantize_and_store_batch(
            layer_idx=self.layer_idx,
            token_ids=token_ids,
            k_batch=key,
            v_batch=value
        )

        # Fused attention
        if self.use_fused_kernel:
            output = self._fused_attention(query, ...)
        else:
            output = self._standard_attention(query, ...)

        return output
```

**Tasks:**
- [ ] Integrate fused kernels into attention layer
- [ ] Add automatic kernel selection
- [ ] Support distributed inference patterns

### 4.2 GPU Memory Management

#### `smartkv/utils/memory.py`

```python
import torch
import psutil

class GPUMemoryManager:
    """Manage GPU memory for SmartKV cache."""

    def __init__(self, device='cuda:0'):
        self.device = device

    def get_available_memory(self) -> int:
        """Get available GPU memory in bytes."""
        if self.device == 'cpu':
            return psutil.virtual_memory().available

        torch.cuda.empty_cache()
        free_mem, total_mem = torch.cuda.mem_get_info(self.device)
        return free_mem

    def estimate_cache_size(
        self,
        num_tokens: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        avg_bits: float = 4.0
    ) -> int:
        """Estimate memory usage for cache."""
        # Quantized KV storage
        payload_bits = num_tokens * num_layers * num_heads * head_dim * avg_bits * 2

        # Scale factors (FP32)
        scale_bits = num_tokens * num_layers * num_heads * 32 * 2

        return (payload_bits + scale_bits) // 8

    def auto_adjust_budget(
        self,
        max_context_length: int,
        num_layers: int,
        num_heads: int,
        head_dim: int
    ) -> float:
        """Automatically adjust memory budget based on available GPU memory."""
        available = self.get_available_memory()

        # Reserve 20% for activations and gradients
        usable = available * 0.8

        # Compute FP16 baseline
        fp16_cache_size = max_context_length * num_layers * num_heads * head_dim * 16 * 2 / 8

        # Compute achievable budget
        budget = usable / fp16_cache_size

        # Clamp to valid range
        from smartkv.core.allocation import compute_minimum_budget
        min_budget = compute_minimum_budget(num_heads, head_dim)

        return max(min_budget, min(1.0, budget))

    def enable_dynamic_batching(self, model, target_memory_ratio=0.9):
        """Adjust batch size dynamically to fit in memory."""
        # ... implementation
```

**Tasks:**
- [ ] Create GPU memory profiling utilities
- [ ] Implement dynamic budget adjustment
- [ ] Add OOM handling with cache eviction
- [ ] Support multi-GPU memory pooling

---

## Phase 5: Testing & Validation (Week 11-12)

### 5.1 CUDA Unit Tests

#### `tests/test_cuda_kernels.py`

```python
import torch
import pytest
from smartkv.kernels import quantized_attention
from smartkv.core._quant_cuda import quantize_per_head_cuda

class TestCUDAKernels:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_quantization_correctness(self):
        """Test CUDA quantization matches CPU."""
        torch.manual_seed(42)

        # Generate test data
        k = torch.randn(4, 8, 128).cuda()  # [N, H, D]
        v = torch.randn(4, 8, 128).cuda()
        bits = 8

        # CPU reference
        from smartkv.core._quant_cpu import quantize_per_head
        k_cpu = k.cpu()
        v_cpu = v.cpu()
        k_q_cpu, v_q_cpu, k_s_cpu, v_s_cpu = quantize_per_head(k_cpu, v_cpu, bits)

        # CUDA implementation
        k_q_cuda, v_q_cuda, k_s_cuda, v_s_cuda = quantize_per_head_cuda(k, v, bits)

        # Compare
        assert torch.allclose(k_q_cpu, k_q_cuda.cpu(), atol=1)
        assert torch.allclose(v_q_cpu, v_q_cuda.cpu(), atol=1)
        assert torch.allclose(k_s_cpu, k_s_cuda.cpu(), rtol=1e-4)
        assert torch.allclose(v_s_cpu, v_s_cuda.cpu(), rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fused_attention_correctness(self):
        """Test fused attention matches reference."""
        torch.manual_seed(42)

        # Test data
        B, H, q_len, k_len, d = 1, 8, 1, 500, 128
        query = torch.randn(B, H, q_len, d).cuda()
        k_int8 = torch.randint(-128, 127, (B, H, k_len, d), dtype=torch.int8).cuda()
        v_int8 = torch.randint(-128, 127, (B, H, k_len, d), dtype=torch.int8).cuda()
        k_scale = torch.randn(B, H, k_len).cuda()
        v_scale = torch.randn(B, H, k_len).cuda()

        # Reference: dequantize then attend
        k_fp = k_int8.float() * k_scale.unsqueeze(-1)
        v_fp = v_int8.float() * v_scale.unsqueeze(-1)
        ref_output = torch.nn.functional.scaled_dot_product_attention(
            query, k_fp, v_fp
        )

        # Fused kernel
        fused_output = quantized_attention(
            query, k_int8, k_scale, v_int8, v_scale
        )

        # Compare
        assert torch.allclose(ref_output, fused_output, atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("context_len", [100, 500, 1000, 2000])
    def test_performance_vs_fp16(self, context_len):
        """Test speedup at different context lengths."""
        torch.manual_seed(42)

        B, H, q_len, d = 1, 8, 1, 128
        query = torch.randn(B, H, q_len, d).cuda()
        k_int8 = torch.randint(-128, 127, (B, H, context_len, d), dtype=torch.int8).cuda()
        v_int8 = torch.randint(-128, 127, (B, H, context_len, d), dtype=torch.int8).cuda()
        k_scale = torch.randn(B, H, context_len).cuda()
        v_scale = torch.randn(B, H, context_len).cuda()

        # Warm up
        for _ in range(10):
            _ = quantized_attention(query, k_int8, k_scale, v_int8, v_scale)

        # Benchmark fused kernel
        torch.cuda.synchronize()
        import time
        start = time.time()
        for _ in range(100):
            _ = quantized_attention(query, k_int8, k_scale, v_int8, v_scale)
        torch.cuda.synchronize()
        fused_time = (time.time() - start) / 100

        # Benchmark reference
        k_fp = k_int8.float() * k_scale.unsqueeze(-1)
        v_fp = v_int8.float() * v_scale.unsqueeze(-1)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = torch.nn.functional.scaled_dot_product_attention(query, k_fp, v_fp)
        torch.cuda.synchronize()
        ref_time = (time.time() - start) / 100

        speedup = ref_time / fused_time
        print(f"Context {context_len}: Speedup = {speedup:.2f}x "
              f"({ref_time*1000:.2f}ms -> {fused_time*1000:.2f}ms)")

        # Assert speedup
        if context_len >= 500:
            assert speedup >= 1.5, f"Insufficient speedup at {context_len} tokens"
```

**Tasks:**
- [ ] Write numerical correctness tests
- [ ] Test multi-GPU support
- [ ] Validate edge cases (batch_size=1, extreme values)
- [ ] Performance regression tests

### 5.2 Integration Tests

#### `tests/test_gpu_generation.py`

```python
import torch
import pytest
from transformers import AutoTokenizer
from smartkv.models.llama_smartkv import LlamaSmartKV, SmartKVConfig

class TestGPUGeneration:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_end_to_end_generation(self):
        """Test full generation pipeline on GPU."""
        config = SmartKVConfig(
            enabled=True,
            memory_budget=0.5,
            device='cuda',
            use_fused_kernel=True
        )

        model = LlamaSmartKV.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            smartkv_config=config
        )
        model.cuda()

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

        generated_text = tokenizer.decode(outputs[0])

        # Verify output quality
        assert len(generated_text) > len(prompt)
        assert "Paris" in generated_text

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_long_context_gpu(self):
        """Test with long context (8K tokens)."""
        # Load Romeo & Juliet or similar long text
        with open('tests/fixtures/romeo_juliet.txt') as f:
            long_text = f.read()

        config = SmartKVConfig(
            enabled=True,
            memory_budget=0.4,
            device='cuda'
        )

        model = LlamaSmartKV.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            smartkv_config=config
        )
        model.cuda()

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        # Encode long context
        inputs = tokenizer(long_text[:20000], return_tensors="pt").to('cuda')

        # Add query
        query = "\n\nQuestion: Who are the main characters?\nAnswer:"
        query_tokens = tokenizer(query, return_tensors="pt").to('cuda')

        # Generate
        outputs = model.generate(
            input_ids=torch.cat([inputs.input_ids, query_tokens.input_ids], dim=1),
            max_new_tokens=50
        )

        answer = tokenizer.decode(outputs[0])
        assert "Romeo" in answer or "Juliet" in answer

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_inference(self):
        """Test batched generation on GPU."""
        config = SmartKVConfig(enabled=True, device='cuda')
        model = LlamaSmartKV.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            smartkv_config=config
        )
        model.cuda()

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token

        prompts = [
            "The capital of France is",
            "The largest planet in our solar system is",
            "The speed of light is"
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')

        outputs = model.generate(**inputs, max_new_tokens=20)

        for i, output in enumerate(outputs):
            text = tokenizer.decode(output)
            print(f"Prompt {i}: {text}")
            assert len(text) > len(prompts[i])
```

**Tasks:**
- [ ] End-to-end generation tests
- [ ] Long-context validation (8K-128K tokens)
- [ ] Batch inference tests
- [ ] Multi-GPU distributed tests

### 5.3 Performance Benchmarks

#### `scripts/benchmark_gpu.py`

```python
import torch
import time
from smartkv.models.llama_smartkv import LlamaSmartKV, SmartKVConfig
from transformers import AutoTokenizer

def benchmark_latency():
    """Benchmark latency vs context length."""
    context_lengths = [100, 200, 500, 1000, 2000, 4000, 8000]

    for ctx_len in context_lengths:
        # SmartKV GPU
        config = SmartKVConfig(enabled=True, memory_budget=0.4, device='cuda')
        model = LlamaSmartKV.from_pretrained("meta-llama/Llama-2-7b-hf", smartkv_config=config)
        model.cuda()

        # Generate context
        dummy_input = torch.randint(0, 32000, (1, ctx_len)).cuda()

        # Warm up
        _ = model(dummy_input)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        smartkv_time = (time.time() - start) / 10

        # FP16 baseline
        model.disable_smartkv()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        fp16_time = (time.time() - start) / 10

        speedup = fp16_time / smartkv_time
        print(f"{ctx_len} tokens: {smartkv_time*1000:.2f}ms (SmartKV) vs "
              f"{fp16_time*1000:.2f}ms (FP16) = {speedup:.2f}x speedup")

def benchmark_memory():
    """Benchmark memory usage."""
    config = SmartKVConfig(enabled=True, memory_budget=0.4, device='cuda')
    model = LlamaSmartKV.from_pretrained("meta-llama/Llama-2-7b-hf", smartkv_config=config)
    model.cuda()

    torch.cuda.reset_peak_memory_stats()

    # Generate with long context
    dummy_input = torch.randint(0, 32000, (1, 4000)).cuda()
    _ = model(dummy_input)

    smartkv_mem = torch.cuda.max_memory_allocated() / 1e9

    # FP16 baseline
    model.disable_smartkv()
    torch.cuda.reset_peak_memory_stats()
    _ = model(dummy_input)
    fp16_mem = torch.cuda.max_memory_allocated() / 1e9

    savings = (1 - smartkv_mem / fp16_mem) * 100
    print(f"Memory: {smartkv_mem:.2f}GB (SmartKV) vs {fp16_mem:.2f}GB (FP16) = {savings:.1f}% savings")

if __name__ == "__main__":
    benchmark_latency()
    benchmark_memory()
```

**Tasks:**
- [ ] Latency benchmarks (context length sweep)
- [ ] Throughput measurements (tokens/sec)
- [ ] Memory usage profiling
- [ ] Compare CUDA vs Triton kernels

---

## Phase 6: Advanced Optimizations (Week 13-14)

### 6.1 Tensor Core Support

**For A100/H100 GPUs:**

```cuda
// Use WMMA (Warp Matrix Multiply-Accumulate) for INT8 matmul
#include <mma.h>
using namespace nvcuda;

__global__ void quantized_attention_tensor_core(
    const float* query,
    const int8_t* key_quantized,
    const float* key_scale,
    float* output,
    // ... params
) {
    // Declare fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> c_frag;

    // Load query to fragment (convert FP32 -> INT8)
    wmma::load_matrix_sync(a_frag, query, 16);

    // Load key fragment
    wmma::load_matrix_sync(b_frag, key_quantized, 16);

    // Compute with Tensor Cores
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Scale and store
    // ...
}
```

**Expected speedup:**
- A100: 10-20× faster for INT8 matmul
- Requires careful alignment and data layout

**Tasks:**
- [ ] Implement Tensor Core path for A100/H100
- [ ] Handle mixed FP32 query + INT8 keys
- [ ] Optimize data layout for WMMA
- [ ] Benchmark vs standard CUDA kernel

### 6.2 Multi-GPU Support

#### `smartkv/distributed/tensor_parallel.py`

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

class TensorParallelSmartKV:
    """Tensor parallel SmartKV for large models."""

    def __init__(self, model, world_size):
        self.model = model
        self.world_size = world_size
        self.rank = dist.get_rank()

        # Split attention heads across GPUs
        self.heads_per_gpu = model.num_heads // world_size
        self.head_start = self.rank * self.heads_per_gpu
        self.head_end = self.head_start + self.heads_per_gpu

    def forward(self, hidden_states):
        # Each GPU computes subset of heads
        query = self.model.q_proj(hidden_states)
        query_local = query[:, :, self.head_start:self.head_end, :]

        # Compute local attention
        output_local = self.model.attention(query_local, ...)

        # All-reduce across GPUs
        dist.all_reduce(output_local, op=dist.ReduceOp.SUM)

        return output_local
```

#### `smartkv/distributed/pipeline_parallel.py`

```python
class PipelineParallelSmartKV:
    """Pipeline parallel for ultra-long context."""

    def __init__(self, model, num_stages):
        # Split layers across GPUs
        self.stages = self._split_model(model, num_stages)

    def forward(self, input_ids):
        # Forward pass through pipeline
        hidden = input_ids
        for stage in self.stages:
            hidden = stage(hidden)
        return hidden
```

**Tasks:**
- [ ] Implement tensor parallelism for attention heads
- [ ] Pipeline parallelism for layer distribution
- [ ] NCCL communication for cache sharing
- [ ] Benchmark multi-GPU scaling

### 6.3 Quantization-Aware Training

#### `smartkv/training/qat.py`

```python
class SmartKVQAT:
    """Quantization-Aware Training for SmartKV."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Straight-through estimator for quantization
        self.quantize_ste = QuantizeSTE.apply

    def forward(self, input_ids, labels):
        # Forward with simulated quantization
        hidden = self.model.embed(input_ids)

        for layer in self.model.layers:
            # Quantize KV with STE
            k = layer.k_proj(hidden)
            v = layer.v_proj(hidden)

            k_quant = self.quantize_ste(k, bits=self.get_bits(k))
            v_quant = self.quantize_ste(v, bits=self.get_bits(v))

            # Attention with quantized KV
            output = layer.attention(query, k_quant, v_quant)
            hidden = output

        # Compute loss
        logits = self.model.lm_head(hidden)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss

class QuantizeSTE(torch.autograd.Function):
    """Straight-Through Estimator for quantization."""

    @staticmethod
    def forward(ctx, input, bits):
        # Forward: actual quantization
        scale = input.abs().max() / (2**(bits-1) - 1)
        output = torch.round(input / scale) * scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: straight-through
        return grad_output, None
```

**Tasks:**
- [ ] Implement STE for differentiable quantization
- [ ] Fine-tune models with SmartKV in the loop
- [ ] Measure perplexity improvement
- [ ] Support different quantization schedules

---

## File Structure (Post-Port)

```
SmartKV/
├── smartkv/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── cache.py                  # ✅ GPU tensor support added
│   │   ├── allocation.py             # ✅ Already device-agnostic
│   │   ├── importance.py             # ✅ GPU attention tracking added
│   │   ├── quantizers.py             # Existing quantizer interfaces
│   │   ├── _quant_cpu.py            # ✅ Existing CPU quantization
│   │   └── _quant_cuda.py           # 🆕 CUDA quantization kernels
│   │
│   ├── csrc/                         # 🆕 C++/CUDA extensions
│   │   ├── quantized_attention.cu   # Main CUDA kernel
│   │   ├── quantized_attention.h    # Header file
│   │   ├── quantized_attention_wrapper.cpp  # PyTorch binding
│   │   └── setup.py                 # Build configuration
│   │
│   ├── kernels/                      # 🆕 High-level kernel interface
│   │   ├── __init__.py
│   │   ├── kernel_selector.py       # Auto-select CUDA vs Triton
│   │   └── triton/                  # Triton implementations
│   │       ├── __init__.py
│   │       ├── quantized_attention.py
│   │       └── bit_packing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llama_smartkv.py         # ✅ GPU forward pass added
│   │   └── attention.py             # ✅ Fused kernel integration
│   │
│   ├── distributed/                  # 🆕 Multi-GPU support
│   │   ├── __init__.py
│   │   ├── tensor_parallel.py       # Tensor parallelism
│   │   └── pipeline_parallel.py     # Pipeline parallelism
│   │
│   ├── training/                     # 🆕 Training utilities
│   │   ├── __init__.py
│   │   └── qat.py                   # Quantization-aware training
│   │
│   └── utils/
│       ├── __init__.py
│       ├── memory.py                 # 🆕 GPU memory management
│       └── profiling.py              # 🆕 Performance profiling
│
├── tests/
│   ├── __init__.py
│   ├── test_quantizers.py            # ✅ Existing
│   ├── test_cache.py                 # ✅ Existing
│   ├── test_cuda_kernels.py         # 🆕 CUDA kernel tests
│   ├── test_triton_kernels.py       # 🆕 Triton kernel tests
│   ├── test_gpu_generation.py       # 🆕 End-to-end GPU tests
│   └── test_distributed.py          # 🆕 Multi-GPU tests
│
├── scripts/
│   ├── benchmark_gpu.py              # 🆕 GPU benchmarking
│   └── profile_memory.py             # 🆕 Memory profiling
│
├── setup.py                          # 🆕 Root package setup
├── requirements.txt                  # Existing CPU deps
├── requirements-gpu.txt              # 🆕 GPU-specific deps
│
├── ALGORITHM_SPECIFICATION.md        # ✅ Existing
├── KERNEL_IMPLEMENTATION_PLAN.md    # ✅ Existing (CPU focus)
└── GPU_PORTING_ROADMAP.md           # 🆕 This document
```

**Legend:**
- ✅ Existing files (will be modified)
- 🆕 New files to create

---

## Dependencies to Add

### `requirements-gpu.txt`

```
# CUDA dependencies
cuda-python>=12.0
ninja>=1.11.0

# Triton compiler
triton>=2.1.0

# FlashAttention (reference implementation)
flash-attn>=2.3.0

# Distributed training
torch-distributed>=2.0.0
deepspeed>=0.10.0  # Optional: for ZeRO optimization

# Profiling
py3nvml>=0.2.7
gpustat>=1.1.0
```

### Update `requirements.txt`

```
# Core dependencies
torch>=2.0.0
transformers>=4.35.0
numpy>=1.24.0
scipy>=1.10.0

# ... existing deps ...

# GPU support (optional)
# Install with: pip install smartkv[gpu]
# See requirements-gpu.txt for details
```

### `setup.py` with optional GPU extras

```python
from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

with open('requirements-gpu.txt') as f:
    gpu_requires = f.read().splitlines()

setup(
    name='smartkv',
    version='0.2.0',
    author='Robby Moseley',
    description='SmartKV: Attention-Guided Adaptive Precision KV-Cache Compression',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        'gpu': gpu_requires,
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
        ]
    },
    python_requires='>=3.9',
)
```

**Installation:**
```bash
# CPU only
pip install smartkv

# With GPU support
pip install smartkv[gpu]

# Development (all deps)
pip install -e ".[gpu,dev]"
```

---

## Success Metrics

### Performance Targets

| Context Length | Current (CPU) | Target (GPU) | Speedup |
|---------------|---------------|--------------|---------|
| 100 tokens    | 14ms          | 14ms         | 1.0× (maintain) |
| 500 tokens    | 41ms          | 12ms         | 3.4× |
| 1K tokens     | ~80ms         | 25ms         | 3.2× |
| 8K tokens     | ~600ms        | 60ms         | 10× |
| 32K tokens    | ~2400ms       | 150ms        | 16× |

### Memory Targets

**Current CPU (no bit-packing):**
| Configuration | FP16 Baseline | SmartKV (40% budget) | Actual Savings |
|--------------|---------------|----------------------|----------------|
| 8K context   | 2.1 GB        | 1.12 GB              | 47% (limited by INT8 storage) |
| 32K context  | 8.4 GB        | 4.47 GB              | 47% |
| 128K context | 33.6 GB       | 17.9 GB              | 47% |

**GPU Target (with bit-packing):**
| Configuration | FP16 Baseline | SmartKV (40% budget, packed) | True Savings |
|--------------|---------------|------------------------------|--------------|
| 8K context   | 2.1 GB        | 0.59 GB                      | 72% ✨ |
| 32K context  | 8.4 GB        | 2.36 GB                      | 72% ✨ |
| 128K context | 33.6 GB       | 9.44 GB                      | 72% ✨ |

**Note:** Bit-packing is essential for GPU to achieve true sub-50% memory usage. The CPU implementation is limited to ~50% minimum due to INT8 storage overhead.

### Quality Targets

- **Perplexity degradation**: <2% vs FP16
- **Numerical accuracy**: atol=1e-4 vs CPU reference
- **Generation quality**: Pass LongBench evaluation
- **Consistency**: Match CPU results (fp16 scale precision)

---

## Risks & Mitigations

### Risk 1: CUDA Build Complexity
**Issue:** CUDA extensions hard to build on different systems

**Mitigation:**
- Provide pre-built wheels for common platforms (PyPI)
- Triton fallback (pure Python, no build required)
- Comprehensive build documentation
- Docker images with pre-installed CUDA toolkit

### Risk 2: Numerical Instability
**Issue:** GPU kernels may introduce numerical errors

**Mitigation:**
- Extensive testing with reference implementations
- Edge case tests (zeros, infinities, NaNs)
- Validate on downstream tasks (perplexity, accuracy)
- Use FP32 accumulation for stability

### Risk 3: Memory Overhead
**Issue:** GPU memory fragmentation, OOM errors

**Mitigation:**
- Dynamic budget adjustment based on available memory
- Gradient checkpointing for training
- Cache eviction policies for long context
- Multi-GPU fallback for large contexts

### Risk 4: Multi-GPU Synchronization
**Issue:** Communication overhead in distributed setting

**Mitigation:**
- Use PyTorch DDP best practices
- NCCL for efficient all-reduce
- Pipeline parallelism to hide latency
- Benchmark communication vs computation ratio

### Risk 5: Performance Regression on Old GPUs
**Issue:** Optimizations for A100 may hurt older GPUs

**Mitigation:**
- Kernel selection based on GPU architecture
- Maintain separate code paths (sm_70, sm_80, sm_89)
- Fallback to PyTorch native operations
- Clear documentation of supported GPUs

---

## Implementation Checklist

### Phase 1: Core GPU Infrastructure ☐
- [ ] Update `cache.py` for GPU tensors
- [ ] Update `importance.py` for GPU tracking
- [ ] Create `_quant_cuda.py` with CUDA quantization

### Phase 2: CUDA Kernel Development ☐
- [ ] Set up `csrc/` directory
- [ ] Create build configuration
- [ ] Implement minimal fused attention kernel
- [ ] Optimize with FlashAttention-2 techniques
- [ ] Add bit-packing support

### Phase 3: Triton Kernels ☐
- [ ] Create `kernels/triton/` directory
- [ ] Implement Triton quantized attention
- [ ] Add kernel selection logic
- [ ] Benchmark CUDA vs Triton

### Phase 4: Model Integration ☐
- [ ] Update `llama_smartkv.py` with GPU forward
- [ ] Integrate fused kernels
- [ ] Create GPU memory utilities
- [ ] Test with real models

### Phase 5: Testing & Validation ☐
- [ ] CUDA kernel unit tests
- [ ] End-to-end GPU generation tests
- [ ] Performance benchmarks
- [ ] Memory profiling

### Phase 6: Advanced Optimizations ☐
- [ ] Tensor Core support (A100/H100)
- [ ] Multi-GPU distributed inference
- [ ] Quantization-aware training
- [ ] Production deployment guide

### Documentation ☐
- [ ] Update README with GPU installation
- [ ] Document kernel selection logic
- [ ] Performance tuning guide
- [ ] Multi-GPU usage examples

---

## Timeline Summary

**Total Duration:** 14 weeks (3.5 months)

| Phase | Weeks | Milestone |
|-------|-------|-----------|
| Phase 1 | 1-2 | GPU infrastructure ready |
| Phase 2 | 3-6 | CUDA kernels operational |
| Phase 3 | 7-8 | Triton kernels available |
| Phase 4 | 9-10 | Model integration complete |
| Phase 5 | 11-12 | All tests passing |
| Phase 6 | 13-14 | Advanced optimizations done |

**Key Milestones:**
- **Week 2:** GPU tensors working in cache
- **Week 4:** Minimal CUDA kernel operational
- **Week 6:** Optimized CUDA kernel achieving 3× speedup
- **Week 8:** Triton kernel parity with CUDA
- **Week 10:** Full model integration on GPU
- **Week 12:** All tests passing, ready for production
- **Week 14:** Multi-GPU and Tensor Core optimizations complete

---

## Next Steps

### Immediate (Week 1)
1. Create `smartkv/core/_quant_cuda.py` with basic CUDA quantization
2. Update `cache.py` to support `device='cuda'`
3. Set up `csrc/` directory structure
4. Write first CUDA kernel tests

### Short-term (Weeks 2-4)
1. Implement minimal fused attention kernel
2. Integrate with existing model code
3. Validate numerical correctness
4. Measure initial performance gains

### Medium-term (Weeks 5-10)
1. Optimize CUDA kernels (FlashAttention-2 style)
2. Add Triton implementations
3. Full model integration and testing
4. Performance benchmarking suite

### Long-term (Weeks 11-14)
1. Multi-GPU support
2. Tensor Core optimizations
3. Production deployment guide
4. Research paper submission

---

**Document Version:** 1.0
**Last Updated:** 2025-10-04
**Author:** Robby Moseley
**Status:** Planning Phase ✅
