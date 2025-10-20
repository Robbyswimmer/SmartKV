# GPU 2/3/4-Bit Packing Overview

This note captures how values will be packed into per-bit buffers on the GPU. It
applies equally to keys (`k_qx`) and values (`v_qx`).

## Scope
- Store per-bit payloads compactly to reduce memory footprint and improve load
  coalescing for the fused kernel.
- Pack/unpack implemented as CUDA kernels; python layer just orchestrates.
- Maintain per-token alignment (no mixing tokens inside a byte).

## Constraints
- Current head_dim is a multiple of 8 (for DP4A) and often 128.
- Each token has `num_heads` vectors of length `head_dim`.
- We write/rewind per token (slot-based), so packing must support appending and in-place updates without expensive rebuilds.

## Proposed Layouts

### 1. 4-bit
- Pack 2 values in one byte (2×4 bits).
- For each (token, head), the data length is `head_dim / 2` bytes.
- Flatten order: [token][head][packed_bytes].

### 2. 2-bit
- Pack 4 values in one byte.
- Layout similarly becomes `[token][head][head_dim / 4]` bytes.

### 3. 3-bit
- Pack 8 values into 24 bits (3 bytes).
- Since 3 doesn’t divide 8 cleanly, use groups of 8 (fill last group with zero padding if `head_dim` not a multiple of 8).
- Payload size per (token, head) = `ceil(head_dim / 8) * 3` bytes.

### Metadata
- For each token we already track global slot, head count, etc. No additional metadata needed beyond the bit-width.
- We keep per bit-tier buffers, so the fused kernel knows the width once it selects a bucket.

## CUDA Routines (Phase 2.3)
1. `pack_bits_cuda(float32 -> int8/packed)`
   - Input: `[N, H, D] float32`, scales, bit-width.
   - Counterpart to current quantizer: after quantizing to int8, call a pack kernel to compress into the bucket buffer.
2. `unpack_bits_cuda(packed -> float32)`
   - For a given bucket, expand the packed bytes back to int8 (or float32) when needed (e.g., for CPU fallback or debugging).
3. Batch operations use thread blocks per (token, head) chunk. The kernel writes contiguous bytes per block to keep loads/stores aligned.

## Host Flow
- Quantization:
  1. Call `quantize_per_head_cuda` to get int8 and scales.
  2. Call `pack_bits_cuda` (if bits < 8) before storing in `bucket_buffers[bits]['k_qx']` and `['v_qx']`.
  3. If `use_packing=False`, keep raw int8 values in the legacy tensors only.
- Retrieval (for CPU or non-packed paths):
  1. If `bits < 8`, call `unpack_bits_cuda` when a full FP32 tensor is needed.
  2. Fused kernel will eventually operate on the packed buffers directly.

## Status
- CUDA pack/unpack kernels for 2-, 3-, and 4-bit values are implemented in
  `smartkv/csrc/bit_packing.cu` and exposed via `smartkv_cuda.pack_values`.
- `SmartKVCache` writes packed payloads into bucket buffers when
  `use_bucketed_layout=True` and packing is enabled.
- Tests compare CUDA quantization vs CPU reference and verify bucket tensor shapes/dtypes.

## Next Action
- Migrate the fused attention kernel to consume packed bucket buffers directly.
- Add CUDA benchmarks to measure throughput improvements from packing.
