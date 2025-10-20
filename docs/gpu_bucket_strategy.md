# SmartKV GPU Bucketed Cache and Fused Kernel Plan

This document captures the staged plan for upgrading SmartKV's CUDA path so that
mixed-precision attention consistently outperforms uniform INT8.

## Phase 1 – Bucketed Cache Layout *(in progress)*

- **Goal:** Store quantized tokens in per-bit buckets (`2`, `3`, `4`, `8` bits) so GPU
  kernels can read homogeneous data without divergence.
- **Data structures:** For each layer and bit-width, maintain contiguous tensors for
  payload, scales, and metadata (token IDs, indices). Token moves during reallocation
  simply swap between buckets. Initial scaffolding lives in `SmartKVCache` via
  `use_bucketed_layout`, with per-bit index sets and helper APIs (`get_bucket_views`,
  etc.) ready for future kernels.
- **Deliverables:**
  - `BucketedCacheLayer` container with tensors per bucket on the target device.
  - Updated `SmartKVCache.quantize_and_store_batch` and `_requantize_single` to write to
    buckets when `use_bucketed_layout=True`.
  - Unit tests ensuring bucket counts match precision map, memory stats align with
    theoretical `avg_bits`.

## Phase 2 – GPU Quantization / Requantization *(in progress)*

- Replace CPU fallback with fused CUDA quantizers that pack tensors directly into the
  bucket layout. A per-head CUDA kernel now produces int8 payloads + scales; bucket
  views surface device tensors (`retrieve_all_quantized` avoids CPU copies). Per-bit
  buffers are populated alongside legacy tensors, and 2/3/4-bit payloads are stored in
  packed form. Next steps: teach the fused kernel to consume these bucket views.
- Implement 2/3/4-bit packing kernels (DP4A-friendly) and expose them via
  `smartkv.core._quant_cuda`.
- Add tests comparing GPU vs CPU quantization outputs.

## Phase 3 – Tiled Mixed-Precision Fused Kernel

- **Key requirements:**
  1. Tile over `k_len` so shared memory stays ≤ 48 KB (supports consumer GPUs).
  2. Process one bucket at a time to keep loads coalesced and avoid predicate-heavy
     branches.
  3. Use streaming softmax (keep running `max` and `sum` per query) to avoid storing all
     logits.
  4. Use integer dot products (`__dp4a`, IMMA) after unpacking or by packing the inputs
     into 4x int8 values.
  5. Accumulate partial `A@V` per bucket; host code loops over buckets and sums results.
- **Outputs:** C++/CUDA kernel(s) in `smartkv/csrc`, Python bindings in
  `smartkv/kernels/__init__.py`, comprehensive launch configuration.

## Phase 4 – Integration

- Upgrade `SmartKVCache.retrieve_all_quantized` to return the bucketed buffers (with
  offsets) expected by the fused kernel.
- Ensure allocator and reallocation routines update bucket metadata during tier
  changes.
- Fuse EMA/head-importance reductions into the attention pipeline to remove extra
  passes.

## Phase 5 – Validation & Benchmarking

- Extend `scripts/benchmark_gpu_attention.py` to benchmark SmartKV vs uniform INT8
  using the new kernel across context lengths (1k–16k) and budgets (0.2–0.5).
- Add Nsight profiling recipes (DRAM throughput, occupancy, shared-mem usage).
- Regression tests covering kernel correctness vs FP16 reference and stress cases
  (large contexts, mixed precisions, drift scenarios).

## Phase 6 – Documentation

- Update `ALGORITHM_SPECIFICATION.md` with the final bucketed layout and streaming
  softmax math.
- Refresh README and any public docs to include performance tables and usage notes.

## Stretch Goals

- Autotune tile sizes/block sizes per GPU architecture.
- Explore FP16/FP8 scales and per-K/V asymmetry for further savings.
- Evaluate end-to-end decode speedups on full models (Llama, etc.) and include latency
  vs. memory trade-off charts.
