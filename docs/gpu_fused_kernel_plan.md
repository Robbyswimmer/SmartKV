# Phase 3: Bucket-Aware Fused Attention Kernel Design

## Goals
- Use the new bucketed storage to run mixed-precision attention efficiently on CUDA.
- Avoid per-token divergence: each kernel pass handles one bit tier at a time.
- Support long contexts without exhausting shared memory by tiling over `k_len`.
- Stream softmax (like FlashAttention) to keep accuracy and avoid temporary buffers.

## Context Recap
- Cache layout now provides per-bit buffers:
  - `token_ids` (bucket), `global_slots`, `k_qx` (packed or int8), `k_scale`, `v_qx`, `v_scale`.
  - For 2/3/4-bit buckets, `k_qx`/`v_qx` are packed `uint8` arrays; 8-bit remains int8.
- Average bit-width is controlled by the allocator; buckets stay relatively stable once hysteresis hits.

## Kernel Structure
1. **Launch Parameters**
   - We still process single-query decode steps (`q_len = 1`) initially. The grid can be `(batch, heads)` or `(batch, heads, buckets)` depending on how many queries we fuse.
   - For each bucket, process logical tiles of keys/values: e.g., `tile_size = 128` tokens. That keeps shared memory under 48KB on SMs that only expose 48K (consumer cards).

2. **Host Loop**
   - For each bit tier in `available_bits`:
     1. Slice bucket buffers for `(num_tokens, num_heads, packed_dim)` + scales.
     2. Launch `quantized_attention_bucket<<<grid, block, shared_mem>>>(query, bucket_payload, scales, mask, ... )` accumulating partial results.
   - Accumulate the `AV` output (either by summing partial results or doing the softmax work per bucket with streaming max/sum).

3. **Kernel Steps per Bucket**
   1. **Tile loading**: load a tile of packed payload + scales into shared memory.
   2. **Unpack on the fly**: for 2/3/4-bit buckets, unpack into registers (e.g., using bit shifts or DP4A helpers). For 8-bit, load directly.
   3. **Dot product**: compute `Q · K^T` for the tile. Use int operations when possible (e.g., DP4A) and fuse scale multiply at the end.
   4. **Streaming softmax**: maintain `max_i` and `sum_i` per query across tiles. After each tile, update the running max/sum and scale previous tile contributions accordingly.
   5. **Probability * V**: multiply normalized probabilities by the tile’s `V` (unpacked) and accumulate `AV`.

4. **Streaming softmax math**
   - Keep per-query `m` (max) and `s` (sum). For each tile, compute tile logits `l`. Then:
     ```
     m_new = max(m, max(l))
     s = s * exp(m - m_new) + sum(exp(l - m_new))
     m = m_new
     ```
   - Multiply tile’s `P` by `exp(m_old - m_new)` before accumulation to keep everything in sync.

5. **Packing/Unpacking Helpers**
   - For 2-bit: 4 values per byte.
   - For 3-bit: 8 values per 3 bytes—special care to avoid branchy code (use lookup or bit-trick).
   - For 4-bit: 2 values per byte.
   - Provide inline device functions to load/unpack up to 8 values at once; lean on warp-wide DP4A when possible.

6. **Memory Footprint**
   - Shared memory per tile: `query + unpacked tile + scratch`. With `tile_size=128` and `head_dim=128`, unpacked tile per head is 16 KB for int8 (`128*128`). For 4-bit (~8 KB). Boost `tile_size` if hardware supports more SMEM.

7. **Thread Layout**
   - Example: `blockDim.x = 128` threads, each handles multiple dimensions:
     - Phase 1 (logits): each thread updates scores for one or more `k_idx` entries.
     - Phase 2 (softmax): reduce within warp, then across warps (use existing block reduce helper).
     - Phase 3 (value accumulation): reuse threads to multiply probabilities by `V` and sum.

8. **Accumulation Strategy**
   - Have the fused kernel return the partial `AV` for the bucket.
   - Host sums these per-bucket outputs after applying the global normalization (because each bucket sees the same final max and sum from the streaming softmax). Alternatively, perform the streaming softmax across buckets in the kernel (bucket loop inside kernel).

9. **Integration Hooks**
   - Update `quantized_attention` to detect `use_bucketed_layout` and dispatch to the new kernel.
   - Provide fallback paths (if bucketed layout disabled or bits unknown, revert to the current minimal kernel).

10. **Testing**
    - Compare new kernel output against FP16 attention for small tensors with known values and varying bit allocations.
    - Validate bucket-handling correctness by forcing different buckets to have data.

11. **Profiling & Tuning**
    - Use Nsight Compute to check shared-mem occupancy, DRAM throughput, divergent branches.
    - Autotune `tile_size` per architecture.

## Next Steps
1. Implement CUDA kernels per bucket (with templated bit width variants).
2. Add wrappers to launch these kernels from Python. *(Host-side assembly is now in
   place via `quantized_attention_bucketed`, currently using the legacy kernel after
   unpacking.)*
3. Update `retrieve_all_quantized`/callers to pass bucket views into the kernel.
4. Benchmark against uniform INT8 at multiple context lengths.
