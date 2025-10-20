#!/usr/bin/env python3
"""Debug test to identify bucket kernel issues."""

import torch
from smartkv.core.cache import SmartKVCache
from smartkv.kernels import quantized_attention, quantized_attention_bucketed, CUDA_AVAILABLE

if not CUDA_AVAILABLE or not torch.cuda.is_available():
    print("CUDA not available, skipping")
    exit(0)

torch.manual_seed(2)
device = torch.device('cuda')

cache = SmartKVCache(
    num_layers=1,
    num_heads=2,
    head_dim=16,
    memory_budget=0.3,
    device='cuda',
    available_bits=[2, 3, 4, 8],
    use_bucketed_layout=True,
)

ids = list(range(6))
cache.precision_map.update({i: [2, 3, 4, 8, 3, 2][i] for i in range(6)})

k = torch.randn(len(ids), 2, 16, device=device)
v = torch.randn(len(ids), 2, 16, device=device)
cache.quantize_and_store_batch(0, ids, k, v)

buckets = cache.get_bucket_views(0)
query = torch.randn(1, 2, 1, 16, device=device)

print("=== BUCKET VIEWS ===")
for bits, view in buckets.items():
    print(f"\nBucket {bits}-bit:")
    print(f"  token_ids: {view['token_ids'].tolist()}")
    print(f"  k_scale shape: {view['k_scale'].shape}")
    print(f"  v_scale shape: {view['v_scale'].shape}")
    print(f"  k_qx shape: {view['k_qx'].shape}")
    print(f"  global_slots: {view['global_slots'].tolist()}")

print("\n=== CALLING BUCKET KERNEL ===")
try:
    bucket_out = quantized_attention_bucketed(query, buckets, use_cuda=True)
    print(f"bucket_out shape: {bucket_out.shape}")
    print(f"bucket_out has NaN: {torch.isnan(bucket_out).any()}")
    print(f"bucket_out has Inf: {torch.isinf(bucket_out).any()}")
    print(f"bucket_out sample: {bucket_out[0, 0, 0, :5]}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== CALLING LEGACY PATH ===")
legacy = cache.retrieve_all_quantized(0)
legacy_out = quantized_attention(
    query,
    legacy['k_qx'].unsqueeze(0),
    legacy['k_scale'].unsqueeze(0),
    legacy['v_qx'].unsqueeze(0),
    legacy['v_scale'].unsqueeze(0),
    use_cuda=True,
)
print(f"legacy_out shape: {legacy_out.shape}")
print(f"legacy_out has NaN: {torch.isnan(legacy_out).any()}")
print(f"legacy_out has Inf: {torch.isinf(legacy_out).any()}")
print(f"legacy_out sample: {legacy_out[0, 0, 0, :5]}")

print("\n=== COMPARISON ===")
if not torch.isnan(bucket_out).any() and not torch.isnan(legacy_out).any():
    diff = (bucket_out - legacy_out).abs()
    print(f"Max diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")
    if diff.max() < 1e-4:
        print("✓ PASS: Outputs match within tolerance")
    else:
        print("✗ FAIL: Outputs differ significantly")
else:
    print("✗ FAIL: NaN detected in output")
