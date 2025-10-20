import torch


def quantize_per_head(k_subset: torch.Tensor, v_subset: torch.Tensor, bits: int):
    """CPU reference implementation matching CUDA layout (N, H, D)."""

    def _prepare(t: torch.Tensor):
        if t.dim() == 3:
            n, h, d = t.shape
            return t, (n,), h, d
        if t.dim() == 4:
            b, n, h, d = t.shape
            reshaped = t.reshape(b * n, h, d)
            return reshaped, (b, n), h, d
        raise ValueError("Expected tensor with 3 or 4 dims (N,H,D) or (B,N,H,D)")

    k_prepared, batch_shape, num_heads, head_dim = _prepare(k_subset)
    v_prepared, _, _, _ = _prepare(v_subset)

    max_val = 2 ** (bits - 1) - 1
    min_val = -2 ** (bits - 1)
    if bits == 1:
        max_val = 0
        min_val = 0

    k_abs_max = k_prepared.abs().amax(dim=2, keepdim=True)
    k_scale_subset = k_abs_max / max(max_val, 1)
    k_scale_subset = torch.where(k_scale_subset == 0, torch.ones_like(k_scale_subset), k_scale_subset)
    k_q = torch.clamp(torch.round(k_prepared / k_scale_subset), min_val, max_val).to(torch.int8)

    v_abs_max = v_prepared.abs().amax(dim=2, keepdim=True)
    v_scale_subset = v_abs_max / max(max_val, 1)
    v_scale_subset = torch.where(v_scale_subset == 0, torch.ones_like(v_scale_subset), v_scale_subset)
    v_q = torch.clamp(torch.round(v_prepared / v_scale_subset), min_val, max_val).to(torch.int8)

    # Reshape back to original layout
    if batch_shape:
        if len(batch_shape) == 1:  # (N, H, D)
            n = batch_shape[0]
            k_q = k_q.view(n, num_heads, head_dim)
            v_q = v_q.view(n, num_heads, head_dim)
            k_scale = k_scale_subset.view(n, num_heads)
            v_scale = v_scale_subset.view(n, num_heads)
        else:  # (B, N, H, D)
            b, n = batch_shape
            k_q = k_q.view(b, n, num_heads, head_dim)
            v_q = v_q.view(b, n, num_heads, head_dim)
            k_scale = k_scale_subset.view(b, n, num_heads)
            v_scale = v_scale_subset.view(b, n, num_heads)
    else:
        k_scale = k_scale_subset.squeeze(-1)
        v_scale = v_scale_subset.squeeze(-1)

    return k_q, v_q, k_scale, v_scale
