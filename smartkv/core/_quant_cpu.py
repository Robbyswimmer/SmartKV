import torch


def quantize_per_head(k_subset: torch.Tensor, v_subset: torch.Tensor, bits: int):
    max_val = 2 ** (bits - 1) - 1
    min_val = -2 ** (bits - 1)
    if bits == 1:
        max_val = 0
        min_val = 0

    k_abs_max = k_subset.abs().amax(dim=2, keepdim=True)
    k_scale_subset = k_abs_max / max(max_val, 1)
    k_scale_subset = torch.where(k_scale_subset == 0, torch.ones_like(k_scale_subset), k_scale_subset)
    k_q = torch.clamp(torch.round(k_subset / k_scale_subset), min_val, max_val).to(torch.int8)

    v_abs_max = v_subset.abs().amax(dim=2, keepdim=True)
    v_scale_subset = v_abs_max / max(max_val, 1)
    v_scale_subset = torch.where(v_scale_subset == 0, torch.ones_like(v_scale_subset), v_scale_subset)
    v_q = torch.clamp(torch.round(v_subset / v_scale_subset), min_val, max_val).to(torch.int8)

    return k_q, v_q, k_scale_subset.view(-1, k_subset.shape[1]), v_scale_subset.view(-1, v_subset.shape[1])
