"""CPU helper for streaming quantized attention."""

from typing import Optional, Tuple
import torch


def quantized_attention_streaming_cpu(
    query: torch.Tensor,
    k_qx: torch.Tensor,
    k_scale: torch.Tensor,
    v_qx: torch.Tensor,
    v_scale: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None,
    tile_size: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    current_k: Optional[torch.Tensor] = None,
    current_v: Optional[torch.Tensor] = None,
    current_mask: Optional[torch.Tensor] = None,
    return_attn: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Streaming attention that dequantizes KV tiles on the fly.

    Args:
        query: [B, H, q_len, d]
        k_qx: [N, H, d] int8
        k_scale: [N, H] float32
        v_qx: [N, H, d] int8
        v_scale: [N, H] float32
        causal_mask: Optional [B, 1, q_len, N]
        tile_size: optional explicit tile length
        dtype: compute dtype
        current_k/current_v: float tensors for the newest token (optional)
        return_attn: whether to reconstruct attention probabilities

    Returns:
        attn_output: [B, H, q_len, d]
        attn_probs (or None if return_attn False): [B, H, 1, N_total]
    """

    B, H, q_len, d = query.shape
    assert q_len == 1, "Streaming kernel currently assumes q_len==1 (decoding)."

    query = query.squeeze(2).to(dtype)
    k_qx = k_qx.to(query.device)
    k_scale = k_scale.to(query.device)
    v_qx = v_qx.to(query.device)
    v_scale = v_scale.to(query.device)
    N = k_qx.shape[0]

    # Choose tile size adaptively
    if tile_size is None or tile_size <= 0:
        target_tiles = 12
        tile_size = max(128, (N + target_tiles - 1) // target_tiles)

    output = torch.zeros((B, H, d), dtype=dtype, device=query.device)
    running_logit_max = torch.full((B, H), -torch.inf, dtype=dtype, device=query.device)
    running_exp_sum = torch.zeros((B, H), dtype=dtype, device=query.device)

    mask = None
    if causal_mask is not None and causal_mask.numel() > 0:
        assert causal_mask.shape[-1] == N, "Causal mask last dim must match KV length"
        mask = causal_mask.squeeze(2).to(query.device).to(dtype)

    inv_sqrt_d = 1.0 / (d ** 0.5)

    logits_tiles = [] if return_attn else None

    query_flat = query.reshape(B * H, d)

    for start in range(0, N, tile_size):
        end = min(start + tile_size, N)
        if end <= start:
            break

        k_tile = k_qx[start:end].to(dtype) * k_scale[start:end].unsqueeze(-1)
        v_tile = v_qx[start:end].to(dtype) * v_scale[start:end].unsqueeze(-1)

        tile_len = end - start
        logits = torch.empty((B, H, tile_len), dtype=dtype, device=query.device)
        k_tile_heads = k_tile.permute(1, 0, 2).contiguous()  # [H, tile, d]

        for h in range(H):
            qh = query_flat[h * B:(h + 1) * B]  # [B, d]
            kh = k_tile_heads[h].transpose(0, 1)  # [d, tile]
            logits[:, h, :] = (qh @ kh).to(dtype)

        logits *= inv_sqrt_d

        if mask is not None:
            logits = logits + mask[..., start:end]

        if return_attn:
            logits_tiles.append(logits.detach().cpu())

        tile_max = torch.amax(logits, dim=-1)
        new_max = torch.maximum(running_logit_max, tile_max)
        exp_scale = torch.exp(running_logit_max - new_max)

        running_exp_sum *= exp_scale
        output *= exp_scale.unsqueeze(-1)

        exp_logits = torch.exp(logits - new_max.unsqueeze(-1))
        running_exp_sum += torch.sum(exp_logits, dim=-1)

        v_tile_heads = v_tile.permute(1, 0, 2).contiguous()
        for h in range(H):
            exp_h = exp_logits[:, h, :]
            vh = v_tile_heads[h]
            output[:, h, :] += exp_h @ vh

        running_logit_max = new_max

        del k_tile, v_tile, logits, exp_logits, k_tile_heads, v_tile_heads

    current_logits_cpu = None
    if current_k is not None and current_v is not None:
        ck = current_k.to(dtype)
        cv = current_v.to(dtype)
        logits_cur = torch.einsum("bhd,bhd->bh", query, ck) * inv_sqrt_d
        if current_mask is not None and current_mask.numel() > 0:
            logits_cur = logits_cur + current_mask.squeeze(-1).squeeze(-1).to(dtype)

        tile_max = logits_cur
        new_max = torch.maximum(running_logit_max, tile_max)
        exp_scale = torch.exp(running_logit_max - new_max)

        running_exp_sum *= exp_scale
        output *= exp_scale.unsqueeze(-1)

        exp_logits_cur = torch.exp(logits_cur - new_max)
        running_exp_sum += exp_logits_cur
        output += exp_logits_cur.unsqueeze(-1) * cv

        running_logit_max = new_max

        if return_attn:
            current_logits_cpu = logits_cur.detach().cpu()

    output = output / running_exp_sum.unsqueeze(-1)
    attn_output = output.unsqueeze(2)

    attn_probs = None
    if return_attn:
        probs_list = []
        final_max_cpu = running_logit_max.detach().cpu()
        final_sum_cpu = running_exp_sum.detach().cpu()

        offset = 0
        for logits_cpu in logits_tiles or []:
            logits_cpu = logits_cpu.to(final_max_cpu.dtype)
            probs = torch.exp(logits_cpu - final_max_cpu.unsqueeze(-1))
            probs /= final_sum_cpu.unsqueeze(-1)
            probs_list.append(probs)
            offset += probs.shape[-1]

        if current_logits_cpu is not None:
            probs_cur = torch.exp(current_logits_cpu - final_max_cpu)
            probs_cur /= final_sum_cpu
            probs_list.append(probs_cur.unsqueeze(-1))

        if probs_list:
            probs = torch.cat(probs_list, dim=-1)
            attn_probs = probs.to(dtype).to(query.device).unsqueeze(2)
        else:
            attn_probs = torch.empty((B, H, 1, 0), dtype=dtype, device=query.device)

    return attn_output, attn_probs
