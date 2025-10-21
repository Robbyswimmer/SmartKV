"""
SmartKV Cache: Attention-guided adaptive precision KV-cache.

Main class that integrates quantizers, importance tracking, and precision
allocation to provide mixed-precision KV-cache compression.
"""

import heapq
import math
from collections import deque
import torch
from typing import Dict, List, Tuple, Optional, Any
import psutil
import os

from smartkv.core._quant_cpu import quantize_per_head

from smartkv.core.quantizers import get_quantizer, QuantizerBase
from smartkv.core.importance import ImportanceTracker
from smartkv.core.allocation import greedy_allocation, compute_memory_usage
from smartkv.core.forecast import ForecastPredictor

# Bit-packing support (GPU only)
try:
    from smartkv.kernels.bit_packing import pack_tensor, unpack_tensor, compute_packed_size, CUDA_AVAILABLE as PACKING_AVAILABLE
except ImportError:
    PACKING_AVAILABLE = False
    pack_tensor = None
    unpack_tensor = None
    def compute_packed_size(num_elements: int, bits: int) -> int:
        return num_elements


class SmartKVCache:
    """
    SmartKV cache with attention-guided adaptive precision.
    
    Dynamically allocates 2-8 bit precision to tokens based on their attention
    patterns, achieving better accuracy-memory tradeoffs than uniform quantization.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        memory_budget: float = 0.5,
        decay: float = 0.9,
        realloc_freq: int = 16,
        available_bits: List[int] = [2, 3, 4, 8],
        device: str = "cpu",
        special_token_ids: Optional[List[int]] = None,
        use_bit_packing: bool = True,
        enable_forecast: bool = False,
        forecast_history: int = 2048,
        forecast_update_interval: int = 32,
        forecast_blend: float = 0.5,
        forecast_lr: float = 0.05,
        utility_alpha: float = 0.5,
        importance_floor: float = 1e-6,
        hysteresis_rank_threshold: float = 0.05,
        hysteresis_intervals: int = 2,
        use_bucketed_layout: bool = False
    ):
        """
        Initialize SmartKV cache.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            memory_budget: Fraction of FP16 memory to use (e.g., 0.5 = 50%)
            decay: EMA decay for importance tracking
            realloc_freq: Reallocate precision every N tokens
            available_bits: Available bit-widths for quantization
            device: Device to store cache on ("cpu" or "cuda")
            special_token_ids: Token IDs to always keep at 8-bit (BOS, EOS, etc.)
            use_bit_packing: Enable bit-packing for sub-byte storage (GPU only, requires CUDA)
            utility_alpha: Exponent for diminishing returns in precision utility (0 < alpha ≤ 1)
            importance_floor: Minimum effective importance to keep spare budget useful
            hysteresis_rank_threshold: Minimum percentile rank change to consider reallocating a token
            hysteresis_intervals: Number of consecutive reallocations the threshold must be exceeded
            use_bucketed_layout: Enable experimental bucketed cache layout for GPU kernels
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.decay = decay
        self.realloc_freq = realloc_freq
        self.available_bits = sorted(available_bits, reverse=True)
        self.device = device

        # Storage configuration
        # Enable bit-packing only if requested, available, and on CUDA
        self.use_packing = use_bit_packing and PACKING_AVAILABLE and (device != "cpu")
        if use_bit_packing and not self.use_packing:
            import warnings
            if not PACKING_AVAILABLE:
                warnings.warn("Bit-packing requested but CUDA extension not available. Falling back to INT8 storage.")
            elif device == "cpu":
                warnings.warn("Bit-packing is only supported on CUDA devices. Falling back to INT8 storage.")

        self.scale_dtype = "fp32"  # FP32 scales (can be changed to "fp16")

        self.enable_forecast = bool(enable_forecast and forecast_history > 0)
        self.forecast_history = max(1, forecast_history)
        self.forecast_update_interval = max(1, forecast_update_interval)
        self.forecast_blend = float(max(0.0, min(1.0, forecast_blend)))
        self.forecast_lr = float(max(forecast_lr, 1e-5))
        self._forecast_feature_dim = 6
        if self.enable_forecast:
            self.forecast_feature_buffer = deque(maxlen=self.forecast_history)
            self.forecast_target_buffer = deque(maxlen=self.forecast_history)
            self.forecast_pending: Dict[int, Tuple[torch.Tensor, int]] = {}
            self.forecast_last_importance: Dict[int, float] = {}
            self.forecast_last_loss: Optional[float] = None
            self.forecast_predictor = ForecastPredictor(
                feature_dim=self._forecast_feature_dim,
                lr=self.forecast_lr,
            )
        else:
            self.forecast_feature_buffer = None
            self.forecast_target_buffer = None
            self.forecast_pending = {}
            self.forecast_last_importance = {}
            self.forecast_last_loss = None
            self.forecast_predictor = None

        self.utility_alpha = float(utility_alpha) if utility_alpha > 0 else 1.0
        self.importance_floor = float(max(importance_floor, 0.0))
        self.hysteresis_rank_threshold = float(max(hysteresis_rank_threshold, 0.0))
        self.hysteresis_intervals = max(int(hysteresis_intervals), 1)
        self._rank_state: Dict[int, Dict[str, float]] = {}
        self.use_bucketed_layout = bool(use_bucketed_layout)

        # Validate and adjust memory budget
        from smartkv.core.allocation import compute_minimum_budget
        self.min_budget = compute_minimum_budget(
            num_heads=num_heads,
            head_dim=head_dim,
            scale_dtype=self.scale_dtype,
            use_packing=self.use_packing
        )

        if memory_budget < self.min_budget:
            import warnings
            warnings.warn(
                f"Requested budget {memory_budget:.4f} is below minimum achievable "
                f"budget {self.min_budget:.4f} with current storage "
                f"(INT8 + {self.scale_dtype.upper()} scales, packing={self.use_packing}). "
                f"Clamping to minimum budget {self.min_budget:.4f} to avoid overallocation.",
                UserWarning
            )
            self.memory_budget = self.min_budget
        else:
            self.memory_budget = memory_budget

        # Special tokens always get 8-bit precision
        self.special_token_ids = set(special_token_ids or [])
        # Common special tokens that should never be low-precision
        self.protected_tokens = set([0, 1, 2])  # BOS, EOS, PAD typically

        self.head_importance = torch.ones(self.num_heads, dtype=torch.float32)
        self.head_importance_decay = 0.9
        self.recency_temperature = 512.0
        self.global_step = 0
        self.last_seen: Dict[int, int] = {}
        self.critical_token_window = 4
        self.critical_min_bits = max(self.available_bits)
        self.min_general_bits = min(self.available_bits)
        
        # Importance tracking
        self.importance_tracker = ImportanceTracker(
            num_layers=num_layers,
            decay=decay,
            device=device
        )
        
        # Precision mapping: token_id -> bits
        self.precision_map: Dict[int, int] = {}
        
        # Quantizers for each bit-width
        self.quantizers: Dict[int, QuantizerBase] = {
            bits: get_quantizer(bits) for bits in available_bits
        }
        
        # Cache storage: (layer, token_id) -> quantized data (metadata view)
        self.k_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.v_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Per-layer quantized buffers
        self.layer_store: List[Dict[str, Any]] = [
            self._init_layer_store() for _ in range(num_layers)
        ]

        # Metadata
        self.token_counter = 0
        self.realloc_counter = 0
        self.total_tokens_stored = 0
        
    def update_attention(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
        token_ids: List[int]
    ) -> None:
        """
        Update importance scores based on attention weights.

        Should be called after each attention computation.

        Args:
            layer_idx: Layer index
            attention_weights: Attention weights [batch, heads, queries, keys]
            token_ids: Token IDs corresponding to keys
        """
        if attention_weights is not None:
            head_dim = attention_weights.size(1)
            if self.head_importance.numel() != head_dim:
                self.head_importance = torch.ones(head_dim, dtype=torch.float32)

            if attention_weights.device.type != 'cpu':
                head_stats = attention_weights.detach().to('cpu')
            else:
                head_stats = attention_weights.detach()

            head_mean = head_stats.mean(dim=(0, 2, 3))
            self.head_importance = (
                self.head_importance * self.head_importance_decay
                + head_mean * (1.0 - self.head_importance_decay)
            )

            normalized = self.head_importance / (self.head_importance.mean() + 1e-6)
            # NOTE: This rescales attention weights by head importance without renormalizing.
            # The result is NOT a probability distribution (doesn't sum to 1), but serves
            # as a heuristic importance signal for precision allocation. If probability
            # semantics are needed, apply softmax after this operation.
            attention_weights = attention_weights * normalized.view(1, -1, 1, 1).to(attention_weights.device)

        self.importance_tracker.update_attention(
            layer_idx,
            attention_weights,
            token_ids
        )

        self.global_step += 1
        current_step = self.global_step
        for token_id in token_ids:
            self.last_seen[int(token_id)] = current_step
            if self.enable_forecast:
                actual_importance = self.importance_tracker.get_importance(int(token_id))
                self._record_forecast_target(int(token_id), actual_importance)

        # Adaptive reallocation frequency based on context length
        # Scale frequency with number of tokens to reduce overhead at long context
        num_tokens = len(token_ids)
        total_tokens = len(self.precision_map) + num_tokens
        layer_factor = 1 + layer_idx // 4
        adaptive_freq = max(self.realloc_freq * layer_factor, max(32, total_tokens // 8))

        # Periodically reallocate precision
        if self.token_counter % adaptive_freq == 0:
            self.allocate_precision(layer_idx, token_ids)
            self.realloc_counter += 1
    
    def allocate_precision(self, layer_idx: int, token_ids: List[int]) -> Dict[int, int]:
        """
        Allocate precision to tokens based on importance.

        Args:
            token_ids: List of token IDs to allocate precision for

        Returns:
            Dict mapping token_id -> allocated bits
        """
        if not token_ids:
            return {}

        candidate_tokens = set(token_ids)
        candidate_tokens.update(self.precision_map.keys())
        candidate_tokens.update(token_id for (_, token_id) in self.k_cache.keys())

        if not candidate_tokens:
            return {}

        protected_set = self.protected_tokens | self.special_token_ids

        importance_scores: Dict[int, float] = {}
        for tid in candidate_tokens:
            score = self.importance_tracker.get_importance(tid)
            if self.recency_temperature > 0:
                age = self.global_step - self.last_seen.get(tid, self.global_step)
                score *= math.exp(-age / self.recency_temperature)
            if tid in protected_set or tid < self.critical_token_window:
                score = float('inf')
            importance_scores[tid] = score

        if self.enable_forecast and self.forecast_predictor is not None:
            feature_batch = []
            token_batch = []
            raw_batch = []
            for tid, score in importance_scores.items():
                if not math.isfinite(score) or score <= 0.0:
                    continue
                feature = self._build_forecast_feature(tid, score)
                feature_batch.append(feature)
                token_batch.append(tid)
                raw_batch.append(score)
                self.forecast_pending[tid] = (feature, self.global_step)
            if feature_batch:
                stacked = torch.stack(feature_batch)
                preds = self.forecast_predictor.predict(stacked).cpu()
                for tid, raw_score, pred in zip(token_batch, raw_batch, preds.tolist()):
                    blended = (1.0 - self.forecast_blend) * raw_score + self.forecast_blend * max(pred, 0.0)
                    importance_scores[tid] = blended
                    self.forecast_last_importance[tid] = raw_score

        if self.enable_forecast:
            for tid, score in importance_scores.items():
                if math.isfinite(score):
                    self.forecast_last_importance[tid] = score

        allocation = self._solve_allocation(candidate_tokens, importance_scores, protected_set)

        previous_map = dict(self.precision_map)
        self.precision_map = allocation

        changed_tokens = {
            tid for tid, bits in allocation.items()
            if previous_map.get(tid) != bits
        }

        if changed_tokens:
            self._requantize_tokens(changed_tokens)

        return {tid: allocation[tid] for tid in token_ids if tid in allocation}

    def _solve_allocation(
        self,
        tokens: set,
        importance_scores: Dict[int, float],
        protected_set: set
    ) -> Dict[int, int]:
        available_bits_desc = sorted(self.available_bits, reverse=True)
        available_bits_asc = list(reversed(available_bits_desc))
        min_bits = available_bits_asc[0]
        max_bits = available_bits_desc[0]

        allocation = {tid: min_bits for tid in tokens}

        for tid in tokens:
            if tid in protected_set or tid < self.critical_token_window:
                allocation[tid] = max_bits

        num_tokens = len(allocation)
        if num_tokens == 0:
            return allocation

        scale_bits = 32 if self.scale_dtype == "fp32" else 16
        elements_per_token = 2 * self.num_heads * self.head_dim

        if self.use_packing:
            current_payload_bits = float(sum(bits * elements_per_token for bits in allocation.values()))
        else:
            current_payload_bits = float(num_tokens * elements_per_token * 8)

        scale_bits_actual = float(num_tokens * 2 * self.num_heads * scale_bits)
        fp16_bits = float(num_tokens * elements_per_token * 16)

        def compute_ratio(payload_bits: float) -> float:
            if fp16_bits <= 0:
                return 0.0
            return (payload_bits + scale_bits_actual) / fp16_bits

        utility_alpha = min(max(self.utility_alpha, 0.1), 1.0)

        def utility(bits: int) -> float:
            if utility_alpha == 1.0:
                return float(bits)
            return float(math.pow(bits, utility_alpha))

        # Compute rank percentiles for hysteresis thresholding
        ranked_tokens = sorted(
            (tid for tid in tokens if tid not in protected_set and tid >= self.critical_token_window),
            key=lambda tid: importance_scores.get(tid, 0.0),
            reverse=True
        )
        rank_percent: Dict[int, float] = {}
        if ranked_tokens:
            denom = max(len(ranked_tokens) - 1, 1)
            for idx, tid in enumerate(ranked_tokens):
                pct = 0.0 if len(ranked_tokens) == 1 else idx / denom
                state = self._rank_state.get(tid)
                if state is None:
                    self._rank_state[tid] = {"pct": pct, "streak": self.hysteresis_intervals}
                else:
                    prev_pct = state.get("pct", pct)
                    delta = abs(pct - prev_pct)
                    if delta >= self.hysteresis_rank_threshold:
                        streak = min(state.get("streak", 0) + 1, self.hysteresis_intervals)
                    else:
                        streak = 0
                    self._rank_state[tid] = {"pct": pct, "streak": streak}
                rank_percent[tid] = pct
        # Ensure protected or out-of-window tokens have state entries for completeness
        for tid in tokens:
            if tid in rank_percent:
                continue
            if tid not in self._rank_state:
                self._rank_state[tid] = {"pct": 1.0, "streak": self.hysteresis_intervals}

        bit_index = {bits: idx for idx, bits in enumerate(available_bits_asc)}
        heap: List[Tuple[float, int, int, int]] = []

        def enqueue(token_id: int, current_bits: int) -> None:
            idx = bit_index.get(current_bits)
            if idx is None or idx >= len(available_bits_asc) - 1:
                return
            next_bits = available_bits_asc[idx + 1]
            importance = importance_scores.get(token_id, 0.0)
            if not math.isfinite(importance):
                importance = self.importance_floor if self.importance_floor > 0 else 0.0
            if importance <= 0.0:
                if self.importance_floor <= 0.0:
                    return
                importance = self.importance_floor
            state = self._rank_state.get(token_id)
            if state is not None and token_id in rank_percent:
                enforce = compute_ratio(current_payload_bits) >= self.memory_budget * 0.85
                if enforce and state.get("streak", 0) < self.hysteresis_intervals:
                    return
            util_gain = utility(next_bits) - utility(current_bits)
            if util_gain <= 0.0:
                return
            delta_bits = next_bits - current_bits
            if delta_bits <= 0:
                return
            score = (importance * util_gain) / delta_bits
            # max-heap via negated score
            heapq.heappush(heap, (-score, token_id, current_bits, next_bits))

        for tid in allocation:
            if allocation[tid] < max_bits:
                enqueue(tid, allocation[tid])

        tol = 1e-6
        current_ratio = compute_ratio(current_payload_bits)

        while heap:
            _score_neg, tid, expected_bits, next_bits = heapq.heappop(heap)
            current_bits = allocation.get(tid, min_bits)
            if current_bits != expected_bits:
                # Stale entry; skip
                continue

            delta_payload = 0.0
            if self.use_packing:
                delta_payload = float((next_bits - current_bits) * elements_per_token)

            ratio_candidate = compute_ratio(current_payload_bits + delta_payload)
            if ratio_candidate > self.memory_budget + tol:
                # Cannot afford this upgrade; skip and try other candidates
                continue

            allocation[tid] = next_bits
            current_payload_bits += delta_payload
            current_ratio = ratio_candidate

            state = self._rank_state.get(tid)
            if state is not None:
                state["streak"] = 0
                state["pct"] = rank_percent.get(tid, state.get("pct", 1.0))
                self._rank_state[tid] = state

            enqueue(tid, next_bits)

            if current_ratio >= self.memory_budget + tol:
                break

        ratio, stats = self._allocation_stats(allocation)
        if hasattr(self, '_logger') and self._logger:
            self._logger.debug(
                "Allocation stats: "
                f"True={ratio:.3f} (target {self.memory_budget:.3f}), "
                f"avg_bits={stats.get('avg_bits', 0.0):.2f}, tokens={stats.get('num_tokens', 0)}"
            )

        return allocation

    def _allocation_stats(self, allocation: Dict[int, int]) -> Tuple[float, Dict[str, float]]:
        stats = compute_memory_usage(
            allocation,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale_dtype=self.scale_dtype,
            use_packing=self.use_packing
        )
        return stats.get('memory_ratio_true', 0.0), stats

    def _remove_slot_from_buckets(self, layer_store: Dict[str, Any], slot: int) -> None:
        if not self.use_bucketed_layout:
            return
        buckets = layer_store.get('buckets')
        if not buckets:
            return
        for bucket in buckets.values():
            bucket['indices'].discard(int(slot))
            bucket['slot_map'].pop(int(slot), None)

    def _register_bucket(self, layer_store: Dict[str, Any], slot: int, bits: int, bucket_index: Optional[int] = None) -> None:
        if not self.use_bucketed_layout:
            return
        buckets = layer_store.get('buckets')
        if not buckets or bits not in buckets:
            return
        self._remove_slot_from_buckets(layer_store, slot)
        buckets[bits]['indices'].add(int(slot))
        if bucket_index is None:
            bucket_index = buckets[bits]['slot_map'].get(int(slot))
        if bucket_index is None:
            buffers = layer_store['bucket_buffers'][bits]
            bucket_index = buffers['size'] - 1
        buckets[bits]['slot_map'][int(slot)] = int(bucket_index)

    def _move_bucket_entry(
        self,
        layer_store: Dict[str, Any],
        slot: int,
        old_bits: int,
        new_bits: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor
    ) -> None:
        if not self.use_bucketed_layout:
            return
        buckets = layer_store.get('buckets')
        if not buckets:
            return
        if old_bits in buckets:
            self._pop_bucket_entry(layer_store, old_bits, slot)
        token_id = int(layer_store['token_ids'][slot].item()) if layer_store['token_ids'] is not None else int(slot)
        self._append_bucket_entry(layer_store, new_bits, token_id, slot, k_tensor, v_tensor, k_scale, v_scale)

    def _get_bucket_indices(self, layer_store: Dict[str, Any], bits: int) -> torch.Tensor:
        if not self.use_bucketed_layout:
            return torch.empty(0, dtype=torch.long, device=self.device)
        buckets = layer_store.get('buckets')
        if not buckets or bits not in buckets:
            return torch.empty(0, dtype=torch.long, device=self.device)
        indices = buckets[bits]['indices']
        if not indices:
            return torch.empty(0, dtype=torch.long, device=self.device)
        sorted_slots = sorted(indices)
        return torch.tensor(sorted_slots, dtype=torch.long, device=self.device)

    def get_bucket_views(self, layer_idx: int) -> Dict[int, Dict[str, torch.Tensor]]:
        """Return per-bit bucket tensors for a layer (experimental)."""
        if not self.use_bucketed_layout:
            raise RuntimeError("Bucketed layout disabled")

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError("Invalid layer index")

        layer_store = self.layer_store[layer_idx]
        results: Dict[int, Dict[str, torch.Tensor]] = {}

        for bits in self.available_bits:
            buffers = layer_store['bucket_buffers'][bits]
            size = buffers['size'] if buffers else 0
            if size == 0:
                continue
            device = buffers['token_ids'].device

            # Extract tensors
            k_scale = buffers['k_scale'][:size].clone()
            v_scale = buffers['v_scale'][:size].clone()
            global_slots = buffers['global_slots'][:size].clone()

            # Transpose scales if needed: [H, num_tokens] -> [num_tokens, H]
            # Scales should be [num_tokens, H] for kernel
            if k_scale.dim() == 2:
                if k_scale.shape[0] == self.num_heads and k_scale.shape[1] == size:
                    # Need transpose: [H, num_tokens] -> [num_tokens, H]
                    k_scale = k_scale.transpose(0, 1).contiguous()
                    v_scale = v_scale.transpose(0, 1).contiguous()
                elif k_scale.shape[0] == size and k_scale.shape[1] == self.num_heads:
                    # Already correct shape
                    pass

            # Sort by global_slots for sequential access
            if global_slots.numel() > 1:
                order = torch.argsort(global_slots)
                global_slots = global_slots.index_select(0, order).contiguous()
                k_scale = k_scale.index_select(0, order).contiguous()
                v_scale = v_scale.index_select(0, order).contiguous()

                # Sort k_qx and v_qx as well
                if self.use_packing and bits < 8 and PACKING_AVAILABLE:
                    k_qx_sorted = buffers['k_qx'][:size].index_select(0, order).clone()
                    v_qx_sorted = buffers['v_qx'][:size].index_select(0, order).clone()
                else:
                    k_qx_sorted = buffers['k_qx'][:size].index_select(0, order).clone()
                    v_qx_sorted = buffers['v_qx'][:size].index_select(0, order).clone()

                token_ids_sorted = buffers['token_ids'][:size].index_select(0, order).clone()
            else:
                k_qx_sorted = buffers['k_qx'][:size].clone()
                v_qx_sorted = buffers['v_qx'][:size].clone()
                token_ids_sorted = buffers['token_ids'][:size].clone()

            views: Dict[str, torch.Tensor] = {
                'token_ids': token_ids_sorted,
                'global_slots': global_slots,
                'k_scale': k_scale,
                'v_scale': v_scale,
                'k_qx': k_qx_sorted,
                'v_qx': v_qx_sorted,
            }

            if self.use_packing and bits < 8 and PACKING_AVAILABLE:
                views['packed_dim'] = torch.tensor(buffers['packed_dim'], device=device)
                views['packed'] = torch.tensor(1, device=device)
            else:
                views['packed_dim'] = torch.tensor(self.head_dim, device=device)
                views['packed'] = torch.tensor(0, device=device)

            results[bits] = views

        return results

    def _build_forecast_feature(self, token_id: int, raw_importance: float) -> torch.Tensor:
        if not self.enable_forecast:
            raise RuntimeError("Forecasting disabled")

        last_importance = self.forecast_last_importance.get(token_id, 0.0)
        delta = raw_importance - last_importance

        age = self.global_step - self.last_seen.get(token_id, self.global_step)
        age_norm = math.tanh(age / max(self.recency_temperature, 1.0))

        current_bits = float(self.precision_map.get(token_id, self.min_general_bits))
        bits_norm = current_bits / max(self.available_bits)

        head_mean = float(self.head_importance.mean().item()) if isinstance(self.head_importance, torch.Tensor) else float(self.head_importance)
        head_norm = math.tanh(head_mean)

        position_norm = math.tanh(token_id / max(1.0, self.total_tokens_stored + 1))
        step_norm = math.tanh(self.global_step / max(1.0, self.recency_temperature * 10.0))

        feature = torch.tensor([
            math.log1p(max(raw_importance, 0.0)),
            math.log1p(abs(delta)),
            age_norm,
            bits_norm,
            head_norm,
            position_norm + step_norm,
        ], dtype=torch.float32)
        return feature

    def _record_forecast_target(self, token_id: int, actual_importance: float) -> None:
        if not self.enable_forecast:
            return
        pending = self.forecast_pending.pop(token_id, None)
        if pending is None:
            return

        feature, step = pending
        if not math.isfinite(actual_importance):
            return

        target = math.log1p(max(actual_importance, 0.0))
        self.forecast_feature_buffer.append(feature.detach().cpu())
        self.forecast_target_buffer.append(target)

        if len(self.forecast_feature_buffer) >= self.forecast_update_interval:
            self._update_forecast_predictor()

    def _update_forecast_predictor(self) -> None:
        if not self.enable_forecast or self.forecast_predictor is None:
            return
        if not self.forecast_feature_buffer:
            return

        features = torch.stack(list(self.forecast_feature_buffer))
        targets = torch.tensor(list(self.forecast_target_buffer), dtype=torch.float32)
        loss = self.forecast_predictor.update(features, targets)
        self.forecast_last_loss = loss
        self.forecast_feature_buffer.clear()
        self.forecast_target_buffer.clear()

    def _requantize_tokens(self, tokens: set) -> None:
        if not tokens:
            return

        cache_items = list(self.k_cache.items())
        for (layer_idx, token_id), meta in cache_items:
            if token_id not in tokens:
                continue
            new_bits = self.precision_map.get(token_id)
            if new_bits is None:
                continue
            self._requantize_single(layer_idx, token_id, int(new_bits))

    def _requantize_single(self, layer_idx: int, token_id: int, bits: int) -> None:
        cache_key = (layer_idx, token_id)
        meta = self.k_cache.get(cache_key)
        if meta is None:
            return

        layer_store = self.layer_store[layer_idx]
        slot = int(meta['index'])
        indices = torch.tensor([slot], dtype=torch.long)
        keys, values = self._dequantize_indices(layer_store, indices)
        if keys.numel() == 0:
            return

        k_float = keys[0].to(self.device)
        v_float = values[0].to(self.device)

        prev_bits = int(meta.get('bits', layer_store['bits'][slot].item() if layer_store['bits'] is not None else bits))
        k_q, v_q, k_scale, v_scale = self._quantize_kv_pair(k_float, v_float, bits)
        self._write_quantized_slot(layer_store, slot, bits, k_q, v_q, k_scale, v_scale, prev_bits=prev_bits)

        meta['bits'] = bits
        meta['qx'] = layer_store['k_qx'][slot]
        meta['scale'] = layer_store['k_scale'][slot]
        self.v_cache[cache_key]['bits'] = bits
        self.v_cache[cache_key]['qx'] = layer_store['v_qx'][slot]
        self.v_cache[cache_key]['scale'] = layer_store['v_scale'][slot]

    def _quantize_kv_pair(
        self,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k_input = k_tensor.unsqueeze(0)
        v_input = v_tensor.unsqueeze(0)

        if self.device == 'cuda':
            try:
                from smartkv.core._quant_cuda import quantize_per_head_cuda
                k_q, v_q, k_scale, v_scale = quantize_per_head_cuda(k_input, v_input, bits)
            except ImportError:
                k_cpu = k_input.cpu()
                v_cpu = v_input.cpu()
                k_q, v_q, k_scale, v_scale = quantize_per_head(k_cpu, v_cpu, bits)
                k_q = k_q.to(self.device)
                v_q = v_q.to(self.device)
                k_scale = k_scale.to(self.device)
                v_scale = v_scale.to(self.device)
        else:
            k_q, v_q, k_scale, v_scale = quantize_per_head(k_input, v_input, bits)

        return (
            k_q.squeeze(0),
            v_q.squeeze(0),
            k_scale.squeeze(0),
            v_scale.squeeze(0)
        )

    def _write_quantized_slot(
        self,
        layer_store: Dict[str, Any],
        slot: int,
        bits: int,
        k_q: torch.Tensor,
        v_q: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        prev_bits: Optional[int] = None
    ) -> None:
        if prev_bits is None:
            prev_bits = bits
        # Ensure tensors are on correct device
        k_device = layer_store['k_qx'].device if layer_store['k_qx'] is not None else self.device
        v_device = layer_store['v_qx'].device if layer_store['v_qx'] is not None else self.device

        k_scale_device = layer_store['k_scale'].device if layer_store['k_scale'] is not None else self.device
        v_scale_device = layer_store['v_scale'].device if layer_store['v_scale'] is not None else self.device

        layer_store['k_scale'][slot] = k_scale.to(k_scale_device)
        layer_store['v_scale'][slot] = v_scale.to(v_scale_device)
        layer_store['bits'][slot] = bits

        if self.use_packing and bits < 8:
            k_vec = k_q.contiguous().unsqueeze(0)
            v_vec = v_q.contiguous().unsqueeze(0)

            layer_store['k_packed'][slot] = pack_tensor(k_vec.flatten().to(k_device), bits)
            layer_store['v_packed'][slot] = pack_tensor(v_vec.flatten().to(v_device), bits)
            layer_store['k_shapes'][slot] = tuple(k_vec.shape)
            layer_store['v_shapes'][slot] = tuple(v_vec.shape)

            layer_store['k_qx'][slot] = torch.zeros_like(layer_store['k_qx'][slot])
            layer_store['v_qx'][slot] = torch.zeros_like(layer_store['v_qx'][slot])
        else:
            if self.use_packing:
                layer_store['k_packed'].pop(slot, None)
                layer_store['v_packed'].pop(slot, None)
                layer_store['k_shapes'].pop(slot, None)
                layer_store['v_shapes'].pop(slot, None)

            layer_store['k_qx'][slot] = k_q.to(k_device)
            layer_store['v_qx'][slot] = v_q.to(v_device)

        if self.use_bucketed_layout:
            if prev_bits == bits:
                self._update_bucket_entry(layer_store, bits, slot, k_q, v_q, k_scale, v_scale)
            else:
                self._move_bucket_entry(layer_store, slot, prev_bits, bits, k_q, v_q, k_scale, v_scale)

    def _init_layer_store(self) -> Dict[str, Any]:
        """Create empty storage buffers for a layer."""
        store = {
            'capacity': 0,
            'size': 0,
            'token_ids': None,
            'k_qx': None,
            'v_qx': None,
            'k_scale': None,
            'v_scale': None,
            'bits': None,
            'buckets': self._create_bucket_dict() if self.use_bucketed_layout else None,
            'bucket_buffers': self._create_bucket_buffers() if self.use_bucketed_layout else None,
        }

        # Bit-packing fields (optional)
        if self.use_packing:
            store['k_packed'] = {}  # Dict[int, torch.Tensor] - slot -> packed tensor
            store['v_packed'] = {}  # Dict[int, torch.Tensor] - slot -> packed tensor
            store['k_shapes'] = {}  # Dict[int, tuple] - slot -> original shape
            store['v_shapes'] = {}  # Dict[int, tuple] - slot -> original shape

        return store

    def _create_bucket_dict(self) -> Dict[int, Dict[str, Any]]:
        return {
            bits: {
                'indices': set(),
                'slot_map': {},  # global slot -> bucket index
            }
            for bits in self.available_bits
        }

    def _create_bucket_buffers(self) -> Dict[int, Dict[str, Any]]:
        return {
            bits: {
                'k_qx': None,
                'v_qx': None,
                'k_scale': None,
                'v_scale': None,
                'token_ids': None,
                'global_slots': None,
                'capacity': 0,
                'size': 0,
                'packed_dim': self.head_dim if not (self.use_packing and bits < 8) else compute_packed_size(self.head_dim, bits),
            }
            for bits in self.available_bits
        }

    def _ensure_layer_capacity(self, layer_store: Dict[str, Any], min_capacity: int) -> None:
        """Ensure buffers can hold at least `min_capacity` tokens."""
        if layer_store['capacity'] >= min_capacity:
            return

        new_capacity = max(64, layer_store['capacity'] * 2 if layer_store['capacity'] else 64)
        while new_capacity < min_capacity:
            new_capacity *= 2

        def _grow_tensor(tensor: Optional[torch.Tensor], shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            # Determine device from existing tensor or use self.device
            device = tensor.device if tensor is not None else self.device

            if tensor is None:
                return torch.empty(shape, dtype=dtype, device=device)
            new_tensor = torch.empty(shape, dtype=dtype, device=device)
            prev = tensor.shape[0]
            if prev > 0:
                new_tensor[:prev] = tensor
            return new_tensor

        layer_store['token_ids'] = _grow_tensor(layer_store['token_ids'], (new_capacity,), torch.long)
        layer_store['k_qx'] = _grow_tensor(
            layer_store['k_qx'],
            (new_capacity, self.num_heads, self.head_dim),
            torch.int8,
        )
        layer_store['v_qx'] = _grow_tensor(
            layer_store['v_qx'],
            (new_capacity, self.num_heads, self.head_dim),
            torch.int8,
        )
        layer_store['k_scale'] = _grow_tensor(layer_store['k_scale'], (new_capacity, self.num_heads), torch.float32)
        layer_store['v_scale'] = _grow_tensor(layer_store['v_scale'], (new_capacity, self.num_heads), torch.float32)
        layer_store['bits'] = _grow_tensor(layer_store['bits'], (new_capacity,), torch.uint8)
        layer_store['capacity'] = new_capacity

    def _ensure_bucket_capacity(self, layer_store: Dict[str, Any], bits: int, min_capacity: int) -> None:
        if not self.use_bucketed_layout:
            return
        buffers = layer_store['bucket_buffers'][bits]
        if buffers['capacity'] >= min_capacity:
            return

        new_capacity = max(32, buffers['capacity'] * 2 if buffers['capacity'] else 32)
        while new_capacity < min_capacity:
            new_capacity *= 2

        device = self.device
        packed_dim = buffers['packed_dim']
        payload_dtype = torch.uint8 if self.use_packing and bits < 8 else torch.int8

        def grow(tensor: Optional[torch.Tensor], shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            if tensor is None:
                return torch.empty(shape, dtype=dtype, device=device)
            new_tensor = torch.empty(shape, dtype=dtype, device=device)
            prev = tensor.shape[0]
            if prev > 0:
                new_tensor[:prev] = tensor
            return new_tensor

        buffers['token_ids'] = grow(buffers['token_ids'], (new_capacity,), torch.long)
        buffers['global_slots'] = grow(buffers['global_slots'], (new_capacity,), torch.long)
        buffers['k_qx'] = grow(buffers['k_qx'], (new_capacity, self.num_heads, packed_dim), payload_dtype)
        buffers['v_qx'] = grow(buffers['v_qx'], (new_capacity, self.num_heads, packed_dim), payload_dtype)
        buffers['k_scale'] = grow(buffers['k_scale'], (new_capacity, self.num_heads), torch.float32)
        buffers['v_scale'] = grow(buffers['v_scale'], (new_capacity, self.num_heads), torch.float32)
        buffers['capacity'] = new_capacity

    def _append_bucket_entry(
        self,
        layer_store: Dict[str, Any],
        bits: int,
        token_id: int,
        slot: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor
    ) -> None:
        if not self.use_bucketed_layout:
            return
        buffers = layer_store['bucket_buffers'][bits]
        self._ensure_bucket_capacity(layer_store, bits, buffers['size'] + 1)

        idx = buffers['size']
        device = buffers['token_ids'].device
        buffers['token_ids'][idx] = int(token_id)
        buffers['global_slots'][idx] = int(slot)

        if self.use_packing and bits < 8 and pack_tensor is not None:
            packed_dim = buffers['packed_dim']
            k_packed = pack_tensor(k_tensor.contiguous(), bits).reshape(self.num_heads, packed_dim)
            v_packed = pack_tensor(v_tensor.contiguous(), bits).reshape(self.num_heads, packed_dim)
            buffers['k_qx'][idx] = k_packed.to(device)
            buffers['v_qx'][idx] = v_packed.to(device)
        else:
            buffers['k_qx'][idx] = k_tensor.to(device)
            buffers['v_qx'][idx] = v_tensor.to(device)

        buffers['k_scale'][idx] = k_scale.to(device)
        buffers['v_scale'][idx] = v_scale.to(device)
        buffers['size'] += 1

        buckets = layer_store['buckets'][bits]
        buckets['indices'].add(int(slot))
        buckets['slot_map'][int(slot)] = idx

    def _update_bucket_entry(
        self,
        layer_store: Dict[str, Any],
        bits: int,
        slot: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor
    ) -> None:
        if not self.use_bucketed_layout:
            return
        buckets = layer_store['buckets'][bits]
        buffers = layer_store['bucket_buffers'][bits]
        bucket_idx = buckets['slot_map'].get(int(slot))
        if bucket_idx is None:
            token_id = int(layer_store['token_ids'][slot].item()) if layer_store['token_ids'] is not None else int(slot)
            self._append_bucket_entry(layer_store, bits, token_id, slot, k_tensor, v_tensor, k_scale, v_scale)
            return
        device = buffers['token_ids'].device
        if self.use_packing and bits < 8 and pack_tensor is not None:
            packed_dim = buffers['packed_dim']
            k_packed = pack_tensor(k_tensor.contiguous(), bits).reshape(self.num_heads, packed_dim)
            v_packed = pack_tensor(v_tensor.contiguous(), bits).reshape(self.num_heads, packed_dim)
            buffers['k_qx'][bucket_idx] = k_packed.to(device)
            buffers['v_qx'][bucket_idx] = v_packed.to(device)
        else:
            buffers['k_qx'][bucket_idx] = k_tensor.to(device)
            buffers['v_qx'][bucket_idx] = v_tensor.to(device)
        buffers['k_scale'][bucket_idx] = k_scale.to(device)
        buffers['v_scale'][bucket_idx] = v_scale.to(device)
        if layer_store['token_ids'] is not None:
            buffers['token_ids'][bucket_idx] = int(layer_store['token_ids'][slot].item())
        buffers['global_slots'][bucket_idx] = int(slot)
        self._register_bucket(layer_store, slot, bits, bucket_idx=bucket_idx)

    def _pop_bucket_entry(
        self,
        layer_store: Dict[str, Any],
        bits: int,
        slot: int
    ) -> Optional[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not self.use_bucketed_layout:
            return None
        buckets = layer_store['buckets'][bits]
        buffers = layer_store['bucket_buffers'][bits]
        bucket_idx = buckets['slot_map'].get(int(slot))
        if bucket_idx is None:
            return None

        size = buffers['size']
        last_idx = size - 1

        token_id = int(buffers['token_ids'][bucket_idx].item())
        k_tensor = buffers['k_qx'][bucket_idx].clone()
        v_tensor = buffers['v_qx'][bucket_idx].clone()
        k_scale = buffers['k_scale'][bucket_idx].clone()
        v_scale = buffers['v_scale'][bucket_idx].clone()

        if bucket_idx != last_idx:
            # move last element into removed slot
            buffers['token_ids'][bucket_idx] = buffers['token_ids'][last_idx]
            buffers['global_slots'][bucket_idx] = buffers['global_slots'][last_idx]
            buffers['k_qx'][bucket_idx] = buffers['k_qx'][last_idx]
            buffers['v_qx'][bucket_idx] = buffers['v_qx'][last_idx]
            buffers['k_scale'][bucket_idx] = buffers['k_scale'][last_idx]
            buffers['v_scale'][bucket_idx] = buffers['v_scale'][last_idx]

            moved_slot = int(buffers['global_slots'][bucket_idx].item())
            buckets['slot_map'][moved_slot] = bucket_idx
        buffers['size'] = last_idx

        # clean up last slot (optional)
        buckets['slot_map'].pop(int(slot), None)
        buckets['indices'].discard(int(slot))

        return token_id, k_tensor, v_tensor, k_scale, v_scale
    
    def quantize_and_store(
        self,
        layer_idx: int,
        token_id: int,
        k_vec: torch.Tensor,
        v_vec: torch.Tensor
    ) -> None:
        """Quantize and store a single KV vector."""
        if k_vec.dim() == 1:
            k_vec = k_vec.unsqueeze(0)
        if v_vec.dim() == 1:
            v_vec = v_vec.unsqueeze(0)
        if k_vec.dim() == 2:
            k_vec = k_vec.unsqueeze(0)
        if v_vec.dim() == 2:
            v_vec = v_vec.unsqueeze(0)

        self.quantize_and_store_batch(
            layer_idx,
            [int(token_id)],
            k_vec,
            v_vec,
        )

    def quantize_and_store_batch(
        self,
        layer_idx: int,
        token_ids: List[int],
        k_batch: torch.Tensor,
        v_batch: torch.Tensor
    ) -> None:
        """Quantize and store a batch of KV vectors for a layer."""
        if not token_ids:
            return

        layer_store = self.layer_store[layer_idx]

        # Keep tensors on target device (CPU or CUDA)
        k_device = k_batch.detach().to(torch.float32)
        v_device = v_batch.detach().to(torch.float32)

        # Move to CPU only if device is CPU, otherwise keep on GPU
        if self.device == "cpu":
            if k_device.device.type != "cpu":
                k_device = k_device.cpu()
            if v_device.device.type != "cpu":
                v_device = v_device.cpu()

        seq_len = k_device.shape[0]
        assert seq_len == len(token_ids), "Mismatched KV batch length"

        bits_list = [self.precision_map.get(token_id, 4) for token_id in token_ids]
        bits_tensor = torch.tensor(bits_list, dtype=torch.uint8, device=k_device.device)

        # Allocate quantized buffers on same device as input
        k_qx = torch.empty((seq_len, self.num_heads, self.head_dim), dtype=torch.int8, device=k_device.device)
        v_qx = torch.empty((seq_len, self.num_heads, self.head_dim), dtype=torch.int8, device=k_device.device)
        k_scale = torch.empty((seq_len, self.num_heads), dtype=torch.float32, device=k_device.device)
        v_scale = torch.empty((seq_len, self.num_heads), dtype=torch.float32, device=k_device.device)

        unique_bits = bits_tensor.unique(sorted=True)
        for bits in unique_bits.tolist():
            mask = bits_tensor == bits
            if not bool(mask.any()):
                continue

            k_subset = k_device[mask]
            v_subset = v_device[mask]

            max_val = 2 ** (bits - 1) - 1
            min_val = -2 ** (bits - 1)
            if bits == 1:
                max_val = 0
                min_val = 0

            # Quantize on device (CPU or CUDA)
            if k_subset.device.type == 'cuda':
                # Use CUDA quantization if available
                try:
                    from smartkv.core._quant_cuda import quantize_per_head_cuda
                    k_q, v_q, k_scale_subset, v_scale_subset = quantize_per_head_cuda(k_subset, v_subset, bits)
                except ImportError:
                    # Fallback to CPU quantization
                    k_subset_cpu = k_subset.cpu()
                    v_subset_cpu = v_subset.cpu()
                    k_q, v_q, k_scale_subset, v_scale_subset = quantize_per_head(k_subset_cpu, v_subset_cpu, bits)
                    k_q = k_q.to(k_subset.device)
                    v_q = v_q.to(k_subset.device)
                    k_scale_subset = k_scale_subset.to(k_subset.device)
                    v_scale_subset = v_scale_subset.to(k_subset.device)
            else:
                # CPU quantization
                k_q, v_q, k_scale_subset, v_scale_subset = quantize_per_head(k_subset, v_subset, bits)

            k_qx[mask] = k_q
            v_qx[mask] = v_q
            k_scale[mask] = k_scale_subset
            v_scale[mask] = v_scale_subset

        for idx, token_id in enumerate(token_ids):
            token_int = int(token_id)
            if token_int not in self.precision_map:
                self.precision_map[token_int] = int(bits_tensor[idx])

        start = layer_store['size']
        new_size = start + seq_len
        self._ensure_layer_capacity(layer_store, new_size)

        token_id_tensor = torch.tensor(token_ids, dtype=torch.long)
        # NOTE: We append tokens sequentially, assuming token_ids arrive in
        # monotonically increasing order (typical for autoregressive generation).
        # If token_ids are not sorted, torch.searchsorted in retrieve/retrieve_all
        # will fail. For non-sequential access patterns, use a dict-based index.
        layer_store['token_ids'][start:new_size] = token_id_tensor

        # Store quantized values (with optional bit-packing)
        if self.use_packing:
            # Store packed data for each token individually
            for idx in range(seq_len):
                slot = start + idx
                bits = int(bits_tensor[idx])

                # Pack if bits < 8, otherwise store as INT8
                if bits < 8:
                    k_vec = k_qx[idx].unsqueeze(0)  # [1, H, D]
                    v_vec = v_qx[idx].unsqueeze(0)  # [1, H, D]

                    # Pack to compressed format
                    k_packed = pack_tensor(k_vec.flatten(), bits)
                    v_packed = pack_tensor(v_vec.flatten(), bits)

                    layer_store['k_packed'][slot] = k_packed
                    layer_store['v_packed'][slot] = v_packed
                    layer_store['k_shapes'][slot] = tuple(k_vec.shape)
                    layer_store['v_shapes'][slot] = tuple(v_vec.shape)

                    # Store dummy INT8 values (won't be used)
                    layer_store['k_qx'][slot] = torch.zeros_like(k_qx[idx])
                    layer_store['v_qx'][slot] = torch.zeros_like(v_qx[idx])
                else:
                    # 8-bit: store directly as INT8 (no packing needed)
                    layer_store['k_qx'][slot] = k_qx[idx]
                    layer_store['v_qx'][slot] = v_qx[idx]

                # Scales are always stored as FP32
                layer_store['k_scale'][slot] = k_scale[idx]
                layer_store['v_scale'][slot] = v_scale[idx]
                layer_store['bits'][slot] = bits_tensor[idx]
        else:
            # No packing: store INT8 directly
            layer_store['k_qx'][start:new_size] = k_qx
            layer_store['v_qx'][start:new_size] = v_qx
            layer_store['k_scale'][start:new_size] = k_scale
            layer_store['v_scale'][start:new_size] = v_scale
            layer_store['bits'][start:new_size] = bits_tensor

        for offset, token_id in enumerate(token_ids):
            slot = start + offset
            token_int = int(token_id)

            cache_key = (layer_idx, token_int)
            self.k_cache[cache_key] = {
                'qx': layer_store['k_qx'][slot],
                'scale': layer_store['k_scale'][slot],
                'bits': int(bits_tensor[offset]),
                'dtype': torch.float32,
                'index': slot,
                'layer': layer_idx,
            }
            self.v_cache[cache_key] = {
                'qx': layer_store['v_qx'][slot],
                'scale': layer_store['v_scale'][slot],
                'bits': int(bits_tensor[offset]),
                'dtype': torch.float32,
                'index': slot,
                'layer': layer_idx,
            }

            if self.use_bucketed_layout:
                self._append_bucket_entry(
                    layer_store,
                    int(bits_tensor[offset]),
                    token_int,
                    slot,
                    k_qx[offset],
                    v_qx[offset],
                    k_scale[offset],
                    v_scale[offset],
                )

        layer_store['size'] = new_size
        self.total_tokens_stored += seq_len
        self.token_counter += seq_len
    
    def retrieve(
        self,
        layer_idx: int,
        token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve and dequantize key and value vectors.

        Args:
            layer_idx: Layer index
            token_id: Token ID

        Returns:
            Tuple of (key, value) tensors

        Raises:
            KeyError: If token not found in cache
        """
        layer_store = self.layer_store[layer_idx]
        size = layer_store['size']
        if size == 0:
            raise KeyError(f"Token {token_id} not found in cache for layer {layer_idx}")

        token_ids = layer_store['token_ids'][:size]
        query = torch.tensor([token_id], dtype=torch.long)
        pos = torch.searchsorted(token_ids, query)
        idx = pos.item()
        if idx >= size or token_ids[idx] != token_id:
            raise KeyError(f"Token {token_id} not found in cache for layer {layer_idx}")

        indices = torch.tensor([idx], dtype=torch.long)
        keys, values = self._dequantize_indices(layer_store, indices)

        k = keys[0]
        v = values[0]

        # Tensors are already on the correct device from layer_store
        return k, v
    
    def retrieve_all(
        self,
        layer_idx: int,
        token_ids: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve all cached KV for a layer with batched dequantization.

        Args:
            layer_idx: Layer index
            token_ids: Optional list of token IDs to retrieve.
                      If None, retrieves all tokens in cache for this layer.

        Returns:
            Tuple of (keys, values) tensors
            Shape: [num_tokens, num_heads, head_dim]
        """
        layer_store = self.layer_store[layer_idx]
        size = layer_store['size']

        if size == 0:
            empty = torch.empty(0, self.num_heads, self.head_dim)
            return empty, empty

        token_ids_tensor = layer_store['token_ids'][:size]
        if token_ids is None:
            indices = torch.arange(size, dtype=torch.long)
        else:
            query = torch.tensor(token_ids, dtype=torch.long)
            pos = torch.searchsorted(token_ids_tensor, query)
            valid = (pos < size) & (token_ids_tensor[pos] == query)
            if not bool(valid.any()):
                empty = torch.empty(0, self.num_heads, self.head_dim)
                return empty, empty
            indices = pos[valid]
            # preserve original query ordering
            order = torch.arange(len(token_ids))[valid]
            sorted_idx = torch.argsort(order)
            indices = indices[sorted_idx]

        keys, values = self._dequantize_indices(layer_store, indices)

        # Tensors are already on the correct device from layer_store
        return keys, values

    def retrieve_all_quantized(
        self,
        layer_idx: int,
        token_ids: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Return raw quantized buffers and metadata for fused attention."""
        layer_store = self.layer_store[layer_idx]
        size = layer_store['size']
        device = layer_store['k_qx'].device if layer_store['k_qx'] is not None else self.device

        if size == 0:
            empty_ids = torch.empty(0, dtype=torch.long, device=device)
            empty_k = torch.empty((0, self.num_heads, self.head_dim), dtype=torch.int8, device=device)
            empty_scale = torch.empty((0, self.num_heads), dtype=torch.float32, device=device)
            return {
                'token_ids': empty_ids,
                'k_qx': empty_k,
                'k_scale': empty_scale,
                'v_qx': empty_k.clone(),
                'v_scale': empty_scale.clone(),
            }

        token_ids_tensor = layer_store['token_ids'][:size]
        if token_ids is None:
            indices = torch.arange(size, dtype=torch.long)
        else:
            query = torch.tensor(token_ids, dtype=torch.long, device=device)
            pos = torch.searchsorted(token_ids_tensor, query)
            valid = (pos < size) & (token_ids_tensor[pos] == query)
            if not bool(valid.any()):
                return {
                    'token_ids': torch.empty(0, dtype=torch.long, device=device),
                    'k_qx': torch.empty((0, self.num_heads, self.head_dim), dtype=torch.int8, device=device),
                    'k_scale': torch.empty((0, self.num_heads), dtype=torch.float32, device=device),
                    'v_qx': torch.empty((0, self.num_heads, self.head_dim), dtype=torch.int8, device=device),
                    'v_scale': torch.empty((0, self.num_heads), dtype=torch.float32, device=device),
                }
            indices = pos[valid]

        result = {
            'token_ids': token_ids_tensor[indices].clone(),
            'k_qx': layer_store['k_qx'][indices].clone(),
            'k_scale': layer_store['k_scale'][indices].clone(),
            'v_qx': layer_store['v_qx'][indices].clone(),
            'v_scale': layer_store['v_scale'][indices].clone(),
        }

        if self.use_bucketed_layout:
            result['buckets'] = self.get_bucket_views(layer_idx)

        return result

    def _dequantize_indices(
        self,
        layer_store: Dict[str, Any],
        indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized dequantization helper with optional bit-unpacking."""
        if self.use_packing:
            # Unpack and dequantize each token individually
            keys_list = []
            values_list = []

            for idx in indices.tolist():
                slot = int(idx)
                bits = int(layer_store['bits'][slot])

                # Check if this token is bit-packed
                if bits < 8 and slot in layer_store['k_packed']:
                    # Unpack from compressed format
                    k_packed = layer_store['k_packed'][slot]
                    v_packed = layer_store['v_packed'][slot]
                    k_shape = layer_store['k_shapes'][slot]
                    v_shape = layer_store['v_shapes'][slot]

                    k_qx = unpack_tensor(k_packed, bits, k_shape).to(torch.float32).squeeze(0)
                    v_qx = unpack_tensor(v_packed, bits, v_shape).to(torch.float32).squeeze(0)
                else:
                    # Use INT8 storage (8-bit or fallback)
                    k_qx = layer_store['k_qx'][slot].to(torch.float32)
                    v_qx = layer_store['v_qx'][slot].to(torch.float32)

                # Dequantize with scales
                k_scale = layer_store['k_scale'][slot].unsqueeze(-1)
                v_scale = layer_store['v_scale'][slot].unsqueeze(-1)

                keys_list.append(k_qx * k_scale)
                values_list.append(v_qx * v_scale)

            keys = torch.stack(keys_list, dim=0)
            values = torch.stack(values_list, dim=0)
        else:
            # Standard dequantization (no bit-packing)
            k_qx = layer_store['k_qx'][indices].to(torch.float32)
            v_qx = layer_store['v_qx'][indices].to(torch.float32)

            k_scale = layer_store['k_scale'][indices].unsqueeze(-1)
            v_scale = layer_store['v_scale'][indices].unsqueeze(-1)

            keys = k_qx * k_scale
            values = v_qx * v_scale

        return keys, values
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict with memory statistics
        """
        # Compute memory usage from precision map
        if self.precision_map:
            allocation_stats = compute_memory_usage(
                self.precision_map,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                scale_dtype=self.scale_dtype,
                use_packing=self.use_packing
            )
        else:
            allocation_stats = {
                'total_bits': 0,
                'avg_bits': 0.0,
                'memory_ratio': 0.0,
                'memory_ratio_true': 0.0,
                'num_tokens': 0,
                'storage_mode': 'packed' if self.use_packing else 'int8',
                'scale_dtype': self.scale_dtype
            }
        
        # Count cache entries
        num_cache_entries = sum(store['size'] for store in self.layer_store)
        
        # Compute precision distribution
        precision_distribution = {}
        for bits in self.available_bits:
            count = sum(1 for b in self.precision_map.values() if b == bits)
            precision_distribution[f"{bits}-bit"] = count
        
        # Get system memory info (if available)
        try:
            process = psutil.Process(os.getpid())
            system_memory_mb = process.memory_info().rss / 1024 / 1024
        except:
            system_memory_mb = None
        
        return {
            'memory_budget': self.memory_budget,
            'memory_ratio': allocation_stats['memory_ratio'],
            'memory_ratio_true': allocation_stats.get('memory_ratio_true', allocation_stats['memory_ratio']),
            'avg_bits': allocation_stats['avg_bits'],
            'num_tokens': allocation_stats['num_tokens'],
            'num_cache_entries': num_cache_entries,
            'precision_distribution': precision_distribution,
            'total_tokens_stored': self.total_tokens_stored,
            'realloc_count': self.realloc_counter,
            'system_memory_mb': system_memory_mb,
            'storage_mode': allocation_stats.get('storage_mode'),
            'scale_dtype': allocation_stats.get('scale_dtype'),
        }
    
    def get_precision(self, token_id: int) -> int:
        """
        Get allocated precision for a token.
        
        Args:
            token_id: Token ID
        
        Returns:
            Allocated bits (defaults to 4 if not allocated)
        """
        return self.precision_map.get(token_id, 4)
    
    def get_importance(self, token_id: int) -> float:
        """
        Get importance score for a token.
        
        Args:
            token_id: Token ID
        
        Returns:
            Importance score
        """
        return self.importance_tracker.get_importance(token_id)
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.k_cache.clear()
        self.v_cache.clear()
        self.layer_store = [self._init_layer_store() for _ in range(self.num_layers)]
        self.precision_map.clear()
        self.importance_tracker.reset()
        self.token_counter = 0
        self.realloc_counter = 0
        self.total_tokens_stored = 0
        self.global_step = 0
        self.last_seen.clear()
        self.head_importance = torch.ones(self.num_heads, dtype=torch.float32)
    
    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()
    
    def get_top_k_important_tokens(self, k: int) -> List[Tuple[int, float]]:
        """
        Get top-k most important tokens.
        
        Args:
            k: Number of top tokens to return
        
        Returns:
            List of (token_id, importance) tuples
        """
        return self.importance_tracker.get_top_k_tokens(k)
    
    def get_bottom_k_important_tokens(self, k: int) -> List[Tuple[int, float]]:
        """
        Get bottom-k least important tokens.
        
        Args:
            k: Number of bottom tokens to return
        
        Returns:
            List of (token_id, importance) tuples
        """
        return self.importance_tracker.get_bottom_k_tokens(k)
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export cache state for checkpointing.
        
        Returns:
            Dict with cache state
        """
        return {
            'config': {
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'head_dim': self.head_dim,
                'memory_budget': self.memory_budget,
                'decay': self.decay,
                'realloc_freq': self.realloc_freq,
                'available_bits': self.available_bits,
            },
            'precision_map': self.precision_map.copy(),
            'token_counter': self.token_counter,
            'realloc_counter': self.realloc_counter,
            'total_tokens_stored': self.total_tokens_stored,
            # Note: We don't export the actual cached tensors as they're large
            # In practice, cache is regenerated during inference
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SmartKVCache(layers={self.num_layers}, "
            f"budget={self.memory_budget:.1%}, "
            f"tokens={len(self.precision_map)}, "
            f"cached={sum(store['size'] for store in self.layer_store)})"
        )
    
    def __len__(self) -> int:
        """Number of tokens with allocated precision."""
        return len(self.precision_map)
