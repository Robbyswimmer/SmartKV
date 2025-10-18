"""
Precision allocation algorithms for SmartKV cache.

Implements various strategies to allocate bit-widths to tokens based on
importance scores, subject to memory budget constraints.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


def greedy_allocation(
    importance_scores: Dict[int, float],
    memory_budget: float,
    available_bits: List[int] = [2, 3, 4, 8],
    fp16_baseline: bool = True
) -> Dict[int, int]:
    """
    Greedy allocation: assign highest precision to most important tokens.
    
    This is the main allocation method used in SmartKV. Iterates through tokens
    in order of importance and assigns the highest precision that fits within
    the remaining budget.
    
    Args:
        importance_scores: Dict mapping token_id -> importance score
        memory_budget: Fraction of FP16 memory to use (e.g., 0.5 = 50%)
        available_bits: List of available bit-widths, sorted high to low
        fp16_baseline: If True, budget is relative to FP16 (16 bits * 2 for K+V)
                      If False, budget is absolute number of bits
    
    Returns:
        Dict mapping token_id -> allocated bits
    """
    if not importance_scores:
        return {}
    
    num_tokens = len(importance_scores)
    
    # Calculate total budget in bits
    if fp16_baseline:
        # FP16: 16 bits per element, 2 elements (K and V) per token
        fp16_bits = num_tokens * 16 * 2
        budget_bits = fp16_bits * memory_budget
    else:
        budget_bits = memory_budget
    
    # Sort tokens by importance (descending)
    sorted_tokens = sorted(
        importance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Ensure available_bits is sorted high to low
    available_bits = sorted(available_bits, reverse=True)
    
    precision_map = {}
    bits_used = 0
    
    for token_id, importance in sorted_tokens:
        # Try to allocate from highest to lowest precision
        allocated = False
        for bits in available_bits:
            cost = bits * 2  # K and V both need storage
            if bits_used + cost <= budget_bits:
                precision_map[token_id] = bits
                bits_used += cost
                allocated = True
                break
        
        # If no precision fits, use minimum (fall back to lowest bits)
        if not allocated:
            min_bits = min(available_bits)
            precision_map[token_id] = min_bits
            bits_used += min_bits * 2
    
    return precision_map


def dynamic_programming_allocation(
    importance_scores: Dict[int, float],
    memory_budget: float,
    available_bits: List[int] = [2, 3, 4, 8],
    fp16_baseline: bool = True
) -> Dict[int, int]:
    """
    Dynamic programming allocation for optimal precision assignment.
    
    Finds optimal allocation that maximizes weighted importance subject to
    memory budget. More expensive than greedy but guarantees optimality.
    Used for ablation studies.
    
    Objective: max Σ(importance[i] * bits[i])
    Subject to: Σ(bits[i] * 2) <= budget
    
    Args:
        importance_scores: Dict mapping token_id -> importance score
        memory_budget: Fraction of FP16 memory to use
        available_bits: List of available bit-widths
        fp16_baseline: If True, budget is relative to FP16
    
    Returns:
        Dict mapping token_id -> allocated bits
    """
    if not importance_scores:
        return {}
    
    num_tokens = len(importance_scores)
    
    # Calculate budget
    if fp16_baseline:
        fp16_bits = num_tokens * 16 * 2
        budget_bits = int(fp16_bits * memory_budget)
    else:
        budget_bits = int(memory_budget)
    
    # Convert to list for indexing
    tokens = list(importance_scores.keys())
    importances = [importance_scores[tid] for tid in tokens]
    available_bits = sorted(available_bits)
    
    n = len(tokens)
    
    # dp[i][b] = (max_value, allocation_dict)
    # i = number of tokens considered, b = bits used
    dp = {}
    
    def solve(idx: int, remaining_bits: int, memo: Dict) -> Tuple[float, Dict[int, int]]:
        """Recursive DP with memoization."""
        if idx == n or remaining_bits <= 0:
            return 0.0, {}
        
        if (idx, remaining_bits) in memo:
            return memo[(idx, remaining_bits)]
        
        token_id = tokens[idx]
        importance = importances[idx]
        
        best_value = 0.0
        best_allocation = {}
        
        # Try each bit-width
        for bits in available_bits:
            cost = bits * 2  # K and V
            if cost <= remaining_bits:
                # Value of allocating this bit-width
                value = importance * bits
                
                # Recursively solve for remaining tokens
                future_value, future_allocation = solve(idx + 1, remaining_bits - cost, memo)
                
                total_value = value + future_value
                
                if total_value > best_value:
                    best_value = total_value
                    best_allocation = {token_id: bits, **future_allocation}
        
        memo[(idx, remaining_bits)] = (best_value, best_allocation)
        return best_value, best_allocation
    
    _, allocation = solve(0, budget_bits, {})
    
    # Fill in any missing tokens with minimum bits
    min_bits = min(available_bits)
    for token_id in tokens:
        if token_id not in allocation:
            allocation[token_id] = min_bits
    
    return allocation


def layer_aware_allocation(
    layer_importance: Dict[Tuple[int, int], float],
    memory_budget: float,
    num_layers: int,
    available_bits: List[int] = [2, 3, 4, 8],
    layer_weight_strategy: str = "linear"
) -> Dict[int, int]:
    """
    Layer-aware allocation: different budgets for different layers.
    
    Hypothesis: Later layers need more precision than early layers.
    Allocates varying budgets to layers and uses greedy allocation within each.
    
    Args:
        layer_importance: Dict mapping (layer_idx, token_id) -> importance
        memory_budget: Overall fraction of FP16 memory
        num_layers: Total number of layers
        available_bits: Available bit-widths
        layer_weight_strategy: How to weight layers
            - "linear": Later layers get linearly more budget
            - "equal": All layers get equal budget
            - "exponential": Later layers get exponentially more budget
    
    Returns:
        Dict mapping token_id -> allocated bits (averaged across layers)
    """
    if not layer_importance:
        return {}
    
    # Compute per-layer budgets
    layer_budgets = _compute_layer_budgets(
        num_layers, 
        memory_budget, 
        layer_weight_strategy
    )
    
    # Group importance by layer
    layer_tokens: Dict[int, Dict[int, float]] = defaultdict(dict)
    for (layer_idx, token_id), score in layer_importance.items():
        layer_tokens[layer_idx][token_id] = score
    
    # Allocate within each layer
    layer_allocations: Dict[int, Dict[int, int]] = {}
    for layer_idx in range(num_layers):
        if layer_idx in layer_tokens:
            layer_allocations[layer_idx] = greedy_allocation(
                layer_tokens[layer_idx],
                layer_budgets[layer_idx],
                available_bits,
                fp16_baseline=True
            )
    
    # Aggregate across layers (average or max)
    token_bits: Dict[int, List[int]] = defaultdict(list)
    for layer_idx, allocation in layer_allocations.items():
        for token_id, bits in allocation.items():
            token_bits[token_id].append(bits)
    
    # Use average bits across layers (can also use max)
    final_allocation = {}
    for token_id, bits_list in token_bits.items():
        avg_bits = np.mean(bits_list)
        # Round to nearest available bit-width
        final_allocation[token_id] = _round_to_nearest_bits(avg_bits, available_bits)
    
    return final_allocation


def uniform_allocation(
    token_ids: List[int],
    bits: int
) -> Dict[int, int]:
    """
    Uniform allocation: all tokens get same precision.
    
    Used as baseline for comparison.
    
    Args:
        token_ids: List of token IDs
        bits: Bit-width to assign to all tokens
    
    Returns:
        Dict mapping token_id -> bits
    """
    return {tid: bits for tid in token_ids}


def random_allocation(
    token_ids: List[int],
    memory_budget: float,
    available_bits: List[int] = [2, 3, 4, 8],
    seed: Optional[int] = None
) -> Dict[int, int]:
    """
    Random allocation: assign random precision to tokens.
    
    Used as ablation baseline to verify importance-guided allocation helps.
    
    Args:
        token_ids: List of token IDs
        memory_budget: Memory budget fraction
        available_bits: Available bit-widths
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping token_id -> bits
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_tokens = len(token_ids)
    fp16_bits = num_tokens * 16 * 2
    budget_bits = fp16_bits * memory_budget
    
    allocation = {}
    bits_used = 0
    
    for token_id in token_ids:
        # Randomly choose a bit-width that fits
        possible_bits = [b for b in available_bits if bits_used + b * 2 <= budget_bits]
        
        if possible_bits:
            bits = np.random.choice(possible_bits)
        else:
            bits = min(available_bits)
        
        allocation[token_id] = bits
        bits_used += bits * 2
    
    return allocation


def position_based_allocation(
    token_ids: List[int],
    memory_budget: float,
    available_bits: List[int] = [2, 3, 4, 8],
    position_importance: str = "early"
) -> Dict[int, int]:
    """
    Position-based allocation: allocate based on token position.
    
    Used for ablation to compare against attention-based allocation.
    
    Args:
        token_ids: List of token IDs (assumed in sequence order)
        memory_budget: Memory budget fraction
        available_bits: Available bit-widths
        position_importance: "early" (early tokens more important) or 
                           "late" (late tokens more important)
    
    Returns:
        Dict mapping token_id -> bits
    """
    # Create synthetic importance based on position
    importance_scores = {}
    num_tokens = len(token_ids)
    
    for idx, token_id in enumerate(token_ids):
        if position_importance == "early":
            # Earlier positions have higher importance
            importance_scores[token_id] = float(num_tokens - idx)
        elif position_importance == "late":
            # Later positions have higher importance
            importance_scores[token_id] = float(idx + 1)
        else:
            raise ValueError(f"Unknown position_importance: {position_importance}")
    
    # Use greedy allocation with synthetic importance
    return greedy_allocation(importance_scores, memory_budget, available_bits)


def _compute_layer_budgets(
    num_layers: int,
    total_budget: float,
    strategy: str
) -> Dict[int, float]:
    """
    Compute per-layer memory budgets.
    
    Args:
        num_layers: Number of layers
        total_budget: Total memory budget
        strategy: Budget allocation strategy
    
    Returns:
        Dict mapping layer_idx -> budget fraction
    """
    if strategy == "equal":
        # Equal budget for all layers
        per_layer_budget = total_budget
        return {i: per_layer_budget for i in range(num_layers)}
    
    elif strategy == "linear":
        # Linear increase: layer 0 gets weight 1, layer N-1 gets weight N
        weights = np.arange(1, num_layers + 1)
        weights = weights / weights.sum()
        
        # Scale weights so average equals total_budget
        budgets = weights * total_budget * num_layers
        return {i: float(budgets[i]) for i in range(num_layers)}
    
    elif strategy == "exponential":
        # Exponential increase: later layers get much more budget
        weights = np.exp(np.linspace(0, 2, num_layers))
        weights = weights / weights.sum()
        
        budgets = weights * total_budget * num_layers
        return {i: float(budgets[i]) for i in range(num_layers)}
    
    else:
        raise ValueError(f"Unknown layer weight strategy: {strategy}")


def _round_to_nearest_bits(value: float, available_bits: List[int]) -> int:
    """
    Round a float value to the nearest available bit-width.
    
    Args:
        value: Float value to round
        available_bits: List of available bit-widths
    
    Returns:
        Nearest bit-width
    """
    return min(available_bits, key=lambda b: abs(b - value))


def compute_minimum_budget(
    num_heads: int = 8,
    head_dim: int = 128,
    scale_dtype: str = "fp32",
    use_packing: bool = False
) -> float:
    """
    Compute minimum achievable budget given storage constraints.

    Args:
        num_heads: Number of attention heads
        head_dim: Head dimension
        scale_dtype: Scale factor dtype ("fp32" or "fp16")
        use_packing: Whether bit-packing is implemented

    Returns:
        Minimum budget ratio (cannot go below this)

    Examples:
        INT8 + FP32 scales: 0.5313
        INT8 + FP16 scales: 0.5156
        4-bit packed + FP32 scales: 0.2656
        4-bit packed + FP16 scales: 0.2578
    """
    scale_bits = 32 if scale_dtype == "fp32" else 16

    if use_packing:
        # Minimum with 2-bit packing
        payload_bits = 2 * num_heads * head_dim * 2  # K and V at 2-bit
    else:
        # INT8 storage regardless of target bits
        payload_bits = 2 * num_heads * head_dim * 8  # K and V at INT8

    # Scale metadata: one scale per head for K and V, plus stored inverse for fast dequant
    scale_bits_total = 4 * num_heads * scale_bits

    # FP16 baseline: 2 * num_heads * head_dim * 16
    fp16_bits = 2 * num_heads * head_dim * 16

    min_ratio = (payload_bits + scale_bits_total) / fp16_bits
    return min_ratio


def compute_memory_usage(
    allocation: Dict[int, int],
    fp16_baseline: bool = True,
    num_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    scale_dtype: str = "fp32",
    use_packing: bool = False
) -> Dict[str, float]:
    """
    Compute memory usage statistics for an allocation.

    NOTE: Current implementation stores all quantized values as INT8,
    regardless of target bit-width (2/3/4/8). This means actual memory
    usage is 8 bits per element, not the target bit-width, unless
    bit-packing is implemented.

    Args:
        allocation: Dict mapping token_id -> bits
        fp16_baseline: If True, compute relative to FP16
        num_heads: Number of attention heads (for scale metadata)
        head_dim: Head dimension (default 128)
        scale_dtype: Scale factor dtype ("fp32" or "fp16")
        use_packing: Whether bit-packing is implemented

    Returns:
        Dict with memory statistics including scale overhead
    """
    if not allocation:
        return {
            'total_bits': 0,
            'total_bits_with_scales': 0,
            'avg_bits': 0.0,
            'memory_ratio': 0.0,
            'memory_ratio_true': 0.0,
            'num_tokens': 0,
            'scale_overhead_pct': 0.0,
            'storage_mode': 'packed' if use_packing else 'int8',
            'scale_dtype': scale_dtype
        }

    num_tokens = len(allocation)
    scale_bits = 32 if scale_dtype == "fp32" else 16

    if num_heads is None or head_dim is None:
        # Normalized view for analytical tests (assume one value per token)
        theoretical_payload = sum(bits * 2 for bits in allocation.values())
        actual_payload = theoretical_payload if use_packing else num_tokens * 2 * 8
        scale_bits_theoretical = 0
        scale_bits_actual = 0
        fp16_bits = num_tokens * 32 if fp16_baseline else num_tokens
    else:
        elements = 2 * num_heads * head_dim
        total_bits_payload = sum(bits * elements for bits in allocation.values())
        theoretical_payload = total_bits_payload
        actual_payload = total_bits_payload if use_packing else num_tokens * elements * 8
        scale_bits_theoretical = num_tokens * 4 * num_heads * scale_bits
        scale_bits_actual = num_tokens * 2 * num_heads * scale_bits
        fp16_bits = num_tokens * elements * 16 if fp16_baseline else num_tokens

    total_bits_theoretical = theoretical_payload + scale_bits_theoretical
    total_bits_actual = actual_payload + scale_bits_actual
    total_bits_with_scales = total_bits_actual

    avg_bits = sum(allocation.values()) / len(allocation) if allocation else 0.0

    if fp16_baseline:
        memory_ratio_theoretical = total_bits_theoretical / fp16_bits if fp16_bits > 0 else 0.0
        memory_ratio_actual = total_bits_actual / fp16_bits if fp16_bits > 0 else 0.0
    else:
        memory_ratio_theoretical = 1.0
        memory_ratio_actual = 1.0

    scale_overhead_pct = (scale_bits_actual / total_bits_actual * 100) if total_bits_actual > 0 else 0.0

    return {
        'total_bits': total_bits_theoretical,  # Theoretical (with packing)
        'total_bits_actual': total_bits_actual,  # Actual (storage + scales)
        'total_bits_with_scales': total_bits_with_scales,
        'payload_bits': actual_payload,  # Just K/V storage (actual)
        'scale_bits': scale_bits_actual,  # Just scales (actual)
        'avg_bits': avg_bits,  # Average assigned bits per token
        'memory_ratio': memory_ratio_theoretical,  # Theoretical budget usage
        'memory_ratio_true': memory_ratio_actual,  # True memory usage
        'num_tokens': num_tokens,
        'scale_overhead_pct': scale_overhead_pct,
        'storage_mode': 'packed' if use_packing else 'int8',
        'scale_dtype': scale_dtype
    }


def validate_allocation(
    allocation: Dict[int, int],
    memory_budget: float,
    available_bits: List[int],
    fp16_baseline: bool = True
) -> Tuple[bool, str]:
    """
    Validate that an allocation meets constraints.
    
    Args:
        allocation: Dict mapping token_id -> bits
        memory_budget: Memory budget constraint
        available_bits: List of valid bit-widths
        fp16_baseline: If True, budget is relative to FP16
    
    Returns:
        (is_valid, error_message)
    """
    if not allocation:
        return True, ""
    
    # Check all bit-widths are valid
    for token_id, bits in allocation.items():
        if bits not in available_bits:
            return False, f"Token {token_id} has invalid bits {bits}, must be in {available_bits}"
    
    # Check memory budget
    stats = compute_memory_usage(allocation, fp16_baseline)
    
    if fp16_baseline:
        if stats['memory_ratio'] > memory_budget + 0.01:  # Small tolerance
            return False, f"Memory usage {stats['memory_ratio']:.2%} exceeds budget {memory_budget:.2%}"
    
    return True, ""
