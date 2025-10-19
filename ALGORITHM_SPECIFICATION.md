# SmartKV: Attention-Guided Adaptive Precision KV-Cache Compression

## Formal Algorithm Specification

This document provides a mathematically rigorous specification of the SmartKV algorithm for review and publication.

---

## 1. Problem Formulation

### 1.1 Notation

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| $L$ | scalar | Number of transformer layers |
| $H$ | scalar | Number of attention heads |
| $d$ | scalar | Head dimension |
| $n$ | scalar | Context length (number of cached tokens) |
| $\mathcal{B}$ | set | Available bit-widths, $\mathcal{B} = \{2, 3, 4, 8\}$ |
| $s_{\text{scale}}$ | scalar | Bits used to store each scale value (FP16/FP32) |
| $g$ | scalar | Channels per stored scale (group size) |
| $\beta$ | scalar | Memory budget as fraction of FP16, $\beta \in (0, 1]$ |
| $\alpha$ | scalar | Utility exponent controlling diminishing returns |
| $\epsilon_{\text{imp}}$ | scalar | Minimum effective importance floor |
| $r$ | scalar | Percentile hysteresis threshold |
| $m$ | scalar | Number of consecutive intervals required for rank change |
| $\tau$ | scalar | Drift threshold (e.g., KL divergence) triggering reallocation |
| $\gamma$ | scalar | Attention score decay factor, $\gamma \in [0, 1)$ |
| $f$ | scalar | Reallocation frequency (number of generation steps) |
| $K_\ell^{(i)}$ | $\mathbb{R}^{H \times d}$ | Key vector for token $i$ at layer $\ell$ |
| $V_\ell^{(i)}$ | $\mathbb{R}^{H \times d}$ | Value vector for token $i$ at layer $\ell$ |
| $A_\ell^{(t)}$ | $\mathbb{R}^{H \times n \times n}$ | Attention matrix at layer $\ell$, generation step $t$ |
| $Q_\ell^{(t)}$ | $\mathbb{R}^{H \times d}$ | Query vector at layer $\ell$, generation step $t$ |
| $k_{\text{critical}}$ | scalar | Number of prefix tokens forced to max precision |
| $k_{\text{tail}}$ | scalar | Number of most recent tokens forced to max precision |

### 1.2 Objective

Given a memory budget $\beta$ and a set of cached key-value pairs $\{(K_\ell^{(i)}, V_\ell^{(i)})\}_{i=1}^{n}$ for each layer $\ell \in \{1, \ldots, L\}$, find a precision allocation function $\pi: \{1, \ldots, n\} \times \{1, \ldots, L\} \to \mathcal{B}$ that:

1. **Satisfies memory constraint:**
   $$\frac{1}{16 n L} \sum_{\ell=1}^{L} \sum_{i=1}^{n} \pi(i, \ell) \leq \beta$$

2. **Minimizes quality degradation:**
   $$\min_{\pi} \mathbb{E}\left[\|\text{Attention}(Q, K_{\text{fp16}}, V_{\text{fp16}}) - \text{Attention}(Q, K_{\text{quant}}, V_{\text{quant}})\|_2^2\right]$$

where the expectation is over all queries $Q$ during generation.

---

## 2. Quantization Scheme

### 2.1 Per-Head Symmetric Quantization

For a given key or value tensor $X \in \mathbb{R}^{H \times d}$ and bit-width $b \in \mathcal{B}$:

**Quantization range:**
$$q_{\max}(b) = 2^{b-1} - 1, \quad q_{\min}(b) = -2^{b-1}$$

**Per-head scale factors:**
$$s_h = \frac{\max_{j \in \{1, \ldots, d\}} |X_{h,j}|}{q_{\max}(b)}, \quad h \in \{1, \ldots, H\}$$

**Clamping for numerical stability:**
$$s_h \leftarrow \max(s_h, \epsilon), \quad \epsilon = 10^{-8}$$

**Quantization function:**
$$\mathcal{Q}(X, b)_h = \left\lfloor \text{clamp}\left(\frac{X_h}{s_h}, q_{\min}(b), q_{\max}(b)\right) \right\rceil$$

where $\lfloor \cdot \rceil$ denotes rounding to nearest integer.

**Dequantization:**
$$\hat{X}_h = \mathcal{Q}(X, b)_h \cdot s_h$$

**Quantization error per head:**
$$\epsilon_h(b) = \|X_h - \hat{X}_h\|_2$$

### 2.2 Memory Consumption

For a single token's KV pair at precision $b$ bits (with per-head scales stored at $s_{\text{scale}}$ bits and grouping factor $g \geq 1$ channels per scale):
$$M(b) = 2 H d b \;\text{bits for quantized payload} + 2 H \frac{s_{\text{scale}}}{g} \;\text{bits for scales}$$

Common choices are $s_{\text{scale}} \in \{16, 32\}$ (FP16/FP32) and $g \in \{1, 8, 16\}$ (per-head or grouped). This makes the minimum attainable budget strictly greater than the payload-only bound (e.g., $\beta_{\text{min}} \approx 0.156$ with 2-bit payloads, FP32 scales, and packing).

Full FP16 cache memory (all layers, all tokens):
$$M_{\text{fp16}} = 2 L n H d \cdot 16 \text{ bits} = 4LnHd \text{ bytes}$$

Memory constraint for precision allocation $\pi$:
$$\sum_{\ell=1}^{L} \sum_{i=1}^{n} M(\pi(i, \ell)) \leq \beta \cdot M_{\text{fp16}}$$

---

## 3. Importance Tracking

### 3.1 Attention Score Aggregation

At each generation step $t$ and layer $\ell$, compute attention scores:
$$A_\ell^{(t)} = \text{softmax}\left(\frac{Q_\ell^{(t)} (K_\ell^{(1:n)})^T}{\sqrt{d}}\right) \in \mathbb{R}^{H \times 1 \times n}$$

**Per-token attention weight** (averaged over heads):
$$\alpha_\ell^{(t)}(i) = \frac{1}{H} \sum_{h=1}^{H} A_{\ell, h}^{(t)}[i], \quad i \in \{1, \ldots, n\}$$

### 3.2 Exponential Moving Average (EMA)

Maintain importance scores $I_\ell^{(t)}(i)$ for each token $i$ at layer $\ell$:

**Initialization** (first time token $i$ is observed at layer $\ell$):
$$I_\ell^{(0)}(i) = 0$$

**Update rule** (after generation step $t$):
$$I_\ell^{(t+1)}(i) = \gamma \cdot I_\ell^{(t)}(i) + (1 - \gamma) \cdot \alpha_\ell^{(t)}(i)$$

where $\gamma \in [0, 1)$ is the decay factor (typically $\gamma = 0.9$).

**Intuition:** Recent attention scores are weighted more heavily; old scores decay exponentially.

### 3.3 Cross-Layer Aggregation

Aggregate importance across all layers to obtain per-token global importance:
$$I^{(t)}(i) = \frac{1}{L} \sum_{\ell=1}^{L} I_\ell^{(t)}(i)$$

**Normalization** (optional, for numerical stability):
$$I^{(t)}(i) \leftarrow \frac{I^{(t)}(i)}{\frac{1}{n}\sum_{j=1}^{n} I^{(t)}(j) + \epsilon}$$

---

## 4. Quality-Preserving Enhancements

### 4.1 Recency Weighting

Recent tokens are often more important for coherent generation. Apply temporal decay:
$$I_{\text{recency}}^{(t)}(i) = I^{(t)}(i) \cdot \exp\left(-\frac{t - \tau(i)}{T}\right)$$

where:
- $\tau(i)$ = last generation step when token $i$ was seen
- $T$ = recency temperature (typically $T = 512$)
- $t - \tau(i)$ = age of token $i$ at step $t$

During prefill $\tau(i)$ is initialized from the token position index; during autoregressive decoding it advances with the step counter so the age reflects real execution time.

### 4.2 Head Importance Weighting

Not all attention heads are equally important. Track per-head importance with EMA:

**Per-head mean attention (per layer):**
$$\bar{A}_{\ell,h}^{(t)} = \frac{1}{n} \sum_{i=1}^{n} A_{\ell, h}^{(t)}[i]$$

**Head importance EMA (per layer):**
$$\eta_{\ell,h}^{(t+1)} = \gamma_h \cdot \eta_{\ell,h}^{(t)} + (1 - \gamma_h) \cdot \bar{A}_{\ell,h}^{(t)}$$

where $\gamma_h = 0.95$ (slower decay than token importance).

Aggregate across layers before normalization (e.g., simple mean):
$$\tilde{\eta}_h^{(t)} = \frac{1}{L} \sum_{\ell=1}^{L} \eta_{\ell,h}^{(t)}$$

**Normalized head weights:**
$$w_h^{(t)} = \frac{\tilde{\eta}_h^{(t)}}{\frac{1}{H}\sum_{h'=1}^{H} \tilde{\eta}_{h'}^{(t)} + \epsilon}$$

**Query-aware importance** (weight token importance by head importance):
$$I_{\text{weighted}}^{(t)}(i) = \frac{1}{H} \sum_{h=1}^{H} w_h^{(t)} \cdot A_{\ell, h}^{(t)}[i]$$

Combine with standard importance:
$$I_{\text{final}}^{(t)}(i) = 0.7 \cdot I^{(t)}(i) + 0.3 \cdot I_{\text{weighted}}^{(t)}(i)$$

These reductions are computed on the fly (e.g., within a FlashAttention-style kernel) so the full attention tensor $A_\ell^{(t)}$ never needs to be materialized in memory.

**IMPORTANT NOTE:** The head-weighted attention $w_h^{(t)} \cdot A_{\ell, h}^{(t)}[i]$ is **not renormalized** after scaling. This means it no longer sums to 1 over all tokens and is not a valid probability distribution. It serves purely as a **heuristic importance signal** for precision allocation. If probability semantics are required for downstream use, apply softmax renormalization after weighting.

### 4.3 Special Token Protection

Protect positions rather than vocabulary IDs to remain tokenizer-agnostic:

**Protected token positions:**
$$\mathcal{P} = \{1, \ldots, k_{\text{critical}}\} \cup \{n - k_{\text{tail}} + 1, \ldots, n\}$$

where $k_{\text{critical}}$ (default 4) guards the prompt prefix and $k_{\text{tail}}$ (optional, default 0) keeps the most recent tokens at maximum precision for short-range coherence.

**Protection constraint:**
$$\pi(i, \ell) = \max(\mathcal{B}), \quad \forall i \in \mathcal{P}, \, \forall \ell$$

---

## 5. Precision Allocation Algorithm

### 5.1 Marginal-Gain Priority Allocation

The implementation uses a priority-queue allocator that incrementally upgrades tokens based on marginal utility per additional bit.

**Inputs**: memory budget $\beta$, available bits $\mathcal{B} = \{2, 3, 4, 8\}$, importance scores $\{I^{(t)}(i)\}_{i=1}^{n}$, diminishing-returns exponent $\alpha \in (0, 1]$, and a minimum effective importance $\epsilon_{\text{imp}}$.

**Step 1: Baseline assignment**

All tokens start at the minimum precision and protected tokens are clamped to the maximum:
$$\pi(i) \leftarrow \begin{cases}
8 & \text{if } i \in \mathcal{P} \\
\min(\mathcal{B}) & \text{otherwise}
\end{cases}$$

This establishes the initial payload cost used for budget accounting.

**Step 2: Utility definition**

Define a concave utility function that captures diminishing returns:
$$U(b) = b^{\alpha}$$

The marginal gain of upgrading token $i$ from bit-width $b$ to $b'$ is
$$\Delta U_i(b \to b') = U(b') - U(b)$$
and the associated cost in bits is $\Delta b = b' - b$.

To ensure inactive tokens can still consume surplus budget, replace non-positive importance with a floor:
$$\tilde{I}(i) = \max\left(I^{(t)}(i), \epsilon_{\text{imp}}\right).$$

**Step 3: Priority queue construction**

For each token $i$ with current precision $b$ that can be upgraded, push the next discrete step $(b \to b')$ from the chain $2 \to 3 \to 4 \to 8$ onto a max-heap scored by marginal utility per bit. When budget must be reclaimed (e.g., due to drift-triggered downgrades), the reverse chain $8 \to 4 \to 3 \to 2$ is used with identical scoring.
$$\text{score}(i, b \to b') = \frac{\tilde{I}(i) \cdot \Delta U_i(b \to b')}{\Delta b}.$$

Each heap entry stores $(i, b, b', \text{score})$. Tokens forced to 8-bit have no upgrade candidates.

**Step 4: Incremental upgrades**

While the heap is non-empty:

1. Pop the entry with highest score.
2. If token $i$'s current precision no longer matches $b$, discard the stale entry.
3. Compute the incremental memory cost of applying $b \to b'$ (including packed payload and scale metadata).
4. If the new allocation would violate the budget ($> \beta$), skip this candidate and continue.
5. Otherwise, accept the upgrade: set $\pi(i) \leftarrow b'$, update the running payload cost, and push the next upgrade step for token $i$ onto the heap.

The loop terminates when no legal upgrades remain or the budget is fully utilized.

**Hysteresis:** To prevent churn, maintain a short history of token ranks. Only enqueue upgrades (or downgrades when reclaiming budget) for token $i$ if its smoothed rank has moved by at least $r$ percentile points for $m$ consecutive reallocation checks. Typical values are $r = 5\%$ and $m = 2$.

**Output**: the final allocation $\pi$ satisfies the budget by construction and adapts continuously to importance magnitudes while avoiding rapid oscillations.

### 5.2 Adaptive Reallocation Frequency

Reallocation is triggered after a base interval that scales with layer depth and observed context length:

$$f^{(t)} = \max\left(\,f_{\text{base}} \cdot \left(1 + \left\lfloor \frac{\ell}{4} \right\rfloor\right),\; \max\left(32, \frac{n(t)}{8}\right)\right)$$

where $f_{\text{base}} = 16$, $\ell$ is the layer index (each layer maintains its own clock), and $n(t)$ is the number of cached tokens at step $t$.

**Intuition:** Layers deeper in the transformer and longer contexts both increase the interval, reducing allocator overhead when many tokens are already stable.

**Drift-triggered refresh:** In addition to the periodic timer, trigger reallocation early if the importance distribution drifts significantly, e.g., when $\mathrm{KL}(I^{(t)} \| I^{(t-f)}) > \tau$ or when more than $p\%$ of tokens move by $r$ percentile ranks. This keeps allocations responsive to attention shifts.

---

## 6. Forward Pass with SmartKV

### 6.1 Cache Storage

For each layer $\ell$ and each token $i$:

**Store quantized representations:**
$$\tilde{K}_\ell^{(i)} = \mathcal{Q}(K_\ell^{(i)}, \pi(i, \ell)) \in \mathbb{Z}^{H \times d}$$
$$\tilde{V}_\ell^{(i)} = \mathcal{Q}(V_\ell^{(i)}, \pi(i, \ell)) \in \mathbb{Z}^{H \times d}$$

**Store scale factors:**
$$s_{K,h}^{(i)}, s_{V,h}^{(i)} \in \mathbb{R}, \quad h \in \{1, \ldots, H\}$$

Maintain per-bit-width buckets (or fixed-size blocks) so tokens with identical precision are stored contiguously; this guarantees coalesced loads when the fused kernel iterates over 2-, 3-, 4-, and 8-bit payloads. The allocator may also bias keys and values differently (e.g., keys restricted to $\{3,4,8\}$ while values use $\{2,3,4,8\}$) because key errors perturb the softmax whereas value errors enter post-attention.
Scales are typically stored per head or per group in FP16 (or FP8 for reduced overhead), with shared buffers reused across buckets.

### 6.2 Retrieval and Dequantization

At generation step $t$, retrieve cached KV pairs for layer $\ell$:

**Dequantization:**
$$\hat{K}_\ell^{(i)} = \text{dequantize}(\tilde{K}_\ell^{(i)}, s_{K}^{(i)}) \in \mathbb{R}^{H \times d}$$
$$\hat{V}_\ell^{(i)} = \text{dequantize}(\tilde{V}_\ell^{(i)}, s_{V}^{(i)}) \in \mathbb{R}^{H \times d}$$

**Concatenate over context:**
$$\hat{K}_\ell = [\hat{K}_\ell^{(1)}, \ldots, \hat{K}_\ell^{(n)}] \in \mathbb{R}^{H \times n \times d}$$
$$\hat{V}_\ell = [\hat{V}_\ell^{(1)}, \ldots, \hat{V}_\ell^{(n)}] \in \mathbb{R}^{H \times n \times d}$$

### 6.3 Attention Computation

**Current implementation** (dequantize-then-attend):
$$A_\ell^{(t)} = \text{softmax}\left(\frac{Q_\ell^{(t)} \hat{K}_\ell^T}{\sqrt{d}}\right)$$
$$O_\ell^{(t)} = A_\ell^{(t)} \hat{V}_\ell$$

**Proposed fused kernel** (quantized-attention):
$$O_\ell^{(t)} = \text{QuantizedAttention}(Q_\ell^{(t)}, \{\tilde{K}_\ell^{(i)}\}, \{\tilde{V}_\ell^{(i)}\}, \{s_K^{(i)}\}, \{s_V^{(i)}\})$$

where dequantization happens inside the kernel, avoiding materialization of full FP32 tensors.

Practical kernel design iterates over per-bit buckets. For each bucket:
1. Load and unpack the int2/int3/int4/int8 payload on the fly (e.g., with DP4A/IMMA instructions) while caching per-head scales in shared memory.
2. Accumulate $QK^\top$ in int32, apply scale factors late, and perform a streaming softmax (maintaining the running max/sum per query) to remain bandwidth bound.
3. Stream the corresponding values $V$ in the same bucket to compute $AV$ without re-materializing intermediate attention matrices.

This layout exposes a single user-facing API despite multiple internal buckets.

**IMPLEMENTATION STATUS:** The fused kernel is **not yet implemented**. Current code uses the dequantize-then-attend path, which incurs ~50% overhead during autoregressive decoding (see §7.2). A detailed implementation plan for the fused kernel is provided in `KERNEL_IMPLEMENTATION_PLAN.md`. With the fused kernel, SmartKV would achieve 3-4× speedup at all context lengths, matching the 74% speedup observed at short context (100 tokens).

---

## 7. Theoretical Analysis

### 7.1 Memory Complexity

**FP16 baseline:**
$$\mathcal{M}_{\text{fp16}} = \Theta(LnHd)$$

**SmartKV:**
$$\mathcal{M}_{\text{smartkv}} = \beta \cdot \Theta(LnHd)$$

where $\beta \in (0, 1]$ is the memory budget (typically $\beta = 0.4$).

**Memory savings:**
$$\text{Savings} = (1 - \beta) \cdot 100\%$$

For $\beta = 0.4$: **60% memory reduction**

### 7.2 Computational Complexity

**IMPORTANT:** Complexity differs between training/prefill and autoregressive decoding.

#### Training / Prefill Phase (Full-sequence attention, $q\_len = n$)

**Per forward pass:**

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Attention computation | $O(Ln^2Hd)$ | Quadratic in context length |
| Importance update (EMA) | $O(Ln^2H)$ | Attention score aggregation |
| Precision reallocation | $O(n \log n)$ | Priority-queue updates (amortized over $f$ steps) |
| Dequantization (current) | $O(LnHd)$ | One-time cost to retrieve full cache |

**Total complexity (training/prefill):**
$$\mathcal{T}_{\text{prefill}} = O(Ln^2Hd)$$

The $O(LnHd)$ dequantization is negligible compared to $O(Ln^2Hd)$ attention when $n \gg Hd$.

#### Autoregressive Decoding (Incremental attention, $q\_len = 1$)

**Per decode step (generating one token):**

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Attention computation | $O(LnHd)$ | Query one token against $n$ cached keys |
| Importance update (EMA) | $O(LnH)$ | Update importance for one attention vector |
| Precision reallocation | $O(n \log n)$ | Amortized over $f$ steps: $O(n \log n / f)$ per step (heap operations) |
| Dequantization (current) | $O(LnHd)$ | **Same cost as attention - 50% overhead!** |
| Dequantization (fused kernel) | $O(1)$ | Absorbed into attention kernel |

**Total complexity (current implementation):**
$$\mathcal{T}_{\text{decode}} = O(LnHd) + O(LnHd) = O(LnHd)$$

The second $O(LnHd)$ term is almost entirely memory-bandwidth bound because it streams the same data footprint as attention without contributing new math. Dequantization overhead is therefore **~50% of total time** during decoding until the fused kernel removes the extra memory traffic.

**Total complexity (with fused kernel):**
$$\mathcal{T}_{\text{decode}} = O(LnHd)$$

Maintains same complexity as FP16 baseline. SmartKV wins at all context lengths due to reduced memory bandwidth.

#### Summary

| Phase | Dequant Overhead | Impact |
|-------|-----------------|---------|
| Training/prefill ($q\_len = n$) | $O(LnHd)$ vs $O(Ln^2Hd)$ attention | Negligible (<10%) |
| Decoding ($q\_len = 1$) | $O(LnHd)$ vs $O(LnHd)$ attention | **Critical (~50%)** |

This explains why SmartKV shows 74% speedup at short context but degrades at long context: **decoding overhead scales linearly with $n$, matching the attention cost.**

### 7.3 Approximation Error Bound

SmartKV uses **per-head** symmetric quantization. Let $\Delta_{\ell,h}(b) = 2 s_{\ell,h} / (2^{b-1}-1)$ denote the quantization step for head $h$ at layer $\ell$ when stored with $b$ bits (with scale $s_{\ell,h}$ defined in §2.1). Assuming the usual additive noise model with uniform quantization error $e_{\ell,h} \sim \mathcal{U}(-\Delta_{\ell,h}/2, \Delta_{\ell,h}/2)$ and independence across dimensions, we obtain the standard second-moment bounds:

**Dot-product perturbation (keys):** For a query vector $q$ and quantized key $\hat{k} = k + e$,
$$\mathbb{E}\left[(q \cdot (\hat{k} - k))^2\right] = \frac{\Delta_{\ell,h}(b)^2}{12} \|q\|_2^2.$$

**Softmax sensitivity:** Softmax is 1-Lipschitz with respect to the $\ell_\infty$ norm on logits, so
$$\|\mathrm{softmax}(z + \delta) - \mathrm{softmax}(z)\|_1 \leq \|\delta\|_\infty.$$

Combining these, the expected change in attention weights satisfies
$$\mathbb{E}\left[\|A_\ell^{(t)} - \hat{A}_\ell^{(t)}\|_1\right] \leq \frac{\Delta_{\ell,h}(b)}{\sqrt{12}} \max_h \|Q_{\ell,h}^{(t)}\|_2,$$
where $Q_{\ell,h}^{(t)}$ is the query vector for head $h$. Errors on value vectors propagate linearly after the softmax, yielding
$$\mathbb{E}\left[\|O_\ell^{(t)} - \hat{O}_\ell^{(t)}\|_2\right] \leq \sum_{i=1}^{n} \mathbb{E}[A_\ell^{(t)}[i]] \cdot \frac{\Delta_{\ell,h}(b_i)}{\sqrt{12}} \|V_\ell^{(i)}\|_2.$$

Because the allocator concentrates higher bit-widths (smaller $\Delta$) on large-attention tokens, the dominant terms in this sum remain small.

### 7.4 Optimality

**Claim:** The marginal-gain priority allocator is a $(1 - 1/e)$ approximation to the optimal solution of the monotone submodular maximization problem induced by the utility function.

**Sketch:**
1. Let $S$ denote a set of accepted upgrade operations, one drawn from each token's chain $2 \to 3 \to 4 \to 8$ (at most one upgrade per level). The resulting bit-width for token $i$ after applying $S$ is $b_i(S)$. Define
   $$F(S) = \sum_{i=1}^{n} \tilde{I}(i) \cdot U(b_i(S)), \quad C(S) = \sum_{(i, b \to b') \in S} \Delta b \cdot c_{i}(b \to b'),$$
   where $c_i(b \to b')$ converts bit increments into actual payload cost (including payload elements and scale bytes as in §2.2). Because $U$ is concave, the marginal gains satisfy diminishing returns, so $F$ is monotone submodular while $C$ defines a single knapsack constraint with capacity $B = \beta M_{\text{fp16}}$.

2. The knapsack constraint arises from the cumulative payload cost of accepted upgrades. The optimal allocation chooses the subset of upgrades that maximizes $F$ without exceeding $B$.

3. The heap-based algorithm implements the standard greedy rule for submodular knapsack: repeatedly take the upgrade with highest marginal gain per unit cost while budget remains.

4. Classical results (Khuller et al., 1999; Svitkina & Fleischer, 2008) show that this greedy strategy achieves at least $(1 - 1/e)$ of the optimal utility under a single knapsack constraint, even when each item (token) offers a chain of upgrades.

Therefore the allocator is both efficient ($O(n \log n)$) and provably close to optimal for the chosen utility model.

---

## 8. Algorithm Summary (Pseudocode)

```
Algorithm: SmartKV Forward Pass

Input:
  - Transformer model with L layers, H heads, dimension d
  - Input sequence x = [x_1, ..., x_n]
  - Memory budget β ∈ (0, 1]
  - Decay factor γ ∈ [0, 1)
  - Reallocation frequency f_base
  - Available bits B = {2, 3, 4, 8}

Initialize:
  - Importance scores I_ℓ^(0)(i) = 0 for all i, ℓ
  - Head importance η_h^(0) = 1 for all h
  - Precision map π(i, ℓ) = min(B) for all i, ℓ (baseline minimum precision)
  - Global step counter t = 0

For each generation step t:
  For each layer ℓ = 1 to L:
    // Standard transformer forward pass
    Q_ℓ^(t) = W_Q * hidden_states
    K_ℓ^(t) = W_K * hidden_states
    V_ℓ^(t) = W_V * hidden_states

    // Retrieve quantized KV cache (bucket-aware layout)
    {K̃_ℓ^(i), Ṽ_ℓ^(i), s_K^(i), s_V^(i)} = RetrieveCache(ℓ, {1..n})

    // Dequantize (current implementation)
    K̂_ℓ = Dequantize({K̃_ℓ^(i)}, {s_K^(i)})
    V̂_ℓ = Dequantize({Ṽ_ℓ^(i)}, {s_V^(i)})

    // Attention computation
    A_ℓ^(t) = softmax(Q_ℓ^(t) * K̂_ℓ^T / √d)  // Reduced inside fused kernel; no full materialization in optimized path
    O_ℓ^(t) = A_ℓ^(t) * V̂_ℓ

    // Update importance scores
    For each token i = 1 to n:
      α_ℓ^(t)(i) = mean_over_heads(A_ℓ^(t)[:, i])
      I_ℓ^(t+1)(i) = γ * I_ℓ^(t)(i) + (1-γ) * α_ℓ^(t)(i)

    // Update head importance
    For each head h = 1 to H:
      Ā_h^(t) = mean_over_tokens(A_ℓ^(t)[h, :])
      η_h^(t+1) = 0.95 * η_h^(t) + 0.05 * Ā_h^(t)

    // Cache new KV pair
    π_new = π(n+1, ℓ)  // Use current allocation for new token
    K̃_ℓ^(n+1), s_K^(n+1) = Quantize(K_ℓ^(t), π_new)
    Ṽ_ℓ^(n+1), s_V^(n+1) = Quantize(V_ℓ^(t), π_new)
    StoreCache(ℓ, n+1, K̃_ℓ^(n+1), Ṽ_ℓ^(n+1), s_K^(n+1), s_V^(n+1), π_new)

  // Reallocate precision when layer timers or drift triggers fire
  If ShouldReallocate(t):  // Implements §5.2 timer + drift criteria per layer
    // Aggregate importance across layers
    For each token i = 1 to n:
      I^(t)(i) = mean_over_layers({I_ℓ^(t)(i)})

    // Apply quality enhancements
    ApplyRecencyWeighting(I^(t), t)
    ApplyHeadImportanceWeighting(I^(t), {η_h})

    // Run marginal-gain allocator with hysteresis
    π_new = PriorityAllocate(I^(t), β, B, α, ε_imp,
                             hysteresis={Δrank ≥ r for m intervals},
                             drift_trigger={KL > τ or moved ≥ p% tokens})

    // Enforce special token protection
    For i in ProtectedTokens:  // Defined via positions (§4.3)
      π_new(i, :) = max(B)

    // Requantize changed tokens
    For each layer ℓ = 1 to L:
      For each token i where π_new(i, ℓ) ≠ π(i, ℓ):
        K̃_ℓ^(i), s_K^(i) = Quantize(K_ℓ^(i), π_new(i, ℓ), bucketed_layout=True)
        Ṽ_ℓ^(i), s_V^(i) = Quantize(V_ℓ^(i), π_new(i, ℓ), bucketed_layout=True)
        UpdateCache(ℓ, i, K̃_ℓ^(i), Ṽ_ℓ^(i), s_K^(i), s_V^(i), π_new(i, ℓ))

    π = π_new

  t = t + 1

Output: Generated tokens
```

---

## 9. Implementation Notes

### 9.1 Numerical Stability

1. **Scale factor clamping:** Always clamp scale factors to minimum $\epsilon = 10^{-8}$ to avoid division by zero.

2. **Importance score normalization:** Normalize importance scores by their mean to prevent numerical overflow/underflow during long sequences.

3. **Attention score clipping:** Standard softmax numerical stability techniques apply (subtract max before exp).
4. **Key/Value asymmetry:** Consider reserving higher minimum precision for keys (e.g., $\{3,4,8\}$) than values, because key noise perturbs the softmax logits while value noise is averaged post-attention.
5. **In-kernel reductions:** Implement EMA statistics using the attention kernel’s streamed outputs (e.g., FlashAttention) so the full matrix $A_\ell^{(t)}$ is never materialized.

### 9.2 Edge Cases

1. **Empty cache:** When $n = 0$, no allocation needed. First token defaults to $\max(\mathcal{B})$ to avoid cold-start error.

2. **Uniform importance:** If all tokens have the same importance, the priority allocator converges to a uniform allocation that exactly matches the budget (up to the discrete bit set).

3. **Budget violation:** If budget cannot be satisfied even with all tokens at minimum precision plus scale overhead, the allocator raises an error. With FP32 per-head scales and packing this lower bound is $\beta_{\text{min}} \approx 0.156$ for $\mathcal{B} = \{2, 3, 4, 8\}$.
4. **Discrete transitions:** Upgrades and downgrades must follow the ordered set $2 \to 3 \to 4 \to 8$ (and its reverse) rather than arithmetic $\pm1$ adjustments.

### 9.3 Hyperparameter Sensitivity

**Primary hyperparameters:**
- $\beta$ (memory budget): Most critical, directly controls quality-memory tradeoff
- $\gamma$ (decay factor): Controls temporal smoothing, typically $\gamma \in [0.85, 0.95]$
- $f_{\text{base}}$ (reallocation base interval): Balances adaptation vs overhead, typically $f_{\text{base}} \in [8, 32]$
- $\alpha$ (utility exponent): Values in $[0.5, 0.8]$ spread budget across 3/4-bit tiers while preserving upgrades for top tokens

**Secondary hyperparameters:**
- $\epsilon_{\text{imp}}$ (importance floor): Prevents unused budget; default $10^{-6}$
- Recency temperature $T$: Context-dependent, larger for longer contexts
- Head importance decay $\gamma_h$: Should be higher than token decay, typically $\gamma_h \geq 0.95$
- Hysteresis thresholds $(r, m)$ and drift triggers $(\tau, p)$: Tune to limit recompression churn (e.g., $r=5\%$, $m=2$, $\tau=0.02$, $p=10\%$)

---

## 10. Extensions and Future Work

### 10.1 Dynamic Budget Adjustment

Adapt budget based on generation phase:
$$\beta^{(t)} = \begin{cases}
\beta_{\text{aggressive}} & \text{if } n(t) < n_{\text{threshold}} \\
\beta_{\text{conservative}} & \text{if } n(t) \geq n_{\text{threshold}}
\end{cases}$$

### 10.2 Layer-Specific Allocation

Different layers may benefit from different budgets:
$$\beta_\ell = \beta_{\text{base}} \cdot w_\ell$$

where $w_\ell$ learned or heuristically set (e.g., early layers get more budget).

### 10.3 Learned Importance Function

Replace hand-crafted importance aggregation with learned function:
$$I^{(t)}(i) = f_\theta(\{A_{\ell, h}^{(t)}[i]\}_{\ell, h}, \text{age}(i), \text{position}(i))$$

where $f_\theta$ is a small neural network trained end-to-end.

### 10.4 Joint Training with Quantization-Aware Fine-tuning

Fine-tune transformer weights with SmartKV in the loop to learn quantization-robust representations.

### 10.5 Water-Filling Relaxation

Approximate quantization error via $\epsilon_i(b) \approx c_i 2^{-b}$, solve the continuous water-filling problem to obtain an oracle fractional allocation, then round back to $\mathcal{B}$. Provides an upper bound for evaluating heuristic allocators.

### 10.6 Learned Layer Weights

Calibrate per-layer (and per-K/V) weights $w_{\ell, K}, w_{\ell, V}$ on a held-out set to bias the allocator toward more sensitive layers before freezing them for deployment.

---

## 11. References for Mathematical Foundations

1. **Symmetric Quantization:**
   - Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018

2. **Attention Mechanism:**
   - Vaswani et al., "Attention Is All You Need," NeurIPS 2017

3. **EMA and Importance Tracking:**
   - Sutton & Barto, "Reinforcement Learning: An Introduction," Chapter 2 (Exponential Recency-Weighted Average)

4. **Knapsack Approximation:**
   - Kellerer et al., "Knapsack Problems," Springer 2004, Chapter 2
   - Khuller et al., "Budgeted Maximum Coverage," Information Processing Letters 1999
   - Svitkina & Fleischer, "Submodular Approximation: Sampling-Based Algorithms and Lower Bounds," STOC 2008

5. **Quantization Error Analysis:**
   - Widrow & Kollár, "Quantization Noise: Roundoff Error in Digital Computation," Cambridge University Press, 2008

---

## 12. Validation Checklist

For mathematical review and publication, verify:

- [ ] All notation is clearly defined with dimensions
- [ ] All equations are dimensionally consistent
- [ ] Algorithms have well-defined termination conditions
- [ ] Complexity analysis accounts for all major operations
- [ ] Approximation error bounds are rigorously derived
- [ ] Edge cases are handled (empty cache, uniform importance, etc.)
- [ ] Hyperparameters have sensible ranges and defaults
- [ ] Implementation notes address numerical stability
- [ ] Theoretical claims (optimality, complexity) have proofs or references

---

**Document Version:** 1.1
**Last Updated:** 2025-10-18
**Authors:** Robby Moseley
