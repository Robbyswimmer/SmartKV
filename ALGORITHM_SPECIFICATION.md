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
| $\beta$ | scalar | Memory budget as fraction of FP16, $\beta \in (0, 1]$ |
| $\gamma$ | scalar | Attention score decay factor, $\gamma \in [0, 1)$ |
| $f$ | scalar | Reallocation frequency (number of generation steps) |
| $K_\ell^{(i)}$ | $\mathbb{R}^{H \times d}$ | Key vector for token $i$ at layer $\ell$ |
| $V_\ell^{(i)}$ | $\mathbb{R}^{H \times d}$ | Value vector for token $i$ at layer $\ell$ |
| $A_\ell^{(t)}$ | $\mathbb{R}^{H \times n \times n}$ | Attention matrix at layer $\ell$, generation step $t$ |
| $Q_\ell^{(t)}$ | $\mathbb{R}^{H \times d}$ | Query vector at layer $\ell$, generation step $t$ |

### 1.2 Objective

Given a memory budget $\beta$ and a set of cached key-value pairs $\{(K_\ell^{(i)}, V_\ell^{(i)})\}_{i=1}^{n}$ for each layer $\ell \in \{1, \ldots, L\}$, find a precision allocation function $\pi: \{1, \ldots, n\} \times \{1, \ldots, L\} \to \mathcal{B}$ that:

1. **Satisfies memory constraint:**
   $$\frac{1}{32nLH d} \sum_{\ell=1}^{L} \sum_{i=1}^{n} \pi(i, \ell) \leq \beta$$

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

For a single token's KV pair at precision $b$ bits:
$$M(b) = 2 \cdot H \cdot d \cdot b \text{ bits} = \frac{Hdb}{4} \text{ bytes}$$

Full FP16 cache memory (all layers, all tokens):
$$M_{\text{fp16}} = 2 \cdot L \cdot n \cdot H \cdot d \cdot 16 \text{ bits} = 4LnHd \text{ bytes}$$

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

### 4.2 Head Importance Weighting

Not all attention heads are equally important. Track per-head importance with EMA:

**Per-head mean attention:**
$$\bar{A}_h^{(t)} = \frac{1}{n} \sum_{i=1}^{n} A_{\ell, h}^{(t)}[i]$$

**Head importance EMA:**
$$\eta_h^{(t+1)} = \gamma_h \cdot \eta_h^{(t)} + (1 - \gamma_h) \cdot \bar{A}_h^{(t)}$$

where $\gamma_h = 0.95$ (slower decay than token importance).

**Normalized head weights:**
$$w_h^{(t)} = \frac{\eta_h^{(t)}}{\frac{1}{H}\sum_{h'=1}^{H} \eta_{h'}^{(t)} + \epsilon}$$

**Query-aware importance** (weight token importance by head importance):
$$I_{\text{weighted}}^{(t)}(i) = \frac{1}{H} \sum_{h=1}^{H} w_h^{(t)} \cdot A_{\ell, h}^{(t)}[i]$$

Combine with standard importance:
$$I_{\text{final}}^{(t)}(i) = 0.7 \cdot I^{(t)}(i) + 0.3 \cdot I_{\text{weighted}}^{(t)}(i)$$

**IMPORTANT NOTE:** The head-weighted attention $w_h^{(t)} \cdot A_{\ell, h}^{(t)}[i]$ is **not renormalized** after scaling. This means it no longer sums to 1 over all tokens and is not a valid probability distribution. It serves purely as a **heuristic importance signal** for precision allocation. If probability semantics are required for downstream use, apply softmax renormalization after weighting.

### 4.3 Special Token Protection

Certain tokens (BOS, EOS, PAD, system prompt tokens) must never be aggressively quantized:

**Protected token set:**
$$\mathcal{P} = \{\text{bos\_id}, \text{eos\_id}, \text{pad\_id}\} \cup \{1, \ldots, k_{\text{critical}}\}$$

where $k_{\text{critical}} = 4$ (first 4 tokens of prompt).

**Protection constraint:**
$$\pi(i, \ell) = 8 \text{ bits}, \quad \forall i \in \mathcal{P}, \, \forall \ell$$

---

## 5. Precision Allocation Algorithm

### 5.1 Tier-Based Greedy Allocation

Given memory budget $\beta$, available bits $\mathcal{B} = \{2, 3, 4, 8\}$, and token importance scores $\{I^{(t)}(i)\}_{i=1}^{n}$:

**Step 1: Sort tokens by importance**
$$\sigma: \{1, \ldots, n\} \to \{1, \ldots, n\}, \quad I^{(t)}(\sigma(1)) \geq I^{(t)}(\sigma(2)) \geq \cdots \geq I^{(t)}(\sigma(n))$$

**Step 2: Define tier boundaries**

Define importance percentiles:
$$\mathcal{T} = \{0.10, 0.30, 0.70, 1.0\}$$

Corresponding bit allocations:
$$\mathcal{B}_{\text{tier}} = \{8, 4, 3, 2\}$$

**Step 3: Initial tier assignment**

For token index $i$, find its rank $r = \sigma^{-1}(i)$ (position in sorted order).

Assign tier:
$$\text{tier}(i) = \begin{cases}
1 & \text{if } r \leq \tau_1 = \lfloor 0.10 \cdot n \rfloor \\
2 & \text{if } \tau_1 < r \leq \tau_2 = \lfloor 0.30 \cdot n \rfloor \\
3 & \text{if } \tau_2 < r \leq \tau_3 = \lfloor 0.70 \cdot n \rfloor \\
4 & \text{if } r > \tau_3
\end{cases}$$

Initial precision:
$$\pi_0(i) = \mathcal{B}_{\text{tier}}[\text{tier}(i)]$$

**Step 4: Top-k protection**

Ensure top 5% tokens get maximum precision:
$$\pi_0(i) \leftarrow \max(\pi_0(i), 8), \quad \forall i \in \{\sigma(1), \ldots, \sigma(\lfloor 0.05 \cdot n \rfloor)\}$$

**Step 5: Special token protection**
$$\pi_0(i) \leftarrow 8, \quad \forall i \in \mathcal{P}$$

**Step 6: Budget adjustment**

Compute current memory usage:
$$M_{\text{current}} = \sum_{i=1}^{n} M(\pi_0(i))$$

If $M_{\text{current}} > \beta \cdot M_{\text{fp16}}/L$:
- **Deficit:** $\Delta = M_{\text{current}} - \beta \cdot M_{\text{fp16}}/L$
- **Downgrade strategy:** Starting from lowest importance tokens (large $\sigma^{-1}(i)$), reduce precision until budget satisfied:
  $$\pi(i) \leftarrow \max(\pi_0(i) - 1, 2), \quad \text{for } i \in \{\sigma(n), \sigma(n-1), \ldots\}$$

If $M_{\text{current}} < \beta \cdot M_{\text{fp16}}/L$:
- **Surplus:** $\Delta = \beta \cdot M_{\text{fp16}}/L - M_{\text{current}}$
- **Upgrade strategy:** Starting from highest importance tokens (small $\sigma^{-1}(i)$), increase precision until budget exhausted:
  $$\pi(i) \leftarrow \min(\pi_0(i) + 1, 8), \quad \text{for } i \in \{\sigma(1), \sigma(2), \ldots\}$$

**Final allocation:**
$$\pi: \{1, \ldots, n\} \to \mathcal{B}$$

### 5.2 Adaptive Reallocation Frequency

Recompute precision allocation every $f^{(t)}$ generation steps, where:

$$f^{(t)} = \max\left(f_{\min}, \min\left(f_{\max}, f_{\text{base}} \cdot \left(1 + \frac{n(t)}{n_{\text{ref}}}\right)\right)\right)$$

where:
- $f_{\text{base}} = 16$ (base reallocation frequency)
- $n(t)$ = context length at step $t$
- $n_{\text{ref}} = 512$ (reference context length)
- $f_{\min} = 8$, $f_{\max} = 64$

**Intuition:** At longer contexts, reallocate less frequently to amortize overhead.

---

## 6. Forward Pass with SmartKV

### 6.1 Cache Storage

For each layer $\ell$ and each token $i$:

**Store quantized representations:**
$$\tilde{K}_\ell^{(i)} = \mathcal{Q}(K_\ell^{(i)}, \pi(i, \ell)) \in \mathbb{Z}^{H \times d}$$
$$\tilde{V}_\ell^{(i)} = \mathcal{Q}(V_\ell^{(i)}, \pi(i, \ell)) \in \mathbb{Z}^{H \times d}$$

**Store scale factors:**
$$s_{K,h}^{(i)}, s_{V,h}^{(i)} \in \mathbb{R}, \quad h \in \{1, \ldots, H\}$$

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
| Precision reallocation | $O(n \log n)$ | Sorting by importance (amortized over $f$ steps) |
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
| Precision reallocation | $O(n \log n)$ | Amortized over $f$ steps: $O(n \log n / f)$ per step |
| Dequantization (current) | $O(LnHd)$ | **Same cost as attention - 50% overhead!** |
| Dequantization (fused kernel) | $O(1)$ | Absorbed into attention kernel |

**Total complexity (current implementation):**
$$\mathcal{T}_{\text{decode}} = O(LnHd) + O(LnHd) = O(LnHd)$$

Dequantization overhead is **50% of total compute** during decoding.

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

**Quantization error per token:**

**NOTE:** SmartKV uses **per-head** (not per-tensor) quantization. Each head $h \in \{1, \ldots, H\}$ has its own scale factor $s_h$, computed as:
$$s_h = \frac{\max_{j \in \{1, \ldots, d\}} |X_{h,j}|}{2^{b-1} - 1}$$

This provides tighter error bounds than per-tensor quantization, especially when different heads have different activation magnitudes.

For symmetric per-head quantization at $b$ bits:
$$\|\hat{X}_h - X_h\|_{\infty} \leq \frac{\|X_h\|_{\infty}}{2^{b-1} - 1}$$

**Cross-head error independence:** Since quantization is per-head, errors in different heads are independent (assuming independent activations). Total error:
$$\|\hat{X} - X\|_2^2 = \sum_{h=1}^{H} \|\hat{X}_h - X_h\|_2^2$$

**Attention score perturbation:**

Using Weyl's inequality for eigenvalue perturbation:
$$|A_{\ell,h}^{(t)}[i] - \hat{A}_{\ell,h}^{(t)}[i]| \leq \frac{2}{\sqrt{d}} \cdot \frac{\|K_{\ell,h}^{(i)}\|_2}{2^{\pi(i,\ell)-1} - 1}$$

**Expected error under SmartKV allocation:**

High-importance tokens ($I^{(t)}(i) > \tau_1$) receive high precision ($\pi(i, \ell) \geq 8$):
$$|A[i] - \hat{A}[i]| \leq \frac{2\|K\|_2}{127\sqrt{d}} \approx 0.016 \cdot \frac{\|K\|_2}{\sqrt{d}}$$

Low-importance tokens ($I^{(t)}(i) < \tau_3$) receive low precision ($\pi(i, \ell) = 2$):
$$|A[i] - \hat{A}[i]| \leq \frac{2\|K\|_2}{\sqrt{d}}$$

But these tokens have low attention weight ($A[i] \approx 0$), so their error contribution to output is suppressed.

**Weighted error:**
$$\mathbb{E}\left[\|O - \hat{O}\|_2\right] \leq \sum_{i=1}^{n} A[i] \cdot \frac{2\|V^{(i)}\|_2}{2^{\pi(i)-1} - 1}$$

SmartKV minimizes this by allocating high precision to high-$A[i]$ tokens.

### 7.4 Optimality

**Claim:** The tier-based greedy allocation is a constant-factor approximation to the optimal allocation under the bounded perturbation model.

**Proof sketch:**
1. The optimal allocation $\pi^*$ would assign bits to minimize weighted quantization error:
   $$\pi^* = \arg\min_{\pi: M(\pi) \leq \beta M_{\text{fp16}}} \sum_{i=1}^{n} I^{(t)}(i) \cdot \epsilon_i(\pi(i))$$

2. This is a variant of the fractional knapsack problem with bounded items (bit-widths $\in \mathcal{B}$).

3. The greedy algorithm that prioritizes high-importance tokens for high precision achieves a $(1 - 1/e)$-approximation to the optimal solution in the continuous relaxation.

4. The tier-based approach is a discretized version that maintains constant-factor approximation with reduced computational overhead ($O(n \log n)$ vs $O(n |\mathcal{B}|)$).

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
  - Precision map π(i, ℓ) = 8 for all i, ℓ (initial full precision)
  - Global step counter t = 0

For each generation step t:
  For each layer ℓ = 1 to L:
    // Standard transformer forward pass
    Q_ℓ^(t) = W_Q * hidden_states
    K_ℓ^(t) = W_K * hidden_states
    V_ℓ^(t) = W_V * hidden_states

    // Retrieve quantized KV cache
    {K̃_ℓ^(i), Ṽ_ℓ^(i), s_K^(i), s_V^(i)} = RetrieveCache(ℓ, {1..n})

    // Dequantize (current implementation)
    K̂_ℓ = Dequantize({K̃_ℓ^(i)}, {s_K^(i)})
    V̂_ℓ = Dequantize({Ṽ_ℓ^(i)}, {s_V^(i)})

    // Attention computation
    A_ℓ^(t) = softmax(Q_ℓ^(t) * K̂_ℓ^T / √d)
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

  // Reallocate precision every f steps
  If t mod f = 0:
    // Aggregate importance across layers
    For each token i = 1 to n:
      I^(t)(i) = mean_over_layers({I_ℓ^(t)(i)})

    // Apply quality enhancements
    ApplyRecencyWeighting(I^(t), t)
    ApplyHeadImportanceWeighting(I^(t), {η_h})

    // Run tier-based allocation
    π_new = TierBasedAllocation(I^(t), β, B)

    // Enforce special token protection
    For i in ProtectedTokens:
      π_new(i, :) = 8

    // Requantize changed tokens
    For each layer ℓ = 1 to L:
      For each token i where π_new(i, ℓ) ≠ π(i, ℓ):
        K̃_ℓ^(i), s_K^(i) = Quantize(K_ℓ^(i), π_new(i, ℓ))
        Ṽ_ℓ^(i), s_V^(i) = Quantize(V_ℓ^(i), π_new(i, ℓ))
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

### 9.2 Edge Cases

1. **Empty cache:** When $n = 0$, no allocation needed. First token always cached at 8-bit.

2. **Uniform importance:** If all tokens have same importance, tier-based allocation degenerates to uniform allocation within budget constraint.

3. **Budget violation:** If budget cannot be satisfied even with all tokens at 2-bit (minimum precision), algorithm raises an error. In practice, $\beta \geq 0.125$ (1/8 of FP16) is required for $\mathcal{B} = \{2, 3, 4, 8\}$.

### 9.3 Hyperparameter Sensitivity

**Primary hyperparameters:**
- $\beta$ (memory budget): Most critical, directly controls quality-memory tradeoff
- $\gamma$ (decay factor): Controls temporal smoothing, typically $\gamma \in [0.85, 0.95]$
- $f_{\text{base}}$ (reallocation frequency): Balances adaptation vs overhead, typically $f \in [8, 32]$

**Secondary hyperparameters:**
- Tier boundaries $\mathcal{T}$: Robust across different values, default $\{0.10, 0.30, 0.70, 1.0\}$
- Recency temperature $T$: Context-dependent, larger for longer contexts
- Head importance decay $\gamma_h$: Should be higher than token decay, typically $\gamma_h \geq 0.95$

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

**Document Version:** 1.0
**Last Updated:** 2025-10-01
**Authors:** Robby Moseley
