# Forecast-Guided Precision Allocation

SmartKV now supports *forecast-guided* precision allocation, enabling the cache to predict
which tokens will become important before they are queried again. This document summarizes
the design and how to enable/inspect the feature.

## Motivation

Traditional SmartKV allocation uses historical attention statistics and heuristics to decide
precision (bit-width) per token. While effective, it is reactive—precision changes occur only
after attention patterns are observed. Forecast-guided allocation introduces a lightweight
predictor that estimates future importance so we can pre-allocate bits to upcoming high-impact
tokens, reducing regret when attention shifts.

## Feature Extraction

When forecasting is enabled, SmartKV logs a feature vector for every token that is considered
by the allocator. The current feature set includes:

- `log_importance`: log of the most recent cumulative importance.
- `log_delta`: log of the absolute change since the previous allocation.
- `recency`: normalized age since the token was last referenced.
- `bits_norm`: current bit-width normalized by the maximum available bits.
- `head_mean`: mean head-importance statistic (after decay).
- `position_step`: combined normalized token position and global step.

Features are stored in a rolling buffer (default 2048 entries) along with the eventual
observed importance on the next attention update.

## Predictor

`smartkv/core/forecast.py` defines a tiny MLP (configurable hidden size) trained online with
Adam using mean squared error between predicted and true log-importance. The predictor mixes
its output with the current heuristic score via a blend factor (default 0.5):

```
predicted = model(feature)
final_score = (1 - blend) * raw_importance + blend * max(predicted, 0)
```

This blended score is then fed into the existing allocation routine, meaning all budget
constraints and re-quantization paths remain intact.

## Configuration

Forecasting can be enabled either directly on `SmartKVCache` or through model configs:

```python
cache = SmartKVCache(
    ...,
    enable_forecast=True,
    forecast_history=4096,
    forecast_update_interval=32,
    forecast_blend=0.5,
    forecast_lr=0.05,
)
```

For the LLaMA integration, the same knobs are exposed via `SmartKVConfig`.

Key parameters:

- `forecast_history`: size of the feature/target replay buffer.
- `forecast_update_interval`: number of samples before training the predictor.
- `forecast_blend`: weight of the predicted score vs. raw importance.
- `forecast_lr`: learning rate for the predictor.

## Diagnostics

- `SmartKVCache.forecast_last_loss` records the most recent training loss for quick health checks.
- `SmartKVCache.forecast_pending` tracks features awaiting supervised targets.
- Profiling script (`scripts/profile_smartkv.py`) now reports the last loss when forecasting is enabled.

## Testing

Unit test `tests/test_cache.py::TestForecasting::test_forecast_predictor_updates` exercises the
pipeline by triggering attention updates, allocation, and verifying that the predictor is trained.
GPU regression tests ensure fused kernels still pass parity checks when forecasting is active.

## Notes

- Forecasting introduces minimal overhead (a small MLP update every few allocations). For production
  usage, monitor `forecast_last_loss` and adjust hyperparameters as needed.
- The current feature set is intentionally lightweight; feel free to extend it with richer signal
  (hidden-state norms, token type embeddings) if additional accuracy is required.
