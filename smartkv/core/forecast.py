"""Forecasting utilities for SmartKV adaptive precision."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ForecastNet(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = 16):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ForecastPredictor:
    """Lightweight online regressor for forecasting token importance."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 16,
        lr: float = 5e-2,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = _ForecastNet(feature_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self._loss = None

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        features = features.to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(features)
        return preds.detach().cpu()

    def update(self, features: torch.Tensor, targets: torch.Tensor) -> float:
        features = features.to(self.device)
        targets = targets.to(self.device)
        self.model.train()
        preds = self.model(features)
        loss = F.mse_loss(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._loss = loss.item()
        return self._loss

    @property
    def last_loss(self) -> Optional[float]:
        return self._loss
