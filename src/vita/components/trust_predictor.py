from __future__ import annotations

from typing import Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrustPredictor(nn.Module):
    """Self-supervised trust estimator based on neighbor action consistency."""

    def __init__(self, hidden_dim: int, action_dim: int, gamma: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))
        self.gamma = float(gamma)

    def forward(
        self,
        neighbor_feat: torch.Tensor,
        neighbor_next_action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict neighbor actions and derive trust from prediction error.

        Args:
            neighbor_feat: [B, K, hidden_dim]
            neighbor_next_action: optional [B, K, action_dim] one-hot labels

        Returns:
            pred_actions: [B, K, action_dim] logits
            trust_mask: [B, K, 1] trust scores in (0, 1]
        """
        if neighbor_feat.dim() != 3:
            raise ValueError(f"Expected neighbor_feat to have shape [B, K, H], got {tuple(neighbor_feat.shape)}")

        logits = self.net(neighbor_feat)
        if neighbor_next_action is None:
            trust = torch.ones(neighbor_feat.size(0), neighbor_feat.size(1), 1, device=neighbor_feat.device)
        else:
            has_label = neighbor_next_action.sum(dim=-1, keepdim=True) > 1e-6
            log_probs = F.log_softmax(logits, dim=-1)
            ce = -(neighbor_next_action * log_probs).sum(dim=-1, keepdim=True)
            denom = math.log(max(int(logits.size(-1)), 2))
            ce = ce / denom
            trust = torch.exp(-self.gamma * ce)
            trust = torch.where(has_label, trust, torch.ones_like(trust))
        trust = torch.nan_to_num(trust, nan=1.0, posinf=1.0, neginf=1.0)
        return logits, trust
