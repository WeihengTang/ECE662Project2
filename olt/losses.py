"""Differentiable OLT loss for end-to-end training of neural networks.

Given features F (N, d) and integer labels y (N,), compute

    L(F, y) = ( sum_c || F_c ||_*  -  lam * || F ||_* ) / N

torch.linalg.svd is differentiable, so this loss can be backpropagated
into a feature extractor. Small singular values are clamped for
numerical stability, matching the OLE reference implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _nuclear_norm_stable(F: torch.Tensor, eig_thd: float = 1e-6) -> torch.Tensor:
    """Sum of singular values of F, ignoring tiny ones for stability."""
    if F.numel() == 0 or F.shape[0] == 0:
        return F.new_zeros(())
    S = torch.linalg.svdvals(F)
    S = torch.where(S > eig_thd, S, torch.zeros_like(S))
    return S.sum()


class OLTLoss(nn.Module):
    """Orthogonal Low-rank Transformation loss used as a regularizer."""

    def __init__(self, lam: float = 1.0, eig_thd: float = 1e-6):
        super().__init__()
        self.lam = lam
        self.eig_thd = eig_thd

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        N = features.shape[0]
        classes = torch.unique(labels)
        per_class = features.new_zeros(())
        for c in classes:
            mask = labels == c
            if mask.sum() < 2:
                continue
            per_class = per_class + _nuclear_norm_stable(features[mask], self.eig_thd)
        total = _nuclear_norm_stable(features, self.eig_thd)
        return (per_class - self.lam * total) / max(N, 1)
