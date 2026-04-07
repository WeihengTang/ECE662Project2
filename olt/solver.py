"""
Orthogonal Low-rank Transformation (OLT) solver.

Objective (column convention, Y has samples as columns):

    min_T  sum_c || T Y_c ||_*  -  || T Y ||_*     s.t.  ||T||_2 = 1

Using the row convention that is more natural in PyTorch, we store data as
X of shape (N, d) with rows = samples. Let F = X T^T with T of shape
(d_out, d), so F has shape (N, d_out). For any class-indexing subset X_c,

    || T Y_c ||_* = || X_c T^T ||_*  ,     || T Y ||_* = || X T^T ||_*

Optimization: Concave-Convex Procedure (CCCP).
The objective is a difference of two convex functions

    u(T) = sum_c || X_c T^T ||_*       (convex)
    v(T) =        || X   T^T ||_*       (convex)
    f(T) = u(T) - v(T)

CCCP step: linearize the concave term  -v(T)  around T^{(t)} via its
subgradient, and solve the remaining convex subproblem for T^{(t+1)}.

Subgradient of || X T^T ||_* w.r.t. T:
    let F = X T^T = U S V^T (thin SVD on F),
    d/dF || F ||_* = U V^T                  (subgradient; unique in the interior)
    d/dT || X T^T ||_* = (U V^T)^T X        (chain rule)

At each inner step we do a subgradient descent step on the convex
subproblem, then enforce the unit-spectral-norm constraint by dividing T
by its top singular value. This is the standard primal subgradient
method for a nuclear-norm problem and matches the nuclear-norm CCCP
scheme used in the OLE/OLT literature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


def _svd_subgradient(F: torch.Tensor, eig_thd: float = 1e-6) -> torch.Tensor:
    """Return U V^T, the subgradient of || . ||_* at F.

    Thin SVD; small singular values are discarded to make the subgradient
    stable when F is (numerically) rank-deficient.
    """
    # torch.linalg.svd returns Vh, not V
    U, S, Vh = torch.linalg.svd(F, full_matrices=False)
    keep = S > eig_thd
    if keep.sum() == 0:
        return torch.zeros_like(F)
    Uk = U[:, keep]
    Vhk = Vh[keep, :]
    return Uk @ Vhk


def nuclear_norm(F: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(F).sum()


def olt_objective(X: torch.Tensor, y: torch.Tensor, T: torch.Tensor) -> float:
    """f(T) = sum_c ||X_c T^T||_* - ||X T^T||_*."""
    F = X @ T.T
    total = nuclear_norm(F)
    per_class = 0.0
    for c in torch.unique(y):
        Fc = F[y == c]
        per_class = per_class + nuclear_norm(Fc)
    return float((per_class - total).item())


@dataclass
class OLTHistory:
    objective: list = field(default_factory=list)
    spectral: list = field(default_factory=list)


def olt_cccp(
    X: torch.Tensor,
    y: torch.Tensor,
    d_out: Optional[int] = None,
    n_outer: int = 30,
    n_inner: int = 5,
    lr: float = 0.05,
    tol: float = 1e-5,
    init: str = "pca",
    verbose: bool = False,
    device: Optional[str] = None,
) -> tuple[torch.Tensor, OLTHistory]:
    """Solve the OLT problem via CCCP.

    Parameters
    ----------
    X : (N, d) tensor  -- rows are samples.
    y : (N,)   tensor  -- integer class labels.
    d_out : output dimension of T (rows). Defaults to d.
    n_outer : outer CCCP iterations (re-linearization of concave term).
    n_inner : inner subgradient steps for the convex subproblem.
    lr : subgradient step size.
    init : 'pca' (top-d_out principal directions of X) or 'identity'.

    Returns
    -------
    T : (d_out, d) learned transformation (||T||_2 = 1).
    history : per-iteration bookkeeping.
    """
    if device is None:
        device = str(X.device)
    X = X.to(device).double()
    y = y.to(device)
    N, d = X.shape
    if d_out is None:
        d_out = d

    # Initialize T
    if init == "pca":
        Xc = X - X.mean(dim=0, keepdim=True)
        # top d_out right singular vectors of Xc give principal directions
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        T = Vh[:d_out].clone()  # (d_out, d)
    elif init == "identity":
        T = torch.zeros(d_out, d, device=device, dtype=X.dtype)
        k = min(d_out, d)
        T[:k, :k] = torch.eye(k, device=device, dtype=X.dtype)
    else:
        raise ValueError(f"Unknown init {init}")

    # project to unit spectral norm
    s_max = torch.linalg.svdvals(T).max()
    T = T / (s_max + 1e-12)

    classes = torch.unique(y).tolist()
    hist = OLTHistory()
    prev_obj = float("inf")
    best_obj = float("inf")
    best_T = T.clone()

    for outer in range(n_outer):
        # Linearize the concave term: compute subgradient of ||X T^T||_* at current T.
        with torch.no_grad():
            F_all = X @ T.T
            G_all = _svd_subgradient(F_all)            # (N, d_out)
            grad_concave_T = G_all.T @ X                # d/dT of ||X T^T||_*  -> (d_out, d)

        # Inner convex subproblem:
        #   min_T  sum_c ||X_c T^T||_*  -  <T, grad_concave_T>
        # Solve by a few subgradient descent steps.
        for _inner in range(n_inner):
            with torch.no_grad():
                grad_convex = torch.zeros_like(T)
                for c in classes:
                    Xc = X[y == c]
                    Fc = Xc @ T.T
                    Gc = _svd_subgradient(Fc)
                    grad_convex += Gc.T @ Xc
                # Scale gradient by 1/N so step size is dataset-size invariant.
                grad = (grad_convex - grad_concave_T) / N
                T = T - lr * grad
                # Project to unit spectral norm (constraint ||T||_2 = 1)
                s_max = torch.linalg.svdvals(T).max()
                T = T / (s_max + 1e-12)

        obj = olt_objective(X, y, T)
        hist.objective.append(obj)
        hist.spectral.append(float(torch.linalg.svdvals(T).max().item()))
        if verbose:
            print(f"[OLT] outer {outer:3d}  obj={obj:.6f}")
        if obj < best_obj:
            best_obj = obj
            best_T = T.clone()
        if abs(prev_obj - obj) < tol * max(1.0, abs(prev_obj)):
            break
        prev_obj = obj

    return best_T.detach(), hist


# ---------- Nearest-Subspace classifier used throughout the project ----------

def fit_class_subspaces(
    F_train: torch.Tensor,
    y_train: torch.Tensor,
    energy: float = 0.95,
    max_rank: Optional[int] = None,
) -> dict:
    """Fit a linear subspace per class from already-transformed features.

    Returns dict[class -> orthonormal basis (d_out, r_c)] where columns
    span the class subspace (centered at zero; features are assumed to
    lie on a low-dim subspace through the origin after OLT).
    """
    subspaces = {}
    for c in torch.unique(y_train).tolist():
        Fc = F_train[y_train == c]                         # (n_c, d_out)
        # SVD: Fc = U S Vh, rows of Vh (columns of Vh.T) span row-space (feature directions).
        U, S, Vh = torch.linalg.svd(Fc, full_matrices=False)
        total = (S ** 2).sum()
        cum = torch.cumsum(S ** 2, dim=0) / (total + 1e-12)
        r = int((cum < energy).sum().item()) + 1
        if max_rank is not None:
            r = min(r, max_rank)
        basis = Vh[:r].T                                   # (d_out, r)
        subspaces[int(c)] = basis
    return subspaces


def nearest_subspace_classifier(
    F_test: torch.Tensor,
    subspaces: dict,
) -> torch.Tensor:
    """Assign each test feature to the class whose subspace has minimum residual.

    residual_c(x) = ||x - B_c B_c^T x||_2
    """
    classes = sorted(subspaces.keys())
    residuals = []
    for c in classes:
        B = subspaces[c]                                   # (d_out, r_c)
        # projection onto subspace
        proj = F_test @ B @ B.T                            # (N, d_out)
        res = torch.linalg.norm(F_test - proj, dim=1)      # (N,)
        residuals.append(res)
    R = torch.stack(residuals, dim=1)                      # (N, C)
    idx = R.argmin(dim=1)
    classes_t = torch.tensor(classes, device=F_test.device)
    return classes_t[idx]
