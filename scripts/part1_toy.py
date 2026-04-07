"""Part 1 Task 1 — OLT on 2D and 3D toy datasets.

Reproduces the starter-notebook synthetic data and applies the OLT
solver. Exports before/after visualizations and diagnostic plots.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from olt import olt_cccp  # noqa: E402
from olt.solver import olt_objective  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part1_toy")
os.makedirs(OUT_DIR, exist_ok=True)


def normalize(v, eps=1e-12):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


# ------------------------------ 3D toy -------------------------------------

def make_3d(n_per_class=150, line_noise=0.01, seed=0):
    torch.manual_seed(seed)
    u0 = torch.tensor([1.0, 0.3, 0.4])
    u1 = torch.tensor([0.4, 1.0, 0.2])
    u2 = torch.tensor([0.4, 0.2, 1.0])
    U = normalize(torch.stack([u0, u1, u2], dim=0))
    Xs, ys = [], []
    for c in range(3):
        t = torch.rand(n_per_class, 1)
        Xc = t * U[c].view(1, 3) + line_noise * torch.randn(n_per_class, 3)
        Xs.append(Xc)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(Xs), torch.cat(ys), U


def make_2d(n_per_class=150, line_noise=0.01, seed=0):
    torch.manual_seed(seed)
    angles = torch.tensor([35.0, 65.0]) * (np.pi / 180.0)
    U = normalize(torch.stack([torch.cos(angles), torch.sin(angles)], dim=1))
    Xs, ys = [], []
    for c in range(2):
        t = torch.rand(n_per_class, 1)
        Xc = t * U[c].view(1, 2) + line_noise * torch.randn(n_per_class, 2)
        Xs.append(Xc)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(Xs), torch.cat(ys), U


def top_pca_direction(Fc: torch.Tensor) -> torch.Tensor:
    Fc = Fc - Fc.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Fc, full_matrices=False)
    return Vh[0] / (Vh[0].norm() + 1e-12)


def plot_scatter_3d(X, y, title, path):
    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    for c in range(int(y.max()) + 1):
        pts = X[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, alpha=0.7, label=f"class {c}")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def plot_scatter_2d(X, y, title, path):
    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    for c in range(int(y.max()) + 1):
        pts = X[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.7, label=f"class {c}")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def plot_cos_matrix(C, title, path):
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    im = ax.imshow(C, vmin=0, vmax=1, cmap="viridis")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    color="white" if C[i, j] < 0.5 else "black", fontsize=9)
    ax.set_xticks(range(C.shape[0])); ax.set_yticks(range(C.shape[0]))
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def plot_singvals(F, y, title, path):
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    classes = sorted(set(y.tolist()))
    width = 0.8 / len(classes)
    for i, c in enumerate(classes):
        Fc = F[y == c]
        s = torch.linalg.svdvals(Fc).cpu().numpy()
        xs = np.arange(len(s)) + i * width
        ax.bar(xs, s, width=width, label=f"class {c}")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def plot_objective(hist, title, path):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(hist.objective, marker="o", ms=3)
    ax.set_xlabel("CCCP outer iteration")
    ax.set_ylabel(r"$f(T) = \sum_c \|TY_c\|_* - \|TY\|_*$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def run_case(name: str, X: torch.Tensor, y: torch.Tensor, d: int):
    print(f"\n=== {name} ({X.shape[0]} samples, d={d}) ===")
    # before
    cos_before = np.zeros((int(y.max()) + 1,) * 2)
    dirs_before = []
    for c in range(int(y.max()) + 1):
        dirs_before.append(top_pca_direction(X[y == c]))
    Db = torch.stack(dirs_before, dim=0)
    cos_before = (Db @ Db.T).abs().cpu().numpy()
    print("|cos| before:\n", cos_before)

    # OLT
    T, hist = olt_cccp(X, y, d_out=d, n_outer=40, n_inner=8, lr=0.05, verbose=False)
    print(f"objective: init={hist.objective[0]:.4f}  best={min(hist.objective):.4f}")

    F = (X.double() @ T.T).float()
    dirs_after = [top_pca_direction(F[y == c]) for c in range(int(y.max()) + 1)]
    Da = torch.stack(dirs_after, dim=0)
    cos_after = (Da @ Da.T).abs().cpu().numpy()
    print("|cos| after:\n", cos_after)

    # figures
    if d == 3:
        plot_scatter_3d(X, y, f"{name} — before OLT", os.path.join(OUT_DIR, f"{name}_before.png"))
        plot_scatter_3d(F, y, f"{name} — after OLT",  os.path.join(OUT_DIR, f"{name}_after.png"))
    else:
        plot_scatter_2d(X, y, f"{name} — before OLT", os.path.join(OUT_DIR, f"{name}_before.png"))
        plot_scatter_2d(F, y, f"{name} — after OLT",  os.path.join(OUT_DIR, f"{name}_after.png"))
    plot_cos_matrix(cos_before, f"{name} |cos| before", os.path.join(OUT_DIR, f"{name}_cos_before.png"))
    plot_cos_matrix(cos_after,  f"{name} |cos| after",  os.path.join(OUT_DIR, f"{name}_cos_after.png"))
    plot_singvals(F, y, f"{name} per-class singular values (after)",
                  os.path.join(OUT_DIR, f"{name}_singvals.png"))
    plot_objective(hist, f"{name} — CCCP objective", os.path.join(OUT_DIR, f"{name}_obj.png"))

    return {
        "name": name,
        "cos_before": cos_before.tolist(),
        "cos_after": cos_after.tolist(),
        "obj_init": hist.objective[0],
        "obj_best": min(hist.objective),
    }


def main():
    results = []
    X3, y3, _ = make_3d()
    results.append(run_case("toy3d", X3, y3, d=3))
    X2, y2, _ = make_2d()
    results.append(run_case("toy2d", X2, y2, d=2))

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
