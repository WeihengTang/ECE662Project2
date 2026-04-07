"""Part 2 Tasks 1 & 2 — OLT on Original MNIST + Rotated MNIST.

Task 1  : learn T_orig and T_rot separately on raw (PCA) features.
          Report a 2x2 cross-modality accuracy matrix with the
          nearest-subspace classification rule.

Task 2  : learn a JOINT transformation T_joint with a feature-column
          concatenation formulation (the mathematically rigorous
          modality-invariance construction):

              Y_c^joint = [ Y_c^orig , Y_c^rot ]       (per-class)
              Y^joint   = [ Y^orig   , Y^rot   ]
              min_T   Σ_c ||T Y_c^joint||_*  -  ||T Y^joint||_*

          This forces the two modalities to share the SAME low-rank
          subspace per class, not merely be independently low-rank.

Per-class subsampling is used to keep nuclear-norm SVDs tractable
(full MNIST is 60k samples; SVD on n_c≈6000 per class would dominate
wall time).
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
import torchvision
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from olt import olt_cccp, fit_class_subspaces, nearest_subspace_classifier  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part2")
os.makedirs(OUT_DIR, exist_ok=True)

N_PER_CLASS_TRAIN = 300  # OLT train subset per class
N_PER_CLASS_TEST  = 200
PCA_DIM = 50
D_OUT   = 20
SEED    = 0

ROT_DEGREES = (-45.0, 45.0)


def load_mnist():
    tr = torchvision.datasets.MNIST(DATA_DIR, train=True,  download=True)
    te = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True)
    X_tr = tr.data.float() / 255.0         # (60000, 28, 28)
    y_tr = tr.targets.long()
    X_te = te.data.float() / 255.0
    y_te = te.targets.long()
    return X_tr, y_tr, X_te, y_te


def rotate_batch(X: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Per-sample random rotation in ROT_DEGREES range."""
    out = torch.empty_like(X)
    angles = rng.uniform(ROT_DEGREES[0], ROT_DEGREES[1], size=X.shape[0])
    for i in range(X.shape[0]):
        out[i] = TF.rotate(X[i:i + 1].unsqueeze(1), float(angles[i])).squeeze()
    return out


def balanced_subset(y: torch.Tensor, n_per_class: int, rng: np.random.Generator):
    idx = []
    for c in torch.unique(y).tolist():
        ii = np.where(y.numpy() == c)[0]
        rng.shuffle(ii)
        idx.append(ii[:n_per_class])
    return np.concatenate(idx)


def accuracy(y_true, y_pred) -> float:
    return float((y_true == y_pred).mean())


def eval_nearest_subspace(F_train, y_train, F_test, y_test, max_rank=6, energy=0.95):
    subs = fit_class_subspaces(F_train, y_train, energy=energy, max_rank=max_rank)
    pred = nearest_subspace_classifier(F_test, subs).cpu().numpy()
    return accuracy(y_test.cpu().numpy(), pred)


def main():
    rng = np.random.default_rng(SEED)
    print("Loading MNIST …")
    X_tr_all, y_tr_all, X_te_all, y_te_all = load_mnist()

    # ---- subset to balanced train/test ----
    idx_tr = balanced_subset(y_tr_all, N_PER_CLASS_TRAIN, rng)
    idx_te = balanced_subset(y_te_all, N_PER_CLASS_TEST,  rng)
    X_tr = X_tr_all[idx_tr]; y_tr = y_tr_all[idx_tr]
    X_te = X_te_all[idx_te]; y_te = y_te_all[idx_te]
    print(f"  train subset: {len(y_tr)} | test subset: {len(y_te)}")

    # ---- build Rotated MNIST for the same indices ----
    print("Building rotated modality …")
    Xr_tr = rotate_batch(X_tr, rng)
    Xr_te = rotate_batch(X_te, rng)

    # ---- flatten + PCA fit on ORIGINAL train ----
    Xtr_flat = X_tr.reshape(len(X_tr), -1).numpy().astype(np.float64)
    Xte_flat = X_te.reshape(len(X_te), -1).numpy().astype(np.float64)
    Xrtr_flat = Xr_tr.reshape(len(Xr_tr), -1).numpy().astype(np.float64)
    Xrte_flat = Xr_te.reshape(len(Xr_te), -1).numpy().astype(np.float64)

    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    Z_otr = pca.fit_transform(Xtr_flat)
    Z_ote = pca.transform(Xte_flat)
    Z_rtr = pca.transform(Xrtr_flat)
    Z_rte = pca.transform(Xrte_flat)

    scale = float(np.linalg.norm(Z_otr, axis=1).mean())
    Z_otr /= scale; Z_ote /= scale; Z_rtr /= scale; Z_rte /= scale
    print(f"PCA d={PCA_DIM}, explained var = {pca.explained_variance_ratio_.sum():.3f}")

    # tensors
    def T_(x): return torch.from_numpy(x).double()
    Ztr_o = T_(Z_otr); Zte_o = T_(Z_ote)
    Ztr_r = T_(Z_rtr); Zte_r = T_(Z_rte)
    ytr   = y_tr.long(); yte = y_te.long()

    def apply_T(Tm, X): return (X @ Tm.T).float()

    results = {"pca_dim": PCA_DIM, "d_out": D_OUT, "n_per_class_train": N_PER_CLASS_TRAIN}

    # ================================================================
    # Task 1 : per-modality OLT
    # ================================================================
    print("\n=== Task 1: per-modality OLT ===")
    T_orig, h_o = olt_cccp(Ztr_o, ytr, d_out=D_OUT, n_outer=60, n_inner=8, lr=0.5)
    T_rot,  h_r = olt_cccp(Ztr_r, ytr, d_out=D_OUT, n_outer=60, n_inner=8, lr=0.5)
    print(f"  T_orig obj: {h_o.objective[0]:.2f} -> {min(h_o.objective):.2f}")
    print(f"  T_rot  obj: {h_r.objective[0]:.2f} -> {min(h_r.objective):.2f}")

    # 2x2 accuracy matrix: rows = transform, cols = test modality
    acc = {}
    for tname, Tm in [("T_orig", T_orig), ("T_rot", T_rot)]:
        F_tr_o = apply_T(Tm, Ztr_o); F_tr_r = apply_T(Tm, Ztr_r)
        F_te_o = apply_T(Tm, Zte_o); F_te_r = apply_T(Tm, Zte_r)
        # train the classifier on the same modality the transform was trained on
        if tname == "T_orig":
            F_tr = F_tr_o
        else:
            F_tr = F_tr_r
        acc[(tname, "orig")] = eval_nearest_subspace(F_tr, ytr, F_te_o, yte)
        acc[(tname, "rot")]  = eval_nearest_subspace(F_tr, ytr, F_te_r, yte)
        print(f"  {tname}: orig={acc[(tname,'orig')]:.4f}  rot={acc[(tname,'rot')]:.4f}")

    task1_matrix = {
        "T_orig -> orig": acc[("T_orig", "orig")],
        "T_orig -> rot":  acc[("T_orig", "rot")],
        "T_rot  -> orig": acc[("T_rot",  "orig")],
        "T_rot  -> rot":  acc[("T_rot",  "rot")],
    }
    results["task1"] = task1_matrix

    # ================================================================
    # Task 2 : joint OLT with feature-column concatenation
    # ================================================================
    print("\n=== Task 2: joint OLT on concatenated modalities ===")
    # Stack samples from both modalities with duplicated labels:
    # that implements "Y_c^joint = [Y_c^o, Y_c^r]" (samples-as-rows).
    Ztr_joint = torch.cat([Ztr_o, Ztr_r], dim=0)
    ytr_joint = torch.cat([ytr,   ytr],   dim=0)
    T_joint, h_j = olt_cccp(
        Ztr_joint, ytr_joint,
        d_out=D_OUT, n_outer=60, n_inner=8, lr=0.5,
    )
    print(f"  T_joint obj: {h_j.objective[0]:.2f} -> {min(h_j.objective):.2f}")

    # Train subspaces using features from BOTH modalities (joint support)
    F_trj = apply_T(T_joint, Ztr_joint)
    F_te_o_j = apply_T(T_joint, Zte_o)
    F_te_r_j = apply_T(T_joint, Zte_r)
    acc_jo = eval_nearest_subspace(F_trj, ytr_joint, F_te_o_j, yte)
    acc_jr = eval_nearest_subspace(F_trj, ytr_joint, F_te_r_j, yte)
    print(f"  T_joint: orig={acc_jo:.4f}  rot={acc_jr:.4f}")
    results["task2"] = {"T_joint -> orig": acc_jo, "T_joint -> rot": acc_jr}

    # ================================================================
    # Figures
    # ================================================================
    # heatmap of cross-modality accuracies
    labels_row = ["T_orig", "T_rot", "T_joint"]
    labels_col = ["orig", "rot"]
    M = np.array([
        [acc[("T_orig", "orig")], acc[("T_orig", "rot")]],
        [acc[("T_rot",  "orig")], acc[("T_rot",  "rot")]],
        [acc_jo, acc_jr],
    ])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(2)); ax.set_xticklabels(labels_col)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels_row)
    for i in range(3):
        for j in range(2):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.5 else "black", fontsize=10)
    ax.set_xlabel("test modality")
    ax.set_title("MNIST cross-modality accuracy")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mnist_cross_modality.png"), dpi=160)
    plt.close(fig)

    # example of rotated digits for the report
    fig, axes = plt.subplots(2, 8, figsize=(8, 2.5))
    for k in range(8):
        axes[0, k].imshow(X_tr[k].numpy(), cmap="gray"); axes[0, k].axis("off")
        axes[1, k].imshow(Xr_tr[k].numpy(), cmap="gray"); axes[1, k].axis("off")
    axes[0, 0].set_title("orig", fontsize=9)
    axes[1, 0].set_title("rot",  fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mnist_samples.png"), dpi=160)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
