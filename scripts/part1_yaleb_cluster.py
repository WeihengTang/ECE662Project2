"""Part 1 Task 3 — OLT subspace clustering on YaleB.

Pipeline
--------
1. Load YaleB, PCA -> 150 dims, feature-scale normalize.
2. Baseline clustering: build a symmetric k-nearest-neighbor graph in PCA
   space and run spectral clustering with n_classes components.
3. Treat these pseudo-labels as ground-truth supervision and run the OLT
   CCCP solver (same one used in Task 2).
4. **Convert clustering to classification via OLT** (per the project
   spec): in the OLT-transformed feature space, fit a per-pseudo-class
   linear subspace and reassign every sample with the nearest-subspace
   classification rule. These reassigned labels are the OLT clustering
   output.
5. Compare baseline and OLT clusterings against the true labels with
   Clustering Accuracy (Hungarian), NMI, and ARI.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from olt import (                                       # noqa: E402
    olt_cccp,
    fit_class_subspaces,
    nearest_subspace_classifier,
    clustering_metrics,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "YaleBCrop025.mat")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part1_yaleb")
os.makedirs(OUT_DIR, exist_ok=True)


def load_yaleb():
    d = loadmat(DATA)
    Y = d["Y"]
    n_feat, n_per, n_subj = Y.shape
    X = Y.transpose(2, 1, 0).reshape(n_subj * n_per, n_feat).astype(np.float64)
    y = np.repeat(np.arange(n_subj), n_per)
    return X, y


def run():
    print("Loading YaleB …")
    X, y_true = load_yaleb()
    n_classes = len(np.unique(y_true))
    print(f"  shape {X.shape}, n_classes={n_classes}")

    # PCA + scale
    pca = PCA(n_components=150, random_state=0)
    Z = pca.fit_transform(X)
    scale = float(np.linalg.norm(Z, axis=1).mean())
    Z /= scale
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # ---- baseline: k-NN graph + spectral clustering ----
    k = 7
    W = kneighbors_graph(Z, n_neighbors=k, mode="connectivity", include_self=False)
    W = 0.5 * (W + W.T)                                # symmetrize
    sc = SpectralClustering(
        n_clusters=n_classes, affinity="precomputed",
        assign_labels="kmeans", random_state=0,
    )
    y_pseudo = sc.fit_predict(W.toarray())
    base_metrics = clustering_metrics(y_true, y_pseudo)
    print(f"[k-NN + spectral baseline]  acc={base_metrics['accuracy']:.4f} "
          f"nmi={base_metrics['nmi']:.4f} ari={base_metrics['ari']:.4f}")

    # ---- OLT supervised by pseudo-labels ----
    Zt = torch.from_numpy(Z).double()
    yp_t = torch.from_numpy(y_pseudo).long()
    T, hist = olt_cccp(
        Zt, yp_t,
        d_out=64, n_outer=60, n_inner=8, lr=0.5, verbose=False,
    )
    print(f"  OLT obj: init={hist.objective[0]:.2f}  best={min(hist.objective):.2f}")

    # ---- convert clustering -> classification via nearest-subspace ----
    F = (Zt @ T.T).float()
    subs = fit_class_subspaces(F, yp_t, energy=0.95, max_rank=9)
    y_olt = nearest_subspace_classifier(F, subs).numpy()
    olt_metrics = clustering_metrics(y_true, y_olt)
    print(f"[OLT (reassigned via NS)]   acc={olt_metrics['accuracy']:.4f} "
          f"nmi={olt_metrics['nmi']:.4f} ari={olt_metrics['ari']:.4f}")

    results = {
        "baseline_knn_spectral": base_metrics,
        "olt_nearest_subspace":  olt_metrics,
        "k_nn": k,
        "pca_dim": 150,
        "olt_d_out": 64,
    }

    # bar chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    groups = ["Acc", "NMI", "ARI"]
    base_vals = [base_metrics["accuracy"], base_metrics["nmi"], base_metrics["ari"]]
    olt_vals  = [olt_metrics["accuracy"],  olt_metrics["nmi"],  olt_metrics["ari"]]
    xs = np.arange(3)
    ax.bar(xs - 0.18, base_vals, width=0.35, label="k-NN + Spectral")
    ax.bar(xs + 0.18, olt_vals,  width=0.35, label="OLT + NS")
    for x, v in zip(xs - 0.18, base_vals):
        ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for x, v in zip(xs + 0.18, olt_vals):
        ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.08); ax.set_ylabel("score")
    ax.set_title("YaleB clustering: baseline vs OLT")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "yaleb_clustering.png"), dpi=160)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "clustering_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", OUT_DIR)


if __name__ == "__main__":
    run()
