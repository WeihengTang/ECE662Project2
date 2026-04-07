"""Part 1 Task 2 — OLT vs LDA on Extended YaleB (PCA preprocessed).

Pipeline
--------
1. Load YaleB (38 subjects, 64 images/subject, 2016 pixels).
2. Random stratified 50/50 train/test split.
3. PCA fit on the train set to d_pca components.
4. Compare three classifiers in the PCA space:
     (a) Raw PCA + nearest subspace (no OLT)
     (b) LDA (sklearn, projection + linear classifier)
     (c) OLT + nearest subspace classifier  <- our method
   Also report nearest-centroid for OLT for a sanity baseline.

The nearest-subspace rule: for each class c build an orthonormal basis of
the transformed training features (SVD, energy ≥ 0.95), then assign a
test feature x to argmin_c ||(I - B_c B_c^T) x||.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from olt import olt_cccp, fit_class_subspaces, nearest_subspace_classifier  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "YaleBCrop025.mat")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part1_yaleb")
os.makedirs(OUT_DIR, exist_ok=True)


def load_yaleb():
    d = loadmat(DATA)
    # Y: (2016, 64, 38) -> flatten to (n_samples, 2016) with labels
    Y = d["Y"]                                # (2016, 64, 38)
    n_feat, n_per, n_subj = Y.shape
    X = Y.transpose(2, 1, 0).reshape(n_subj * n_per, n_feat).astype(np.float64)
    y = np.repeat(np.arange(n_subj), n_per)
    return X, y


def stratified_split(y, frac_train: float, rng):
    idx_train, idx_test = [], []
    for c in np.unique(y):
        idxs = np.where(y == c)[0]
        rng.shuffle(idxs)
        k = int(round(frac_train * len(idxs)))
        idx_train.append(idxs[:k])
        idx_test.append(idxs[k:])
    return np.concatenate(idx_train), np.concatenate(idx_test)


def nearest_centroid_predict(F_train, y_train, F_test):
    centroids = {}
    for c in np.unique(y_train):
        centroids[int(c)] = F_train[y_train == c].mean(axis=0)
    classes = sorted(centroids.keys())
    C = np.stack([centroids[c] for c in classes], axis=0)
    # dist: (Ntest, C)
    dist = ((F_test[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return np.array(classes)[dist.argmin(axis=1)]


def accuracy(y_true, y_pred) -> float:
    return float((y_true == y_pred).mean())


def run():
    print("Loading YaleB …")
    X, y = load_yaleb()
    print(f"  shape {X.shape}, n_classes={len(np.unique(y))}")

    rng = np.random.default_rng(0)
    idx_tr, idx_te = stratified_split(y, 0.5, rng)
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    # ---- PCA ----
    d_pca = 150
    pca = PCA(n_components=d_pca, whiten=False, random_state=0)
    Z_tr = pca.fit_transform(X_tr)
    Z_te = pca.transform(X_te)
    # Scale features so typical row norm ~ 1 (keeps the OLT nuclear norms
    # in a sensible numerical range and decouples lr from dataset scale).
    scale = float(np.linalg.norm(Z_tr, axis=1).mean())
    Z_tr = Z_tr / scale
    Z_te = Z_te / scale
    print(f"PCA: d={d_pca}, explained variance = {pca.explained_variance_ratio_.sum():.3f}")

    results = {"pca_dim": d_pca, "train_size": len(y_tr), "test_size": len(y_te)}

    # ---- Baseline: nearest subspace directly on PCA features ----
    Ztr_t = torch.from_numpy(Z_tr).double()
    Zte_t = torch.from_numpy(Z_te).double()
    ytr_t = torch.from_numpy(y_tr).long()

    subs = fit_class_subspaces(Ztr_t, ytr_t, energy=0.95, max_rank=9)
    pred_pca_ns = nearest_subspace_classifier(Zte_t, subs).numpy()
    acc_pca_ns = accuracy(y_te, pred_pca_ns)
    print(f"[PCA only, nearest-subspace]       acc = {acc_pca_ns:.4f}")
    results["pca_nearest_subspace"] = acc_pca_ns

    # Nearest centroid on PCA features
    pred_pca_nc = nearest_centroid_predict(Z_tr, y_tr, Z_te)
    acc_pca_nc = accuracy(y_te, pred_pca_nc)
    print(f"[PCA only, nearest-centroid]       acc = {acc_pca_nc:.4f}")
    results["pca_nearest_centroid"] = acc_pca_nc

    # ---- LDA baseline ----
    lda = LinearDiscriminantAnalysis()
    lda.fit(Z_tr, y_tr)
    acc_lda = accuracy(y_te, lda.predict(Z_te))
    print(f"[LDA]                              acc = {acc_lda:.4f}")
    results["lda"] = acc_lda

    # ---- OLT ----
    d_out = 64   # reasonable embedding dimension (< n_classes * avg rank)
    print(f"Training OLT: d_in={d_pca}, d_out={d_out}")
    T, hist = olt_cccp(
        Ztr_t, ytr_t,
        d_out=d_out, n_outer=60, n_inner=8, lr=0.5, verbose=False,
    )
    print(f"  OLT objective: init={hist.objective[0]:.4f}  best={min(hist.objective):.4f}")
    F_tr = (Ztr_t @ T.T).float()
    F_te = (Zte_t @ T.T).float()

    subs_olt = fit_class_subspaces(F_tr, ytr_t, energy=0.95, max_rank=9)
    pred_olt_ns = nearest_subspace_classifier(F_te, subs_olt).numpy()
    acc_olt_ns = accuracy(y_te, pred_olt_ns)
    print(f"[OLT, nearest-subspace]            acc = {acc_olt_ns:.4f}")
    results["olt_nearest_subspace"] = acc_olt_ns

    pred_olt_nc = nearest_centroid_predict(F_tr.numpy(), y_tr, F_te.numpy())
    acc_olt_nc = accuracy(y_te, pred_olt_nc)
    print(f"[OLT, nearest-centroid]            acc = {acc_olt_nc:.4f}")
    results["olt_nearest_centroid"] = acc_olt_nc

    # Objective trace figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(hist.objective, marker="o", ms=3)
    ax.set_xlabel("CCCP outer iteration")
    ax.set_ylabel(r"$\sum_c\|TY_c\|_* - \|TY\|_*$")
    ax.set_title("YaleB: OLT objective")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "yaleb_obj.png"), dpi=160)
    plt.close(fig)

    # Bar chart of accuracies
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    names = ["PCA+NS", "PCA+NC", "LDA", "OLT+NS", "OLT+NC"]
    vals = [acc_pca_ns, acc_pca_nc, acc_lda, acc_olt_ns, acc_olt_nc]
    bars = ax.bar(names, vals, color=["#888", "#bbb", "#4a7abc", "#d27b2e", "#e8a56b"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"YaleB classification (PCA d={d_pca}, OLT d_out={d_out})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "yaleb_accuracy.png"), dpi=160)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    run()
