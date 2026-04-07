"""Part 2 Task 3 — joint OLT on top of FROZEN CNN features (LOCAL).

Consumes results/part2/cnn_features.npz produced by train_part2_cnn.py on
the server. Learns a joint OLT transformation on concatenated
(original + rotated) penultimate features, then compares:

    (a) frozen baseline CNN, classified by its own fc2 head
        (accuracies were computed on the server and stored in the npz)
    (b) frozen baseline CNN features + joint OLT + nearest subspace
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
from olt import olt_cccp, fit_class_subspaces, nearest_subspace_classifier  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part2")
FEATS = os.path.join(OUT_DIR, "cnn_features.npz")


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def main():
    if not os.path.exists(FEATS):
        raise SystemExit(f"Missing {FEATS}. Run scripts/train_part2_cnn.py on the server first.")

    d = np.load(FEATS)
    f_tr_o = d["f_tr_o"].astype(np.float64); f_tr_r = d["f_tr_r"].astype(np.float64)
    f_te_o = d["f_te_o"].astype(np.float64); f_te_r = d["f_te_r"].astype(np.float64)
    y_tr = d["y_tr"]; y_te = d["y_te"]
    acc_base_orig = float(d["baseline_acc_orig"])
    acc_base_rot  = float(d["baseline_acc_rot"])
    print(f"Feature shapes: f_tr_o {f_tr_o.shape}, f_te_o {f_te_o.shape}")
    print(f"Baseline frozen CNN: orig={acc_base_orig:.4f}  rot={acc_base_rot:.4f}")

    # Scale features so row norms average ~1
    scale = float(np.linalg.norm(f_tr_o, axis=1).mean())
    f_tr_o /= scale; f_tr_r /= scale
    f_te_o /= scale; f_te_r /= scale

    # Joint OLT on [orig, rot] features (feature-column concatenation)
    Ztr_joint = np.concatenate([f_tr_o, f_tr_r], axis=0)
    ytr_joint = np.concatenate([y_tr, y_tr], axis=0)
    Ztr_t = torch.from_numpy(Ztr_joint).double()
    ytr_t = torch.from_numpy(ytr_joint).long()

    T, hist = olt_cccp(
        Ztr_t, ytr_t,
        d_out=32, n_outer=80, n_inner=8, lr=0.5, verbose=False,
    )
    print(f"OLT obj: init={hist.objective[0]:.2f} best={min(hist.objective):.2f}")

    def apply(F): return (torch.from_numpy(F).double() @ T.T).float()
    F_tr = apply(Ztr_joint); y_sub = torch.from_numpy(ytr_joint).long()
    F_te_o = apply(f_te_o);  F_te_r = apply(f_te_r)

    subs = fit_class_subspaces(F_tr, y_sub, energy=0.95, max_rank=6)
    pred_o = nearest_subspace_classifier(F_te_o, subs).numpy()
    pred_r = nearest_subspace_classifier(F_te_r, subs).numpy()
    acc_olt_orig = accuracy(y_te, pred_o)
    acc_olt_rot  = accuracy(y_te, pred_r)
    print(f"Frozen + joint OLT:  orig={acc_olt_orig:.4f}  rot={acc_olt_rot:.4f}")

    results = {
        "baseline_frozen_cnn": {"orig": acc_base_orig, "rot": acc_base_rot},
        "frozen_plus_olt_ns":  {"orig": acc_olt_orig,  "rot": acc_olt_rot},
    }
    with open(os.path.join(OUT_DIR, "cnn_olt_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # bar chart
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    xs = np.arange(2); w = 0.35
    ax.bar(xs - w/2, [acc_base_orig, acc_base_rot], width=w, label="frozen CNN")
    ax.bar(xs + w/2, [acc_olt_orig,  acc_olt_rot ], width=w, label="frozen CNN + joint OLT")
    for x, v in zip(xs - w/2, [acc_base_orig, acc_base_rot]):
        ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for x, v in zip(xs + w/2, [acc_olt_orig,  acc_olt_rot]):
        ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(["original", "rotated"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("test accuracy")
    ax.set_title("MNIST: frozen CNN vs frozen CNN + joint OLT")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mnist_frozen_vs_olt.png"), dpi=160)
    plt.close(fig)

    print("Saved:", OUT_DIR)


if __name__ == "__main__":
    main()
