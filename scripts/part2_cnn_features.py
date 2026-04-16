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
from olt import (  # noqa: E402
    fit_affine_class_subspaces,
    fit_class_subspaces,
    nearest_affine_subspace_classifier,
    nearest_subspace_classifier,
    olt_cccp,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part2")
FEATS = os.path.join(OUT_DIR, "cnn_features.npz")


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def eval_origin_ns(F_train, y_train, F_test, y_test, energy=0.95, max_rank=6):
    subs = fit_class_subspaces(F_train, y_train, energy=energy, max_rank=max_rank)
    pred = nearest_subspace_classifier(F_test, subs).numpy()
    return accuracy(y_test, pred)


def eval_affine_ns(F_train, y_train, F_test, y_test, energy=0.95, max_rank=6):
    subs = fit_affine_class_subspaces(F_train, y_train, energy=energy, max_rank=max_rank)
    pred = nearest_affine_subspace_classifier(F_test, subs).numpy()
    return accuracy(y_test, pred)


def run_feature_suite(name, f_tr_o, f_tr_r, f_te_o, f_te_r, y_tr, y_te):
    print(f"\n=== Feature suite: {name} ===")
    print(f"Feature shapes: f_tr_o {f_tr_o.shape}, f_te_o {f_te_o.shape}")

    scale = float(np.linalg.norm(f_tr_o, axis=1).mean())
    f_tr_o = f_tr_o / scale
    f_tr_r = f_tr_r / scale
    f_te_o = f_te_o / scale
    f_te_r = f_te_r / scale

    Ztr_raw = np.concatenate([f_tr_o, f_tr_r], axis=0)
    ytr_raw = np.concatenate([y_tr, y_tr], axis=0)
    raw_tr_t = torch.from_numpy(Ztr_raw).float()
    raw_te_o = torch.from_numpy(f_te_o).float()
    raw_te_r = torch.from_numpy(f_te_r).float()
    ytr_raw_t = torch.from_numpy(ytr_raw).long()

    acc_raw_orig = eval_origin_ns(raw_tr_t, ytr_raw_t, raw_te_o, y_te)
    acc_raw_rot = eval_origin_ns(raw_tr_t, ytr_raw_t, raw_te_r, y_te)
    acc_aff_raw_orig = eval_affine_ns(raw_tr_t, ytr_raw_t, raw_te_o, y_te)
    acc_aff_raw_rot = eval_affine_ns(raw_tr_t, ytr_raw_t, raw_te_r, y_te)
    print(f"Raw features + origin-NS: orig={acc_raw_orig:.4f}  rot={acc_raw_rot:.4f}")
    print(f"Raw features + affine-NS: orig={acc_aff_raw_orig:.4f}  rot={acc_aff_raw_rot:.4f}")

    Ztr_mu = Ztr_raw.mean(axis=0, keepdims=True)
    Ztr_ctr = Ztr_raw - Ztr_mu
    teo_ctr = f_te_o - Ztr_mu
    ter_ctr = f_te_r - Ztr_mu
    ctr_tr_t = torch.from_numpy(Ztr_ctr).float()
    ctr_te_o = torch.from_numpy(teo_ctr).float()
    ctr_te_r = torch.from_numpy(ter_ctr).float()
    acc_ctr_orig = eval_origin_ns(ctr_tr_t, ytr_raw_t, ctr_te_o, y_te)
    acc_ctr_rot = eval_origin_ns(ctr_tr_t, ytr_raw_t, ctr_te_r, y_te)
    print(f"Global-centered raw + origin-NS: orig={acc_ctr_orig:.4f}  rot={acc_ctr_rot:.4f}")

    Ztr_t = torch.from_numpy(Ztr_raw).double()
    Ztr_ctr_t = torch.from_numpy(Ztr_ctr).double()
    ytr_t = torch.from_numpy(ytr_raw).long()
    olt_sweep = {}

    print("\nOLT d_out sweep:")
    for d_out in [32, 64, 96, 128]:
        T, hist = olt_cccp(Ztr_t, ytr_t, d_out=d_out, n_outer=80, n_inner=8, lr=0.5, verbose=False)
        T_ctr, hist_ctr = olt_cccp(Ztr_ctr_t, ytr_t, d_out=d_out, n_outer=80, n_inner=8, lr=0.5, verbose=False)

        def apply(Tm, F): return (torch.from_numpy(F).double() @ Tm.T).float()

        F_tr = apply(T, Ztr_raw)
        F_te_o_olt = apply(T, f_te_o)
        F_te_r_olt = apply(T, f_te_r)
        ao = eval_origin_ns(F_tr, ytr_t, F_te_o_olt, y_te)
        ar = eval_origin_ns(F_tr, ytr_t, F_te_r_olt, y_te)

        F_tr_ctr = apply(T_ctr, Ztr_ctr)
        F_te_o_ctr = apply(T_ctr, teo_ctr)
        F_te_r_ctr = apply(T_ctr, ter_ctr)
        ao_aff = eval_affine_ns(F_tr_ctr, ytr_t, F_te_o_ctr, y_te)
        ar_aff = eval_affine_ns(F_tr_ctr, ytr_t, F_te_r_ctr, y_te)

        print(
            f"  d_out={d_out:3d}  "
            f"orig-NS={ao:.4f}/{ar:.4f}  "
            f"center+affine={ao_aff:.4f}/{ar_aff:.4f}"
        )
        olt_sweep[d_out] = {
            "origin_ns": {"orig": ao, "rot": ar, "obj_best": min(hist.objective)},
            "centered_affine_ns": {
                "orig": ao_aff,
                "rot": ar_aff,
                "obj_best": min(hist_ctr.objective),
            },
        }

    return {
        "raw_origin_ns": {"orig": acc_raw_orig, "rot": acc_raw_rot},
        "raw_affine_ns": {"orig": acc_aff_raw_orig, "rot": acc_aff_raw_rot},
        "raw_global_center_origin_ns": {"orig": acc_ctr_orig, "rot": acc_ctr_rot},
        "olt_d_out_sweep": olt_sweep,
        "primary_origin_olt_ns": olt_sweep[32]["origin_ns"],
        "primary_centered_affine_olt_ns": olt_sweep[32]["centered_affine_ns"],
    }


def main():
    if not os.path.exists(FEATS):
        raise SystemExit(f"Missing {FEATS}. Run scripts/train_part2_cnn.py on the server first.")

    d = np.load(FEATS)
    f_tr_o = d["f_tr_o"].astype(np.float64); f_tr_r = d["f_tr_r"].astype(np.float64)
    f_te_o = d["f_te_o"].astype(np.float64); f_te_r = d["f_te_r"].astype(np.float64)
    y_tr = d["y_tr"]; y_te = d["y_te"]
    acc_base_orig = float(d["baseline_acc_orig"])
    acc_base_rot  = float(d["baseline_acc_rot"])
    print(f"Baseline frozen CNN: orig={acc_base_orig:.4f}  rot={acc_base_rot:.4f}")
    post_relu = run_feature_suite("post_relu_fc1", f_tr_o, f_tr_r, f_te_o, f_te_r, y_tr, y_te)

    pre_relu = None
    if all(k in d.files for k in ["f_tr_o_pre", "f_tr_r_pre", "f_te_o_pre", "f_te_r_pre"]):
        pre_relu = run_feature_suite(
            "pre_relu_fc1",
            d["f_tr_o_pre"].astype(np.float64),
            d["f_tr_r_pre"].astype(np.float64),
            d["f_te_o_pre"].astype(np.float64),
            d["f_te_r_pre"].astype(np.float64),
            y_tr,
            y_te,
        )

    results = {
        "baseline_frozen_cnn_head":  {"orig": acc_base_orig,  "rot": acc_base_rot},
        "post_relu_fc1": post_relu,
    }
    if pre_relu is not None:
        results["pre_relu_fc1"] = pre_relu
    with open(os.path.join(OUT_DIR, "cnn_olt_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # bar chart — three methods
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = np.arange(2); w = 0.25
    acc_raw_orig = post_relu["raw_origin_ns"]["orig"]
    acc_raw_rot = post_relu["raw_origin_ns"]["rot"]
    acc_olt_orig = post_relu["primary_origin_olt_ns"]["orig"]
    acc_olt_rot = post_relu["primary_origin_olt_ns"]["rot"]
    vals = [
        ("CNN head",          [acc_base_orig, acc_base_rot]),
        ("CNN feat + NS",     [acc_raw_orig,  acc_raw_rot]),
        ("CNN feat + OLT+NS", [acc_olt_orig,  acc_olt_rot]),
    ]
    for i, (lbl, vs) in enumerate(vals):
        bars = ax.bar(xs + (i - 1) * w, vs, width=w, label=lbl)
        for x, v in zip(xs + (i - 1) * w, vs):
            ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(["original", "rotated"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("test accuracy")
    ax.set_title("MNIST: frozen CNN — effect of joint OLT on features")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mnist_frozen_vs_olt.png"), dpi=160)
    plt.close(fig)

    print("Saved:", OUT_DIR)


if __name__ == "__main__":
    main()
