"""Part 3 — analyse OLT-regularized CNN sweep and produce report figure."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part3")
METRICS = os.path.join(OUT_DIR, "metrics.json")


def main():
    if not os.path.exists(METRICS):
        raise SystemExit(f"Missing {METRICS}. Run scripts/train_part3_olt_nn.py on the server first.")
    with open(METRICS) as f:
        runs = json.load(f)

    lams = [r["lambda"] for r in runs]
    a_o  = [r["acc_orig"] for r in runs]
    a_r  = [r["acc_rot"]  for r in runs]
    ratio = [r["feature_diagnostics"]["ratio_global_over_perclass"] for r in runs]

    print(f"{'lambda':>8}  {'acc_orig':>9}  {'acc_rot':>8}  {'glob/per':>9}")
    for r in runs:
        print(f"{r['lambda']:>8.2f}  {r['acc_orig']:>9.4f}  {r['acc_rot']:>8.4f}  "
              f"{r['feature_diagnostics']['ratio_global_over_perclass']:>9.4f}")

    # Figure: accuracies vs lambda
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    ax.plot(lams, a_o, marker="o", label="orig MNIST")
    ax.plot(lams, a_r, marker="s", label="rotated MNIST")
    ax.set_xlabel(r"$\lambda$ (OLT loss weight)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Effect of OLT regularizer on CNN")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(lams, ratio, marker="^", color="C2")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\|F\|_* / \sum_c \|F_c\|_*$")
    ax.set_title("Feature subspace orthogonality\n(higher = more orthogonal classes)")
    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "part3_lambda_sweep.png"), dpi=160)
    plt.close(fig)
    print("Saved figure ->", os.path.join(OUT_DIR, "part3_lambda_sweep.png"))


if __name__ == "__main__":
    main()
