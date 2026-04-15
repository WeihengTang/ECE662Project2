"""
Re-plot all presentation figures with:
  - No figure titles; captions added below each figure
  - Larger text, clean sans-serif font (no bold)
  - Saved as new files in presentation/figs/ (slide_* prefix)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

# ── Data paths ────────────────────────────────────────────────────────
P1_RESULTS = "/home/www/ECE-662-Project/part3/results_part3"
P2_METRICS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "results", "part3", "metrics.json")

TASK_NAMES = ["gaussian", "motion", "defocus"]
NUM_TASKS = 3


def load_json(path):
    with open(path) as f:
        return json.load(f)


def slide_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 16,
        "font.weight": "normal",
        "axes.titlesize": 17,
        "axes.titleweight": "normal",
        "axes.labelsize": 16,
        "axes.labelweight": "normal",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.8,
        "lines.markersize": 10,
    })


def savefig(fig, name):
    path = os.path.join(FIGS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def add_caption(fig, text, y=-0.02):
    """Removed — captions are now handled in LaTeX via \\captionof."""
    pass


# ═══════════════════════════════════════════════════════════════════════
# P1 Fig 1 — Blur examples (kernels + blurred digits)
# ═══════════════════════════════════════════════════════════════════════

def plot_blur_examples():
    sys.path.insert(0, "/home/www/ECE-662-Project/part3")
    from dataset_blur import get_blur_kernel, BlurredMNIST

    # 稍微增加宽度的比例，给左侧的 ylabel 留出空间
    fig, axes = plt.subplots(2, NUM_TASKS, figsize=(4 * NUM_TASKS + 1, 5))

    for t in range(NUM_TASKS):
        # --- 第一行：Kernel ---
        kernel = get_blur_kernel(t).squeeze().numpy()
        axes[0, t].imshow(kernel, cmap="hot", interpolation="nearest")
        axes[0, t].set_title(f"Task {t}: {TASK_NAMES[t]}", fontsize=16, fontweight='bold', pad=15)
        
        if t == 0:
            # 在最左边添加行标签
            axes[0, t].set_ylabel("Kernel", fontsize=18, labelpad=15)

        # --- 第二行：Blurred Digit ---
        ds = BlurredMNIST(t, train=False,
                          data_root="/home/www/ECE-662-Project/part3/data")
        blurred, clean = ds[0]
        axes[1, t].imshow(blurred.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        
        # 【修改1】删除了原本的 axes[1, t].set_title("blurred digit")
        
        if t == 0:
            # 在最左边添加行标签
            axes[1, t].set_ylabel("Blurred digit", fontsize=18, labelpad=15)

        # --- 坐标轴处理 【修改2】 ---
        for r in range(2):
            # 不使用 axis("off")，而是手动隐藏刻度和边框
            axes[r, t].set_xticks([])
            axes[r, t].set_yticks([])
            for spine in axes[r, t].spines.values():
                spine.set_visible(False)

    # 调整布局，确保左侧标签不被切掉
    plt.tight_layout()
    add_caption(fig, "Synthetic blur tasks: PSF kernels (top) and example degraded digits (bottom).")
    savefig(fig, "slide_blur_examples.pdf")


# ═══════════════════════════════════════════════════════════════════════
# P1 Fig 2 — Forgetting curves
# ═══════════════════════════════════════════════════════════════════════

def plot_forgetting_curves(bl, dcf):
    fig, axes = plt.subplots(1, NUM_TASKS, figsize=(5 * NUM_TASKS, 4.5),
                             sharey=True)

    for j in range(NUM_TASKS):
        ax = axes[j]
        xs = list(range(NUM_TASKS))

        ys_bl = [bl["psnr_after"].get(str(t), {}).get(str(j), np.nan)
                 for t in range(NUM_TASKS)]
        ax.plot(xs, ys_bl, "o--", color="tab:red", label="Baseline (naive)")

        ys_dcf = [dcf["psnr_after"].get(str(t), {}).get(str(j), np.nan)
                  for t in range(NUM_TASKS)]
        ax.plot(xs, ys_dcf, "s-", color="tab:blue", label="DCF-CL (ours)")

        ax.set_xlabel("Training phase (after Task $t$)")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"T{t}" for t in xs])
        ax.set_title(f"PSNR on Task {j} ({TASK_NAMES[j]})")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("PSNR (dB)")
        if j == NUM_TASKS - 1:
            ax.legend(fontsize=13, loc="lower left")

    plt.tight_layout()
    add_caption(fig, "Forgetting curves: PSNR on each task after sequential training phases T0 \u2192 T1 \u2192 T2.")
    savefig(fig, "slide_forgetting_curves.pdf")


# ═══════════════════════════════════════════════════════════════════════
# P1 Fig 3 — PSNR heatmaps
# ═══════════════════════════════════════════════════════════════════════

def plot_psnr_heatmaps(bl, dcf):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for idx, (data, label) in enumerate([(bl, "Baseline"), (dcf, "DCF-CL")]):
        ax = axes[idx]
        mat = np.zeros((NUM_TASKS, NUM_TASKS))
        for t in range(NUM_TASKS):
            for j in range(NUM_TASKS):
                mat[t, j] = data["psnr_after"].get(str(t), {}).get(str(j), np.nan)

        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto",
                       vmin=np.nanmin(mat) - 1, vmax=np.nanmax(mat) + 1)
        for t in range(NUM_TASKS):
            for j in range(NUM_TASKS):
                v = mat[t, j]
                if not np.isnan(v):
                    ax.text(j, t, f"{v:.1f}", ha="center", va="center",
                            fontsize=14)

        ax.set_xticks(range(NUM_TASKS))
        ax.set_xticklabels([f"Task {j}\n{TASK_NAMES[j]}" for j in range(NUM_TASKS)],
                           fontsize=13)
        ax.set_yticks(range(NUM_TASKS))
        ax.set_yticklabels([f"After T{t}" for t in range(NUM_TASKS)],
                           fontsize=13)
        ax.set_xlabel("Evaluated on", fontsize=14)
        ax.set_ylabel("Trained through", fontsize=14)
        ax.set_title(label, fontsize=16)
        fig.colorbar(im, ax=ax, label="PSNR (dB)", shrink=0.8)

    plt.tight_layout()
    add_caption(fig, "PSNR matrix (rows = training phase, columns = evaluation task).")
    savefig(fig, "slide_psnr_heatmaps.pdf")


# ═══════════════════════════════════════════════════════════════════════
# P1 Fig 4 — Memory footprint
# ═══════════════════════════════════════════════════════════════════════

def plot_memory_footprint(bl, dcf):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    bl_per_task = bl["params_per_task"]
    bl_total = bl_per_task * NUM_TASKS

    dcf_atoms = dcf["shared_atom_params"]
    dcf_coeff_per_task = [dcf["coeff_per_task"].get(str(t), 0)
                          for t in range(NUM_TASKS)]
    dcf_total = dcf["total_dcf_memory"]

    width = 0.5

    # Baseline stacked bars
    bottom = 0
    for t in range(NUM_TASKS):
        ax.bar(0, bl_per_task, width, bottom=bottom,
               color=f"C{t}", alpha=0.7,
               label=f"Task {t} ({TASK_NAMES[t]})")
        bottom += bl_per_task

    # DCF-CL stacked bars
    ax.bar(1, dcf_atoms, width, color="tab:gray", alpha=0.8,
           label="Shared atoms")
    bottom = dcf_atoms
    for t in range(NUM_TASKS):
        ax.bar(1, dcf_coeff_per_task[t], width, bottom=bottom,
               color=f"C{t}", alpha=0.7)
        bottom += dcf_coeff_per_task[t]

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline\n(separate models)", "DCF-CL\n(ours)"],
                       fontsize=14)
    ax.set_ylabel("Total Stored Parameters", fontsize=14)
    ax.yaxis.get_major_formatter().set_scientific(False)

    max_val = max(bl_total, dcf_total)
    ax.text(0, bl_total * 1.02, f"{bl_total:,}",
            ha="center", fontsize=13)
    ax.text(1, dcf_total * 1.02, f"{dcf_total:,}",
            ha="center", fontsize=13)
    ratio = bl_total / dcf_total
    ax.text(1, dcf_total * 1.08, f"({ratio:.2f}\u00d7 compression)",
            ha="center", fontsize=12, color="tab:blue")

    ax.set_ylim(0, max_val * 1.35)
    ax.legend(fontsize=12, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    add_caption(fig, "Memory footprint for 3 tasks: baseline (separate models) vs. DCF-CL (shared atoms + per-task coefficients).")
    savefig(fig, "slide_memory_footprint.pdf")


# ═══════════════════════════════════════════════════════════════════════
# P2 Fig — Lambda sweep (accuracy + orthogonality)
# ═══════════════════════════════════════════════════════════════════════

def plot_lambda_sweep():
    with open(P2_METRICS) as f:
        runs = json.load(f)

    lams = [r["lambda"] for r in runs]
    a_o = [r["acc_orig"] for r in runs]
    a_r = [r["acc_rot"] for r in runs]
    ratio = [r["feature_diagnostics"]["ratio_global_over_perclass"] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(lams, a_o, marker="o", label="Original MNIST", linewidth=2.8,
            markersize=10)
    ax.plot(lams, a_r, marker="s", label="Rotated MNIST", linewidth=2.8,
            markersize=10)
    ax.set_xlabel(r"$\lambda$ (OLT loss weight)")
    ax.set_ylabel("Test accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(lams, ratio, marker="^", color="C2", linewidth=2.8, markersize=10)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\|F\|_* \;/\; \sum_c \|F_c\|_*$")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    add_caption(fig,
                "Left: clean and rotated MNIST accuracy vs. \u03bb.  "
                "Right: feature-subspace orthogonality ratio (higher = more orthogonal).")
    savefig(fig, "slide_lambda_sweep.png")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    slide_style()

    bl = load_json(os.path.join(P1_RESULTS, "baseline_cl_results.json"))
    dcf = load_json(os.path.join(P1_RESULTS, "dcf_cl_results.json"))

    plot_blur_examples()
    plot_forgetting_curves(bl, dcf)
    plot_psnr_heatmaps(bl, dcf)
    plot_memory_footprint(bl, dcf)
    plot_lambda_sweep()

    print("\nAll slide figures saved to:", FIGS_DIR)


if __name__ == "__main__":
    main()
