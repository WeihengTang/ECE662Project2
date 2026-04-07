"""Part 3 Option A — OLT-regularized CNN on MNIST (SERVER / GPU).

We extend the OLT framework to a neural-network loss. Let f_theta(x) be
the penultimate-layer feature of a CNN classifier, and let F_B be the
matrix of features for a mini-batch with labels y_B. The training
objective is:

    L = CrossEntropy(W f_theta(x), y) + lambda * L_OLT(F_B, y_B)

where the OLT regularizer is

    L_OLT(F, y) = ( sum_c || F_c ||_*  -  || F ||_* ) / |B|

The first sum encourages the per-class feature blocks to be low-rank
(intra-class compression), and subtracting || F ||_* — which is upper
bounded by sum_c || F_c ||_* with equality iff the class subspaces are
orthogonal — drives orthogonality across classes. Combined with
cross-entropy, this should yield features that are better separated and
more invariant to nuisance perturbations such as rotation.

This script trains
    baseline      lambda = 0
    olt_low       lambda = 0.1
    olt_mid       lambda = 0.5
    olt_high      lambda = 1.0
on Original MNIST and evaluates each on Original AND Rotated MNIST.

All variants share the same seed, architecture, optimizer, schedule and
data order — only the lambda differs.

Outputs (under results/part3/):
    metrics.json     accuracies and per-class nuclear-norm diagnostics
    cnn_lambda_*.pt  per-variant model weights
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))
from olt.losses import OLTLoss  # noqa: E402

DATA_DIR = os.path.join(THIS_DIR, "..", "data")
OUT_DIR  = os.path.join(THIS_DIR, "..", "results", "part3")
os.makedirs(OUT_DIR, exist_ok=True)

ROT_DEGREES = (-45.0, 45.0)
SEED = 0


class MnistCNN(nn.Module):
    """Same architecture as scripts/train_part2_cnn.py for fair comparison."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x, return_features=False):
        feat = self.features(x)
        x = self.dropout2(feat)
        logits = self.fc2(x)
        if return_features:
            return logits, feat
        return logits


def rotate_batch_np(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Xt = torch.from_numpy(X).unsqueeze(1).float()
    out = torch.empty_like(Xt)
    angs = rng.uniform(*ROT_DEGREES, size=X.shape[0])
    for i in range(X.shape[0]):
        out[i] = TF.rotate(Xt[i:i + 1], float(angs[i]))
    return out.squeeze(1).numpy()


def make_loaders(batch_size: int):
    tr = torchvision.datasets.MNIST(DATA_DIR, train=True,  download=True)
    te = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True)
    Xtr = tr.data.numpy().astype(np.float32) / 255.0
    ytr = tr.targets.numpy().astype(np.int64)
    Xte = te.data.numpy().astype(np.float32) / 255.0
    yte = te.targets.numpy().astype(np.int64)

    mean, std = 0.1307, 0.3081
    def norm(x): return (x - mean) / std

    Xtr_t = torch.from_numpy(norm(Xtr)).unsqueeze(1)
    ytr_t = torch.from_numpy(ytr)
    train_ds = TensorDataset(Xtr_t, ytr_t)
    g = torch.Generator(); g.manual_seed(SEED)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    return train_dl, Xte, yte, norm


def evaluate(model, X_np, y_np, norm, device, batch=512):
    model.eval()
    xt = torch.from_numpy(norm(X_np)).unsqueeze(1)
    preds = []
    with torch.no_grad():
        for i in range(0, len(xt), batch):
            preds.append(model(xt[i:i+batch].to(device)).argmax(1).cpu().numpy())
    return float((np.concatenate(preds) == y_np).mean())


def feature_diagnostics(model, X_np, y_np, norm, device, n_per_class=200):
    """Mean per-class nuclear norm and global nuclear norm of frozen features.

    Lower per-class / global ratio means stronger inter-class orthogonality
    (the OLT objective directly).
    """
    model.eval()
    rng = np.random.default_rng(0)
    idx = []
    for c in range(10):
        ii = np.where(y_np == c)[0]; rng.shuffle(ii); idx.append(ii[:n_per_class])
    idx = np.concatenate(idx)
    xt = torch.from_numpy(norm(X_np[idx])).unsqueeze(1).to(device)
    with torch.no_grad():
        feats = []
        for i in range(0, len(xt), 256):
            feats.append(model.features(xt[i:i+256]))
        F_all = torch.cat(feats, 0)                # (N, 128)
    yy = y_np[idx]
    per_class = []
    for c in range(10):
        Fc = F_all[yy == c]
        per_class.append(float(torch.linalg.svdvals(Fc).sum()))
    total = float(torch.linalg.svdvals(F_all).sum())
    return {
        "sum_per_class_nuclear": float(np.sum(per_class)),
        "global_nuclear": total,
        "ratio_global_over_perclass": total / max(1e-9, float(np.sum(per_class))),
    }


def train_one(lam: float, args, device):
    print(f"\n========== training lambda={lam} ==========")
    torch.manual_seed(SEED); np.random.seed(SEED)
    train_dl, Xte, yte, norm = make_loaders(args.batch_size)

    model = MnistCNN().to(device)
    opt = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.gamma)
    olt = OLTLoss(lam=1.0).to(device)   # lam scales total contribution below

    for epoch in range(args.epochs):
        model.train()
        n, correct, ce_sum, olt_sum = 0, 0, 0.0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, feat = model(xb, return_features=True)
            ce = F.cross_entropy(logits, yb)
            if lam > 0:
                reg = olt(feat, yb)
            else:
                reg = torch.zeros((), device=device)
            loss = ce + lam * reg
            loss.backward(); opt.step()
            ce_sum  += float(ce)  * xb.size(0)
            olt_sum += float(reg) * xb.size(0)
            correct += int((logits.argmax(1) == yb).sum())
            n += xb.size(0)
        sch.step()
        print(f"  ep{epoch+1:2d}  ce={ce_sum/n:.4f}  olt={olt_sum/n:.4f}  acc={correct/n:.4f}")

    # Evaluate on clean and rotated test
    rng = np.random.default_rng(SEED)
    Xte_rot = rotate_batch_np(Xte, rng)
    acc_orig = evaluate(model, Xte, yte, norm, device)
    acc_rot  = evaluate(model, Xte_rot, yte, norm, device)
    diag     = feature_diagnostics(model, Xte, yte, norm, device)
    print(f"  -> orig={acc_orig:.4f}  rot={acc_rot:.4f}  "
          f"global/per_class={diag['ratio_global_over_perclass']:.4f}")

    name = f"cnn_lambda_{lam:g}.pt".replace(".", "p", 1) if "." in str(lam) else f"cnn_lambda_{lam}.pt"
    name = f"cnn_lambda_{lam}.pt"
    torch.save(model.state_dict(), os.path.join(OUT_DIR, name))
    return {
        "lambda": lam,
        "acc_orig": acc_orig,
        "acc_rot":  acc_rot,
        "feature_diagnostics": diag,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1.0)
    p.add_argument("--gamma",      type=float, default=0.7)
    p.add_argument("--lambdas",    type=float, nargs="+",
                   default=[0.0, 0.1, 0.5, 1.0])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    results = []
    for lam in args.lambdas:
        results.append(train_one(lam, args, device))

    out_path = os.path.join(OUT_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print("\nSummary:")
    print(f"{'lambda':>8}  {'acc_orig':>9}  {'acc_rot':>8}  {'glob/per':>9}")
    for r in results:
        print(f"{r['lambda']:>8.2f}  {r['acc_orig']:>9.4f}  {r['acc_rot']:>8.4f}  "
              f"{r['feature_diagnostics']['ratio_global_over_perclass']:>9.4f}")


if __name__ == "__main__":
    main()
