"""Part 2 Task 3 — train a 2-layer CNN on Original MNIST (SERVER / GPU).

Architecture: the pytorch/examples MNIST CNN
    Conv(1->32, 3) -> ReLU -> Conv(32->64, 3) -> ReLU -> MaxPool(2) ->
    Dropout(0.25) -> Flatten -> Linear(9216->128) -> ReLU -> Dropout(0.5) ->
    Linear(128->10)

Trains for 10 epochs, saves:
    results/part2/cnn_backbone.pt     (model state_dict)
    results/part2/cnn_features.npz    (penultimate-layer features + labels
                                       for both original and rotated test
                                       and a held-out training subset used
                                       later to fit the OLT transform.)

Run on the server.  No OLT is used here.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "part2")
os.makedirs(OUT_DIR, exist_ok=True)

ROT_DEGREES = (-45.0, 45.0)
SEED = 0


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)     # penultimate (128-d features)
        self.fc2 = nn.Linear(128, 10)

    def features_pre_fc2(self, x, post_relu: bool = True):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if post_relu:
            x = F.relu(x)
        return x

    def features(self, x):
        return self.features_pre_fc2(x, post_relu=True)

    def forward(self, x):
        feat = self.features(x)
        x = self.dropout2(feat)
        return self.fc2(x)


def rotate_batch_np(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Xt = torch.from_numpy(X).unsqueeze(1).float()              # (N,1,28,28)
    out = torch.empty_like(Xt)
    angs = rng.uniform(*ROT_DEGREES, size=X.shape[0])
    for i in range(X.shape[0]):
        out[i] = TF.rotate(Xt[i:i + 1], float(angs[i]))
    return out.squeeze(1).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.7)
    args = p.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    tr = torchvision.datasets.MNIST(DATA_DIR, train=True,  download=True)
    te = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True)
    Xtr = tr.data.numpy().astype(np.float32) / 255.0          # (60000,28,28)
    ytr = tr.targets.numpy().astype(np.int64)
    Xte = te.data.numpy().astype(np.float32) / 255.0
    yte = te.targets.numpy().astype(np.int64)

    # Standard MNIST normalization (matches pytorch/examples)
    mean, std = 0.1307, 0.3081
    def norm(x): return (x - mean) / std

    Xtr_t = torch.from_numpy(norm(Xtr)).unsqueeze(1)          # (N,1,28,28)
    ytr_t = torch.from_numpy(ytr)
    Xte_t = torch.from_numpy(norm(Xte)).unsqueeze(1)
    yte_t = torch.from_numpy(yte)

    train_ds = TensorDataset(Xtr_t, ytr_t)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = MnistCNN().to(device)
    opt = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.gamma)

    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward(); opt.step()
            loss_sum += float(loss) * xb.size(0)
            total += xb.size(0)
            correct += int((logits.argmax(1) == yb).sum())
        sch.step()
        train_acc = correct / total
        print(f"epoch {epoch+1:2d}  loss={loss_sum/total:.4f}  train_acc={train_acc:.4f}")

    # -------------------- evaluation (clean + rotated) -----------------
    model.eval()
    rng = np.random.default_rng(SEED)
    Xte_rot = rotate_batch_np(Xte, rng)

    def eval_set(X_np):
        xt = torch.from_numpy(norm(X_np)).unsqueeze(1)
        preds = []
        with torch.no_grad():
            for i in range(0, len(xt), 512):
                p_i = model(xt[i:i+512].to(device)).argmax(1).cpu().numpy()
                preds.append(p_i)
        return np.concatenate(preds)

    pred_orig = eval_set(Xte)
    pred_rot  = eval_set(Xte_rot)
    acc_orig = float((pred_orig == yte).mean())
    acc_rot  = float((pred_rot  == yte).mean())
    print(f"\n[baseline CNN]  test_orig={acc_orig:.4f}  test_rot={acc_rot:.4f}")

    # ------------ feature extraction (penultimate layer) ---------------
    def extract(X_np, labels, post_relu=True):
        xt = torch.from_numpy(norm(X_np)).unsqueeze(1)
        feats = []
        with torch.no_grad():
            for i in range(0, len(xt), 512):
                f_i = model.features_pre_fc2(
                    xt[i:i+512].to(device), post_relu=post_relu
                ).cpu().numpy()
                feats.append(f_i)
        return np.concatenate(feats, 0), labels

    # Subsample train to keep the OLT solver fast (300/class).
    rng2 = np.random.default_rng(SEED + 1)
    idx_tr = []
    for c in range(10):
        ii = np.where(ytr == c)[0]
        rng2.shuffle(ii)
        idx_tr.append(ii[:300])
    idx_tr = np.concatenate(idx_tr)
    Xtr_sub = Xtr[idx_tr]; ytr_sub = ytr[idx_tr]
    Xtr_rot_sub = rotate_batch_np(Xtr_sub, rng2)

    f_tr_o, _ = extract(Xtr_sub,      ytr_sub, post_relu=True)
    f_tr_r, _ = extract(Xtr_rot_sub,  ytr_sub, post_relu=True)
    f_te_o, _ = extract(Xte,          yte,     post_relu=True)
    f_te_r, _ = extract(Xte_rot,      yte,     post_relu=True)
    f_tr_o_pre, _ = extract(Xtr_sub,     ytr_sub, post_relu=False)
    f_tr_r_pre, _ = extract(Xtr_rot_sub, ytr_sub, post_relu=False)
    f_te_o_pre, _ = extract(Xte,         yte,     post_relu=False)
    f_te_r_pre, _ = extract(Xte_rot,     yte,     post_relu=False)

    np.savez_compressed(
        os.path.join(OUT_DIR, "cnn_features.npz"),
        f_tr_o=f_tr_o, y_tr=ytr_sub,
        f_tr_r=f_tr_r,
        f_te_o=f_te_o, y_te=yte,
        f_te_r=f_te_r,
        f_tr_o_pre=f_tr_o_pre,
        f_tr_r_pre=f_tr_r_pre,
        f_te_o_pre=f_te_o_pre,
        f_te_r_pre=f_te_r_pre,
        baseline_acc_orig=np.array(acc_orig),
        baseline_acc_rot =np.array(acc_rot),
    )
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "cnn_backbone.pt"))
    print(f"\nSaved: {OUT_DIR}/cnn_backbone.pt and cnn_features.npz")


if __name__ == "__main__":
    main()
