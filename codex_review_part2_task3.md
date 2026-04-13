# Code Review Request: Part 2 Task 3 — Joint OLT on Frozen CNN Features

## Context

This is an ECE 662 course project on Orthogonal Low-Rank Transformation (OLT). OLT finds a linear map T that minimizes `sum_c ||T Y_c||_* - ||T Y||_*` subject to `||T||_2 = 1`, where `||.||_*` is the nuclear norm. The goal is to compress each class to a low-rank subspace while making class subspaces mutually orthogonal.

In Part 2 Task 3, we:
1. Train a CNN on Original MNIST (server-side, already done).
2. Freeze the CNN and extract 128-d penultimate (fc1) features for both Original and Rotated MNIST.
3. Learn a joint OLT transformation on the concatenated features from both modalities.
4. Evaluate using the nearest-subspace classifier.

## The Claim to Verify

We observed the following results:

| Method | Original MNIST | Rotated MNIST |
|---|---|---|
| Frozen CNN head (fc2) | 99.24% | 86.39% |
| Frozen CNN features + nearest-subspace (no OLT) | 98.96% | **89.89%** |
| Frozen CNN features + OLT + nearest-subspace | 98.50% | 88.11% |

**Our conclusion:** OLT does NOT help on top of frozen CNN features. The nearest-subspace classifier alone (without OLT) already achieves the best rotated accuracy. OLT slightly hurts (-1.8 pts on rotated vs the fair NS baseline). We attribute this to a mismatch: OLT assumes linear subspace structure (nuclear norm), but CNN features are nonlinear ReLU activations that are well-clustered but not necessarily low-rank in the matrix sense.

## What to Review

Please carefully examine the following files and answer these questions:

### 1. Is the feature extraction correct?
**File:** `scripts/train_part2_cnn.py` (lines ~90-170)
- Does `model.features(x)` correctly extract the penultimate layer (post-fc1, post-ReLU, pre-dropout, pre-fc2)?
- Are the features extracted with `model.eval()` and `torch.no_grad()`?
- Is the same normalization (mean=0.1307, std=0.3081) applied to both original and rotated images before feature extraction?
- Are the rotated images generated correctly (per-sample random rotation in [-45°, +45°])?

### 2. Is the fair baseline (NS without OLT) computed correctly?
**File:** `scripts/part2_cnn_features.py` (lines ~47-60)
- Are the raw CNN features from both modalities concatenated for training the subspace bases?
- Is the same `fit_class_subspaces` + `nearest_subspace_classifier` pipeline used for both the "no OLT" and "with OLT" evaluations?
- Is the same feature scaling applied to both paths?

### 3. Is the OLT transformation applied correctly?
**File:** `scripts/part2_cnn_features.py` (lines ~62-73)
- Is the joint OLT trained on concatenated [original, rotated] training features with duplicated labels (this is the correct formulation for modality-invariant OLT)?
- Is the learned T applied to test features correctly (`F @ T.T`)?
- Is `d_out=32` too aggressive a reduction from 128-d? (We tested d_out=64, 96, 128 and all gave similar results — OLT still underperformed raw NS in every case.)

### 4. Is the OLT solver itself correct?
**File:** `olt/solver.py`
- Does the CCCP correctly linearize the concave term and solve the convex subproblem?
- Is the spectral norm projection (`T / sigma_max(T)`) applied after each inner step?
- Is the gradient scaled by 1/N?
- Does the solver return the best-objective iterate (not the last)?

### 5. Is the conclusion valid?
Given what you find in the code:
- Is the comparison fair (same classifier, same features, same train/test split)?
- Is it plausible that OLT's linear low-rank assumption is a poor fit for nonlinear CNN features?
- Could there be a bug that explains OLT underperforming, or is the negative result genuine?

## Key Files to Read
- `scripts/train_part2_cnn.py` — CNN training + feature extraction (ran on server)
- `scripts/part2_cnn_features.py` — local analysis: raw NS baseline + OLT + comparison
- `olt/solver.py` — the CCCP OLT solver
- `olt/losses.py` — differentiable OLT loss (not used in this task, but related)
- `results/part2/cnn_olt_summary.json` — saved results

## Expected Output
Please provide:
1. Any bugs or issues you find.
2. Whether the comparison methodology is sound.
3. Whether you agree or disagree with the conclusion, and why.
