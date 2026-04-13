# Part 2 Task 3 Review Findings

## Findings

- No implementation bug stands out that would explain the OLT drop by itself. The checked-in analysis reproduces the saved result exactly: raw nearest-subspace gets `0.8989` on rotated MNIST and OLT+NS gets `0.8811`, matching [results/part2/cnn_olt_summary.json](/mnt/c/Users/Wayne/Documents/ECE662Project2/results/part2/cnn_olt_summary.json:2).
- The main methodology gap is that the repo does not document the claimed `d_out=64,96,128` sweep. The only OLT run in code is `d_out=32` at [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:75), and the saved summary contains only that one OLT result at [results/part2/cnn_olt_summary.json](/mnt/c/Users/Wayne/Documents/ECE662Project2/results/part2/cnn_olt_summary.json:10).
- The broader conclusion is also based on a single fixed seed and one 300/class subsample for OLT training, not a repeated experiment. See `SEED = 0` at [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:36) and the 300/class sampling at [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:155).
- Minor documentation issue: the comment says "feature-column concatenation," but the implementation actually concatenates samples by rows, which is the right thing for this row-major codebase. See [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:67).

## Answers

### 1. Is the feature extraction correct?

Yes.

- `MnistCNN.features()` returns after `fc1` and ReLU, before `dropout2` and `fc2`, at [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:49), [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:55), [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:60).
- Extraction happens under `torch.no_grad()` at [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:148), and `model.eval()` is set once before evaluation and extraction at [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:125). That is correct, though slightly brittle because `extract()` relies on caller state.
- The same MNIST normalization is reused for clean and rotated images via `norm()` at [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:92), [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:130), and [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:146).
- Rotations are per-sample uniform in `[-45,45]` at [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:35), [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:67), and [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:69).

### 2. Is the fair baseline (NS without OLT) computed correctly?

Yes.

- Raw original and rotated train features are stacked together with duplicated labels at [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:55) and [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:56).
- Both the raw path and OLT path use the same `fit_class_subspaces` and `nearest_subspace_classifier` functions at [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:62), [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:63), [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:83), and [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:84).
- The same scalar feature scaling is applied to both paths at [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:48) and [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:50).
- Minor caveat: the scalar is estimated from original-train features only, not the concatenated pool.

### 3. Is the OLT transformation applied correctly?

Yes.

- OLT is trained on the joint train pool with duplicated labels at [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:68) and [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:69).
- Test features are transformed as `F @ T.T` at [scripts/part2_cnn_features.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/part2_cnn_features.py:79).
- I cannot verify the claim that `d_out=64,96,128` behaved similarly, because that sweep is not in the checked-in code or results.

### 4. Is the OLT solver itself correct?

Basically yes.

- CCCP linearizes the concave `-||X T^T||_*` term using the current full-data subgradient at [olt/solver.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/olt/solver.py:143) and [olt/solver.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/olt/solver.py:147), then forms the convex-part subgradient from the classwise nuclear norms at [olt/solver.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/olt/solver.py:154) and [olt/solver.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/olt/solver.py:159).
- Spectral projection is applied after each inner step at [olt/solver.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/olt/solver.py:163).
- The gradient is scaled by `1/N` at [olt/solver.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/olt/solver.py:160).
- The solver returns `best_T`, not the last iterate, at [olt/solver.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/olt/solver.py:172) and [olt/solver.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/olt/solver.py:179).
- The main nuance is that the convex subproblem is not solved exactly; it is approximated by a fixed number of inner subgradient steps. That is a reasonable heuristic, not an obvious bug.
- `olt/losses.py` is not used here, but it is consistent with the same objective form at [olt/losses.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/olt/losses.py:35).

### 5. Is the conclusion valid?

Mostly yes, with narrower wording.

- The raw-NS vs OLT-NS comparison is fair: same frozen features, same train/test split, same label set, same classifier family, same scaling, same subspace-fitting code.
- The proposed explanation is plausible. Your subspace model assumes zero-centered linear structure, as stated in [olt/solver.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/olt/solver.py:190), while the CNN features are ReLU activations from [scripts/train_part2_cnn.py](/mnt/c/Users/Wayne/Documents/ECE662Project2/scripts/train_part2_cnn.py:55); those are often better described as clustered or affine rather than low-rank through the origin.
- I do not see a code bug that invalidates the negative result. I agree with the narrower statement: in this experiment, OLT hurt on frozen CNN features relative to the fair raw-NS baseline.
- I would not state the stronger universal claim "OLT does NOT help" without multi-seed runs and a recorded `d_out` sweep.
