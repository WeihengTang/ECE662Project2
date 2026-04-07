 ECE 662 Project 2 — Orthogonal Low-Rank Transformation (OLT)                                                                                                     
                                                                                                                                                                  
 Context                                                                                                                                                          
                                                                                                                                                                  
 Complete Purdue ECE 66200 Project 2 on OLT: implement the nuclear-norm-based transformation with CCCP optimization, evaluate on toy, face, and MNIST data,       
 extend to a neural-network regularizer, and produce a LaTeX report + Beamer deck. Starter notebook provides only 2D/3D synthetic datasets; everything else       
 (solver, YaleB/MNIST pipelines, CNN training, report) must be built. Heavy training runs on the user's GPU server via a git push/pull workflow; LaTeX is         
 compiled by the user on Overleaf.
                                                                                                                                                                  
 Key formulation (used throughout):                                                                                                                               
 min_T  Σ_c ||T Y_c||_*  −  ||T Y||_*     s.t.  ||T||_2 = 1                                                                                                       
 Optimized by CCCP: linearize the concave −||TY||_* term via its subgradient U V^T from SVD of TY, then solve the remaining convex sum-of-nuclear-norms problem   
 for T (proximal/projected gradient), re-normalize to unit spectral norm, repeat.                                                                                 
                                                                                                                                                                  
 Answered design choices                                                                                                                                          
                                                                                                                                                                  
 - Part 2 second modality: Rotated MNIST (random per-sample rotation, e.g. ±45°).                                                                                 
 - YaleB: auto-downloaded by a script (Extended YaleB cropped, 38 subjects × ~64 illum).                                                                          
 - Part 3 Option A backbone: small CNN on MNIST (same arch as Part 2 Task 3), add OLT loss on penultimate features alongside cross-entropy.                       

 Repository layout to create                                                                                                                                      
                                                                                                                                                                  
 ECE662Project2/                                                                                                                                                  
 ├── data/                         # datasets (downloaded)
 ├── olt/
 │   ├── __init__.py
 │   ├── solver.py                 # CCCP OLT solver (numpy/torch)
 │   ├── losses.py                 # differentiable nuclear-norm OLT loss for NN training
 │   └── metrics.py                # clustering acc, NMI, ARI
 ├── scripts/
 │   ├── download_yaleb.py
 │   ├── part1_toy.py              # runs OLT on 2D/3D toy, exports figs
 │   ├── part1_yaleb_classify.py   # PCA → OLT vs LDA on YaleB
 │   ├── part1_yaleb_cluster.py    # OLT subspace clustering vs kNN/SSC
 │   ├── part2_mnist_modality.py   # per-modality & joint OLT on raw MNIST
 │   ├── part2_cnn_features.py     # extract frozen CNN feats + joint OLT
 │   ├── train_part2_cnn.py        # HEAVY — server: train 2-layer CNN on MNIST
 │   └── train_part3_olt_nn.py     # HEAVY — server: CE + λ·OLT loss CNN
 ├── results/                      # figures, metrics json, checkpoints
 ├── report/
 │   ├── report.tex                # LaTeX report
 │   └── slides.tex                # Beamer slides
 └── Project2_OLT_solved.ipynb     # completed Task 1 notebook with visuals

 Step-by-step execution plan

 Step 1 — Core OLT solver (local)

 Files: olt/solver.py, olt/losses.py, olt/metrics.py.
 - olt_cccp(Y, labels, d_out, n_iter, lr, tol) returning T.
   - Init T = top-d_out rows of identity (or PCA basis), then scale to ||T||_2=1.
   - Each CCCP iter:
       i. Z = T Y; SVD → G = U V^T (subgradient of concave term, chain-rule to T: G Y^T).
     ii. Inner convex solve: minimize Σ_c ||T Y_c||_* − trace(G Y^T T^T) by a few proximal-gradient steps on T, where the proximal step on Σ_c ||T Y_c||_* is
 handled by its own subgradient Σ_c U_c V_c^T Y_c^T.
     iii. Project: T ← T / σ_max(T).
   - Stop on small relative change of objective.
 - olt/losses.py: OLTLoss(features, labels, lam) — differentiable (torch SVD) loss Σ_c ||F_c||_* − λ ||F||_* for use inside SGD training (Part 3).
 - olt/metrics.py: clustering_accuracy (Hungarian), wrap sklearn NMI/ARI.
 - Sanity unit tests on 2D/3D toy directly.

 Step 2 — Part 1 Task 1 toy visualization (local)

 scripts/part1_toy.py + update Project2_OLT_solved.ipynb:
 - Run solver on provided 2D and 3D toy generators.
 - Plot before/after scatter, per-class top direction cosine matrix, singular-value bar charts.
 - Export PNGs to results/part1_toy/.

 Step 3 — Part 1 Task 2 YaleB classification (local, CPU OK)

 scripts/download_yaleb.py: fetch Extended YaleB cropped from a known mirror (e.g. http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip) into data/.
 scripts/part1_yaleb_classify.py:
 - Load 38 subjects, resize to 32×32, random 50/50 train/test split per subject.
 - PCA to d=150 (fit on train).
 - Train OLT solver on PCA-train features (d_out≈n_classes or less).
 - Baselines: LDA (sklearn), raw PCA+NN.
 - Classification rule: nearest-subspace — for each class build basis of transformed train features, assign test sample to class with smallest residual ||(I−P_c)
  T x||. Also report nearest-centroid for sanity.
 - Save accuracy table to results/part1_yaleb_classify.json.

 Step 4 — Part 1 Task 3 YaleB subspace clustering (local)

 scripts/part1_yaleb_cluster.py:
 - Start from k-NN-graph spectral clustering (or simple k-means on PCA features) to produce pseudo-labels.
 - Initialize OLT with those pseudo-labels; re-cluster transformed features; optionally one refinement round.
 - Report Clustering Acc / NMI / ARI against true labels for (a) baseline and (b) OLT.
 - Save to results/part1_yaleb_cluster.json.

 Step 5 — Part 2 Task 1 & 2 MNIST+Rotated (local)

 scripts/part2_mnist_modality.py:
 - Load MNIST (torchvision); build Rotated MNIST via per-sample torchvision.transforms.functional.rotate random angle ∈ [−45°,45°].
 - Flatten + PCA to 50 dims (fit on original train).
 - Task 1: learn T_orig, T_rot separately; report 2×2 accuracy matrix with nearest-subspace rule.
 - Task 2: learn T_joint optimizing
 Σ_c (||T Y_c^o||_* + ||T Y_c^r||_*) − ||T Y^o||_* − ||T Y^r||_*
 via same CCCP. Re-evaluate the 2×2 matrix; expect better off-diagonal.
 - Save to results/part2_modality.json + confusion figures.

 Step 6 — Part 2 Task 3: CNN features (server for training)

 Two files:
 - scripts/train_part2_cnn.py HEAVY — trains 2-layer CNN (conv-conv-fc-fc) on Original MNIST, saves results/part2_cnn.pt and baseline test accuracy on
 orig+rotated.
 - scripts/part2_cnn_features.py (local) — loads checkpoint, extracts penultimate features on both modalities, learns joint OLT (reuses Part 2 Task 2 solver),
 re-evaluates.

 Step 7 — Part 3 Option A: OLT-regularized CNN (server)

 scripts/train_part3_olt_nn.py HEAVY:
 - Same CNN as Part 2, loss = CE + λ · OLTLoss(penultimate_features, labels) with differentiable torch SVD.
 - Train baseline (λ=0) and OLT variant (λ tuned, e.g. 0.1–1.0) with identical seeds/schedule.
 - Report clean + rotated-MNIST test accuracy, feature-space rank/orthogonality diagnostics.
 - Save checkpoints & metrics JSON under results/part3/.

 Step 8 — Deliverables (local, no compilation)

 - report/report.tex — full write-up: formulation, CCCP derivation, Part 1/2/3 experiments with tables & figures, Part 3 Option A
 motivation/justification/results/conclusion.
 - report/slides.tex — Beamer, 10–12 slides.
 - I will output the .tex paths and the list of referenced PNG/PDF figures; user compiles on Overleaf.

 Server / heavy-training workflow

 For Steps 6 and 7, I will:
 1. Write the script(s).
 2. Give the user the exact git add / git commit / git push commands locally.
 3. Give the git pull + python scripts/train_*.py ... commands to run on the GPU server.
 4. Tell the user which artifacts to git add on the server (checkpoints in results/…pt, metrics JSON, feature tensors) to git pull back locally for analysis and
 report generation.
 5. Wait for confirmation before moving to the next heavy step.

 Critical files to read before editing

 - references/Project2_OLT.ipynb — cell layout, toy generators, plotting helpers (reuse).
 - references/OrthogonalLowrankEmbedding/pytorch_OLE/OLE.py — SVD subgradient pattern, numerical thresholding (reuse for olt/losses.py).
 - references/project2.pdf + presentation PDF — formulas and deliverable checklist.

 Verification strategy

 - Solver unit test: on 2D/3D toy, post-OLT top per-class directions should be near-orthogonal (|cos| < 0.1) and per-class rank ≈ 1; objective monotonically
 decreases across CCCP iters.
 - Part 1 YaleB: OLT test acc within a few % of LDA; document both.
 - Part 1 clustering: OLT should improve NMI/ARI over the k-NN/spectral baseline used for pseudo-labels.
 - Part 2: diagonal (same-modality) accuracies high for per-modality T; joint T should raise off-diagonal cross-modality accuracy.
 - Part 3: OLT-regularized CNN should match or beat baseline clean accuracy and show larger gain on rotated test set; log penultimate-feature per-class nuclear
 norm to confirm low-rank structure emerged.
 - Deliverables: every figure referenced in report.tex / slides.tex must exist under results/; list the exact paths to the user.

 Execution order (confirmations expected from user)

 1. Step 1 + Step 2 + Step 3 + Step 4 (local, Part 1 end-to-end) → share results.
 2. Step 5 (local, Part 2 Tasks 1–2) → share results.
 3. Step 6 script written → user runs on server → pull back → Step 6 analysis.
 4. Step 7 script written → user runs on server → pull back → Step 7 analysis.
 5. Step 8 (report + slides).