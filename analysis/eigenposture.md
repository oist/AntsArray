# Antennal eigenpostures: July 24 block01

This analysis learns posture modes from six antennal landmark positions (12
coordinates), then groups ants using hourly distributions of those mode
amplitudes, short-timescale amplitude derivatives, and anterior/lateral velocity.
Predefined straightness, spreading, asymmetry and spatial occupancy are **not
model inputs**. Straightness is calculated only after fitting, to interpret the
learned geometry on held-out day 2.

The output is `20260724/block01/analysis_outputs/eigenposture_20261009`.
Start with `index.html`, `REPORT.md` and `eigenposture_figures.pdf`. The result is
exploratory. A candidate two-component mixture may resemble the spatial classes
without passing the stability criteria; `K=1` means no supported partition under
this workflow. Diagnostic K=2 labels must not replace selected assignments.

## Representation

- Reconstruct head-relative antenna coordinates from the existing **unfitted**
  segment-direction and segment-length cache. Alignment follows the anterior
  petiole-to-tag axis. Divide by one fixed day-1 median axis length per ant;
  do not normalize instantaneous antenna reach or remove an ant's mean posture.
- Apply four-frame causal coordinate averaging and local quality checks for
  camera transitions, duplicates and invalid geometry. Velocity uses the
  existing five-frame fit projected onto anterior and lateral body axes.
- Fit coordinate PCA on day 1, with equal colony/ant/hour/minute weights and
  equal valid-window weights within each minute. Retain 90% coordinate variance
  for the primary run. This is a multi-mode posture basis, not the later
  one-dimensional ant phenotype axis.
- Each clip contributes mean posture amplitudes, RMS amplitude derivatives,
  and signed mean/RMS anterior and lateral velocity. Hourly means require five
  observed minute clips. Training ants require twelve valid hours for posture,
  dynamics and velocity. Missing data remain missing.
- Each ant is represented by q25/median/q75 across its hourly measurements.
  Standardized profiles yield a shared phenotype PC1. Fit unequal-variance
  Gaussian mixtures for K=1–4 separately by colony. A multi-group fit must beat
  K=1 BIC, have median ant-bootstrap ARI ≥0.8 and independent even/odd-minute
  ARI ≥0.6, and at least max(4,10% of ants) in each group. Choose the smaller K
  within two BIC units of the best eligible solution; otherwise retain K=1.

The primary rule repeats the prior analysis for comparison; a BIC improvement
below two is still inconclusive evidence for the number of classes. One shared
phenotype axis is a modeling constraint, not a claim that all ant behavior is
one-dimensional. The bootstrap conditions on the learned posture basis; its
subspace stability is checked separately. Even/odd fits learn posture PCA
independently. Day-2 phenotype refits retain the day-1 posture basis.

## Reproduce

From the repository root, use the dependencies in `pyproject.toml`. Exact
versions and executed source snapshots accompany the published report.

```bash
block=/bucket/ReiterU/Ants/basler/20260724/block01
cache=/path/to/unfitted/pose_cache
result=/path/to/eigenposture_result

python -m analysis.eigenposture_features --pose-cache "$cache" --output "$result"
python -m analysis.eigenposture_analysis --stage fit --output "$result"
python -m analysis.eigenposture_report --output "$result" \
  --spatial-reference "$block/analysis_outputs/grid_occupancy_0p25mm_20260916/track_cluster_ids.csv" \
  --occupancy-root "$block/stitched/grid_occupancy_histograms_arena_0p25mm"
```

To recreate the unfitted cache, use `analysis.postural_dynamics_extract` with
`--all-tracks`; see [the raw extraction instructions](velocity_posture.md).
That extractor is shared, but the old handcrafted posture features and motif
models are not used here. This protocol is intentionally scoped to July 24
block01: 114 identities, 24 fps, and two complete clock-matched days beginning
July 24 10:00 JST. Other datasets require their own clock/calibration settings.

For the retained-variance sensitivity, reuse cached coordinate moments:

```bash
python -m analysis.eigenposture_features --source "$result" \
  --output "$result/rank95" --variance-target .95
python -m analysis.eigenposture_analysis --stage fit --output "$result/rank95"

python -m analysis.eigenposture_features --source "$result" \
  --output "$result/full_rank" --variance-target 1
python -m analysis.eigenposture_analysis --stage fit --output "$result/full_rank"

python -m analysis.eigenposture_analysis --stage diagnostics --output "$result" \
  --spatial-reference "$block/analysis_outputs/grid_occupancy_0p25mm_20260916/track_cluster_ids.csv"
```

These sensitivity settings were added after inspecting the primary 90% result.
They are reconstruction-based checks, not a prospectively specified validation.
The diagnostic stage freezes K=2 candidate labels before their spatial comparison
and never changes the selected model or assignment files.

## Data and tests

`eigenposture_measurements.npz` contains minute measurements, hourly means,
counts and independently learned even/odd-minute measurements. Array order is
given by `ants` and `feature_names`. `landmark_pca.npz` contains the learned mean,
full basis, eigenvalues and retained rank. `sampled_landmarks.npz` supports
post-fit interpretation; it is not a replacement for the exact weighted PCA
moments. `ant_coordinate_moments.npz` preserves those per-ant moments.

`UNSUPERVISED_FROZEN.json` fingerprints the learned basis, measured features,
profiles, models, K table and assignments before spatial evaluation.
`spatial_comparison.csv` retains every eligible ant and every disagreement.
`rank_sensitivity.csv` clearly separates selected K from diagnostic K=2 accuracy.

```bash
python -m pytest analysis/test_eigenposture.py -q
```

Tests cover coordinate reconstruction, preservation of extension, local
missingness/camera changes, equal-colony PCA weighting, derivative projection,
hourly coverage, portable frozen transforms, K=1 fallback and post-fit geometry.

This follows the learn-shape-first principle of
[Stephens et al. (2008)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000028).
Their eigenworms used tangent angles; these are landmark-coordinate modes.
This workflow does not infer behavioral attractors or equations of motion.
