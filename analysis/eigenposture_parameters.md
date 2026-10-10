# Posture-PC/velocity parameter audit

This analysis asks whether the July 24 split is sensitive to time aggregation,
feature scaling, ant-profile compression, or Gaussian-mixture covariance. It uses
only learned landmark posture PCs, their within-clip derivatives, and signed/RMS
forward and lateral velocity. No hand-defined straightness measure is fitted.

Published results:
`20260724/block01/analysis_outputs/eigenposture_parameters_20261009/index.html`
under the basler bucket. Six figures are also available as
`parameter_audit_figures.pdf`. Earlier analyses remain intact.

## Inspect the analysis interactively

Open [`eigenposture_interactive.py`](eigenposture_interactive.py) in VS Code or
Spyder and run its numbered `# %%` cells in order. Alternatively, from the
repository root:

```bash
python -i analysis/eigenposture_interactive.py
```

Edit `DATA`, `SIDE`, `BIN_MINUTES`, `POSTURE_PCS`, `SCALING`, `COVARIANCE`, and
`BOOTSTRAPS` at the top. The default fits the right colony; set `SIDE = "left"`
for the other colony. Use 30 bootstraps for a preview or the default 500 for the
full stability calculation. Rerun the cells in order after changing parameters.

**The interactive version was revised on 2026-10-10:**

- Section 3 fits landmark PCA over the full 48 hours in both colonies. Daily
  coordinate moments are weighted by observed hours before averaging ants and
  colonies equally. PCA eligibility requires at least 24 observed posture hours
  across the full recording (41 left, 45 right ants for these inputs).
- Minute coordinate means are recovered from all 12 modes in the archived
  `full_rank` cache and projected onto the new basis. This reconstruction is
  complete up to cache rounding; the previous four-mode truncation is not used.
- PC-rate and RMS-velocity features are removed. Eligibility depends only on
  posture and velocity coverage. Bins contain sampled observations, not
  continuous clips. One clustering row is one ant, with six features and no
  extra SD features or concatenated time-bin dimensions.
- The six features are four posture-PC means and the **maximum signed forward
  and lateral velocities**. Posture retains mean within bins, then mean across
  observed bins. Velocity now uses maximum within valid bins, then maximum
  across bins: each ant's largest valid sampled-clip mean over the day. The cache
  stores clip means, so this is not an instantaneous or frame-by-frame maximum,
  nor a maximum of time-bin averages. Velocities retain the existing invertible
  `asinh(v / 0.1)` transform; posture scores remain linear. The velocity and
  posture families are scaled to equal total variance. This scaling preserves
  all six dimensions and the relative amplitudes of the posture PCs.
- Clustering directly uses all six balanced features with the default four PCs.
  There is no additional ant-profile PCA, whitening, or one-dimensional score.
  Gaussian-mixture initialization, fitting, BIC and stability checks all use the
  same full feature space. Two-feature plots are display views only.
- Day1 still fits feature balancing and the mixture. Day2 is a repeatability
  comparison using those fixed stages and the shared, full-recording posture
  basis; it is not held-out validation of the complete analysis.

This is a small **ant-level profile**, not a clustering of individual time bins
or a model of postural dynamics. Maximum means the largest **signed** value,
not the largest magnitude: a large negative lateral excursion does not become
a large positive peak. Maxima capture an observed peak, not its frequency, and
are sensitive to outliers and the number of observed clips. Posture means still
average over excursions. Increasing bin size uses the same sampled observations
and only affects velocity maxima through bin-coverage filtering. `binned`
contains velocity maxima and posture means for inspecting the time course.

With five-minute bins and the unchanged shared-covariance/BIC/stability rule,
the maximum-velocity version selects **K=1 in both colonies**. K=2 worsens BIC
relative to K=1 by 4.82 left and 9.46 right. Its median ant-bootstrap ARI is
0.809 left and 0.417 right (500 resamples; tenth percentiles -0.014 and 0.053).
The left K=3 fit improves BIC but fails temporal stability (median ARI 0.464);
left K=4 and right K=3/4 each contain a singleton. Maximum velocity therefore
does not improve the two-class result with the other choices held fixed.

The preceding mean-velocity version selected K=2 left and K=1 right at both
5- and 240-minute bins. Its left split agreed with 39/41 spatial assignments,
with median bootstrap ARI 0.902; the right five-minute K=2 fit had median ARI
0.668. Those figures describe the mean-velocity version, not the maximum version.

All K=1–4 fits and their diagnostics remain in `mixtures` and `k_table`; this
is a changed analysis, not a reproduction of the earlier one-axis separation.
The script labels single-group retention as trivial and does not present
majority-class spatial matching as recovery of two classes.

The file contains ordinary NumPy/scikit-learn calculations, no imports from the
analysis pipeline, and no automatic output-file writes. Inspect
`coordinate_modes`, `coordinate_means`, `minute`, `binned`, `profile_table`,
`state`, `balanced_table`, `k_table`, and `assignments`. The `state` dictionary
contains imputed data, centering, scales, and the full balanced matrix.

Inputs additionally include `full_rank/eigenposture_measurements.npz` and
`full_rank/landmark_pca.npz`, both present in the published original analysis.
The starting point remains the saved tracking/body-alignment/clip-QC cache;
upstream coordinate extraction is in
[`eigenposture_features.py`](eigenposture_features.py).

The batch commands and numerical results below describe the **earlier
2026-10-09 parameter audit**. They are retained as a record of that analysis and
are not the expected output of the revised interactive script.

## Result

The principal instability is the separate-variance mixture, not hourly binning
or an unstable raw-landmark PCA. The selected common setting is five-minute
means, then each channel's mean/SD over bins; equal total variance for velocity,
posture, and dynamics families; one ant-profile PC; and shared mixture variance.
Family scaling retains relative amplitudes among posture modes instead of
standardizing every small mode independently.

The selected K=2 partition has median bootstrap ARI 1.00 in both colonies
(500 resamples; tenth percentiles 0.902/0.909). All leave-one-ant-out refits preserve
all assignments. Spatial agreement is 39/41 left and 39/45 right, versus 39/41 and
41/45 for the preceding, less stable unequal-variance candidate. Stability and
spatial agreement are distinct objectives. BIC gain over K=1 is 2.82 left
(modest) and 9.57 right. These results do not establish discrete biological types.

Changing bins from 1 through 30 minutes preserves all selected labels; 60–240
minutes change one right label. Retaining only the first two landmark PCs,
transferring the other colony's frozen model, stricter coverage, and tighter
mixture convergence/varied regularization preserve the compared labels.
Next-day frozen retention is 39/41 left and 43/44 right; independent day2 refits
have ARI 0.810/0.820 against day1. Day2 had already been examined previously.

## Input and time support

Use the preceding `eigenposture_20261009` output as `--source`. Required files are:

- `eigenposture_measurements.npz`: 114 ants × 2880 minutes × 12 channels, plus
  original hourly profiles and feature names/families.
- `all_ant_coverage.csv`: the fixed eligibility cohort (41 left,45 right of57
  tracked identities per colony).
- `spatial_reference.csv`: loaded only after selection/model freezing.

The first day is July24 10:00–July25 10:00 JST; the second is the next24 hours.
Minute measurements summarize one sampled2.5-second clip per minute, not a
continuous minute. Twelve channels: forward/lateral means and RMS velocities;
four raw-coordinate posture-PC means; four RMS derivatives within sampled clips.
The coordinate basis is fixed in this audit (four modes,93.2% variance). Its
independent subspace stability was evaluated by the preceding analysis.

For a bin of b minutes, require at least ceil(b/12) valid clips per channel
(minimum one). For a disjoint half-time split, halve that count before rounding.
Bin means weight valid minutes equally; profiles weight observed bins equally.
Full-day eligibility is inherited from the prior >=12-hour rule, holding the
cohort fixed across parameters. Report missing ordered bins through the coverage
audit; impute them using training-ants' column medians, without missingness flags.
Signed velocities use asinh(v/0.1); RMS velocities use log1p(v/0.1); RMS posture
rates use log1p(rate). Posture PC positions remain linear. Mean/SD features are
computed in physical units before these transforms. All scales, imputation and
ant-profile PCA are fitted on training ants and frozen for new observations.

## Selection procedure

The screen has152 configurations: bin widths1,5,15,30,60,120,240; quartiles or
mean/SD summaries (plus time-ordered concatenation for bins>=15); individual
feature standardization or equal family weighting; one/two ant-profile PCs;
separate/shared covariance. Every configuration fits the same ants. GMM scores
are normalized by training PC1 SD; covariance regularization is0.001. Compare
random initializations and a sorted-score initialization; retain best converged
likelihood. Finite-column retention/imputation and scaling are training-only.

Screen24 bootstraps/configuration; compare independently fitted alternating
minute and alternating30-minute partitions. Rank the mean of the worst-colony
bootstrap median and two split ARIs, requiring group sizes>=5 and positive
K2-vs-K1 BIC improvement in both colonies. Validate top8 plus the7 original-profile
bin controls with200 bootstraps, leave-one-ant-out refits, and30 random disjoint
half-hour splits paired within each hour.

Final candidates require group sizes>=5, positive K2-vs-K1 BIC improvement,
bootstrap median>=0.8 and random time-split median>=0.6 in both colonies. Rank
mean of worst-colony bootstrap and time-split tenth-percentile ARIs; tie-break
bootstrap median then configuration key. `SELECTED_FROZEN.json` records the
choice before loading spatial labels or using day2 for evaluation.

For this fixed choice, examine K=1–4. Require positive BIC improvement overK1,
group sizes>=5, median bootstrap>=0.8 and median temporal split>=0.6; prefer the
smaller admissible K within2 BIC units of the best. K2 uses500 fresh bootstrap
resamples; K3/4 use100. Temporal splits alternate1,5,15,30,60,120-minute blocks.
Both colonies select K2. This is a conditional K check for the selected setting,
not a global test for multimodality over every possible feature representation.

The exact old hourly pipeline is separately audited with300 bootstrap samples,
with either refitted or fixed profile scaling/PCA and separate/shared mixture
variance. This diagnostic retains the original unnormalized scores,
regularization and initialization, so it reproduces its complete-data fit.
Screen curves use normalized scores and are not exact numerical reruns of it.

Screening and validation reuse the same biological data. New bootstrap seeds are
not new animals. The search is exploratory, and day2 was previously inspected.
Do not interpret internal selection scores as unbiased generalization estimates.
Spatial comparisons, bin-neighborhood checks, feature ablations, coverage and
cross-colony transfer are post-selection diagnostics; do not reselect on them.

## Reproduce

Run from the repository root with Python3.12, NumPy, pandas, SciPy, scikit-learn,
joblib and matplotlib. Exact runtime versions are archived in the publication's
`reproduction/environment.json`. Pytest is needed for tests only. Avoid BLAS
oversubscription with multiple worker processes:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
source_dir=/path/to/eigenposture_20261009
output_dir=/path/to/new_parameter_audit
occupancy_dir=/path/to/grid_occupancy_histograms_arena_0p25mm

python -m analysis.eigenposture_parameters screen --source "$source_dir" --output "$output_dir" --workers 4 --bootstrap 24
python -m analysis.eigenposture_parameters validate --source "$source_dir" --output "$output_dir" --workers 4 --bootstrap 200
python -m analysis.eigenposture_parameters finish --source "$source_dir" --output "$output_dir"
python -m analysis.eigenposture_parameters diagnose --source "$source_dir" --output "$output_dir"
python -m analysis.eigenposture_parameters controls --source "$source_dir" --output "$output_dir" --workers 4 --bootstrap 200
python -m analysis.eigenposture_parameter_report --source "$source_dir" --output "$output_dir" --occupancy "$occupancy_dir"
python -m pytest analysis/test_eigenposture_parameters.py analysis/test_eigenposture.py -q
```

A published copy contains portable inputs at `reproduction/input/` and occupancy
arrays at `reproduction/occupancy/`, usable as the respective source arguments.
Run the frozen archived code with `PYTHONPATH=/path/to/reproduction/code` when
checking against the original environment. Seeds, grid, source hash, and ranking
rules are saved in the two protocol JSONs. Models/assignments are hashed before
spatial evaluation. The per-ant file includes spatial labels only in
`assignments_evaluated.csv`; `assignments.csv` contains the pre-evaluation fit.

Tests verify hourly equivalence, disjoint temporal support, loss of burst
variability under coarse binning, family scaling, training-only transforms and
serialization, and rejection of unsupported K2 splits. The previous8 geometry,
PCA-weighting, QC, projection and K-selection tests also pass.
