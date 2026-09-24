# Hourly posture and body-velocity activity: 20260724/block01

The current analysis uses **24 concatenated hourly motif-frequency vectors**
per ant. The plotted labels follow the selected K: K=1 displays one unsplit
cohort. It never substitutes a forced two-group comparison. All 57 tracked
identities per colony appear in the availability audit.

## Run the hourly analysis

Use NumPy, pandas, SciPy, scikit-learn, matplotlib and pyarrow, plus pytest
for validation. The following pipeline keeps the preceding velocity run's
posture PCA, scaling and motif dictionary fixed. That dictionary was trained
on 52 day-1 ants; additional ants are scored using the same representation.
No old ant-group assignments are reused.

```bash
python -m analysis.postural_dynamics_extract \
  --block /bucket/ReiterU/Ants/basler/20260724/block01 \
  --prepare /path/to/hourly_run --all-tracks

# Run each of the 114 task indices. Existing matching velocity caches can
# be copied with their JSON metadata and timestamps preserved.
python -m analysis.postural_dynamics_extract \
  --tasks /path/to/hourly_run/tasks.json --task-index 0 \
  --output /path/to/pose_velocity_cache

python -m pytest analysis/test_postural_dynamics.py \
  analysis/test_postural_dynamics_hourly.py -q

python -m analysis.postural_dynamics_hourly \
  --tasks /path/to/hourly_run/tasks.json \
  --pose-cache /path/to/pose_velocity_cache \
  --dictionary /bucket/ReiterU/Ants/basler/20260724/block01/analysis_outputs/postural_dynamics_velocity_20260924 \
  --block /bucket/ReiterU/Ants/basler/20260724/block01 \
  --spatial-atoms /bucket/ReiterU/Ants/basler/20260724/block01/analysis_outputs/long_timescale_0723_0724_0729_20260915/temporal_clusters_4h/atoms/2 \
  --output /path/to/hourly_results
```

The dictionary input is a generated data artifact, like the tracking files.
Its hash and source run are recorded. To reproduce it from tracking data,
use the archived velocity procedure below; its ant partitions are not used
by the hourly pipeline.

An hour requires five accepted clips; an ant requires 16 measured hours
on day 1. This replaces the original >=240-clip daily cohort threshold and
removes the >40% detection prescreen. Sixteen of 24 hours guarantees every
pair has at least eight shared hours. All other quality rules are unchanged.
The thresholds are declared before clustering; they are not adjusted for
spatial agreement. The initial 24/28 ants came from 41/45 detection-qualified
identities, minus 17 per colony with insufficient accepted daily clips.

Square-root hourly probabilities are concatenated in clock order. Masked
KMeans uses the mean squared discrepancy over observed hours, with equal
weight per ant. Its center update accounts for different observed-hour
counts. A center hour without any contributing ant falls back to the
training colony's clock-hour mean. Missing hours are neither zero activity
nor interpolated posture. Hourly estimates with only five clips remain noisy.

Fit K=1–6. A K>=2 partition must have >=4 ants per group and median bootstrap
ARI >=0.8 over 100 whole-ant resamples. Select the smallest K within 0.02
of the best eligible silhouette, using distances on shared hours. Otherwise
select K=1. The displayed grouping, next-day assignments, maps and forecasts
all use exactly this selected partition. For K=1, retention and spatial-label
agreement are not informative and are marked not applicable.

Day 1 is July 24 10:00–July 25 10:00. Day 2 is the following 24 clock-matched
hours and does not select training ants. Test-hour loss masks unobserved
hours. Display/continuous-model PCA uses training clock-hour means for its
basis, with each ant projected on observed features; this filling does not
enter clustering. Controls include independently shuffled hourly order,
frozen alternative feature dictionaries, hourly camera usage and observed-hour
masks. Pairwise-distance correlations are descriptive, not significance tests
on supposedly independent ant pairs. The original camera-centering correction
was not fully stored and is not approximated in this revision.

Outputs include figures 02–07, `0724_hourly_postural_dynamics.pdf`, an offline
hourly explorer, `hourly_motif_frequencies.csv`, `hourly_motif_profiles.npz`,
all-ant quality and hourly coverage tables, selected groups, K-selection tables,
held-out validation, `HOURLY_FROZEN.json`, source/code hashes and software
versions. Missing hourly frequencies are NaN/null; sample counts are explicit.
The publication keeps the earlier results intact.

## Archived whole-day aggregation procedure

The following describes the preceding whole-day aggregation, retained to
reproduce its frozen motif dictionary and historical figures. Its forced
K=2 diagnostic view is superseded by the selected-K hourly workflow above.


The figures now start at step 2: eigenpostures and a sampled-posture density
heat map. Step 3 explains and illustrates motifs of joint posture and signed
body velocity. Steps 4–7 are refitted from those new motifs. Measurement QC
remains in tables and methods; it is not an opening figure.

This is an exploratory revision after the earlier posture-only analysis.
Models from that version are preserved in its published output and Git
commit `43c82c1`. New features and outcomes are labeled
`posture_velocity_v2`; old feature caches are rejected.

## Reproduce

The repository dependencies include NumPy, pandas, SciPy, scikit-learn,
matplotlib and pyarrow; use the test extra for pytest. Each run records the
exact software versions. The tracked `colony_behavioral_landscape.py` supplies
whole-ant bootstrap and Hellinger helpers.

```bash
python -m analysis.postural_dynamics_extract \
  --block /bucket/ReiterU/Ants/basler/20260724/block01 \
  --prepare /path/to/run

# Run every task index in tasks.json (86 detection-screened ants for 0724).
# Independent array tasks are safe; use a compute-writable output disk.
python -m analysis.postural_dynamics_extract \
  --tasks /path/to/run/tasks.json --task-index 0 \
  --output /path/to/pose_velocity_cache

python -m pytest analysis/test_postural_dynamics.py -q

python -m analysis.postural_dynamics \
  --tasks /path/to/run/tasks.json \
  --pose-cache /path/to/pose_velocity_cache \
  --block /bucket/ReiterU/Ants/basler/20260724/block01 \
  --spatial-atoms /bucket/ReiterU/Ants/basler/20260724/block01/analysis_outputs/long_timescale_0723_0724_0729_20260915/temporal_clusters_4h/atoms/2 \
  --output /path/to/results

# Rebuild figures from saved results without refitting numerical models.
python -m analysis.postural_dynamics_plots \
  --block /bucket/ReiterU/Ants/basler/20260724/block01 \
  --output /path/to/results
```

The implementation is scoped to this recording, 24 Hz and 0.016 mm/pixel.
Different recordings require explicit calibration/time-window parameters
and renewed validation. New outputs should use a new directory, such as
`analysis_outputs/postural_dynamics_velocity_20260924`, to preserve the
previous posture-only results.

## Features and motif construction

1. **Intrinsic posture.** The ten landmarks cover the tag anchor, head,
   petiole, gaster tip and antennae, with no leg landmarks. Express eight
   segment directions relative to the petiole-to-tag anterior axis. The 16
   direction cosines remove translation, global rotation and uniform scale.
   Four-frame causal smoothing precedes 12 Hz sampling. Learn posture-only
   PCA from balanced training ants and retain 95% variance.
2. **Signed body velocity.** Estimate the slope of TrackX/TrackY over the
   five most recent raw frames by least squares, convert to mm/s, and project
   onto the mean unit anterior axis in that window and its perpendicular
   `(-a_y, a_x)`. Forward velocity is positive toward the head. Lateral sign
   follows that defined axis, without assigning anatomical left/right from
   image coordinates. Store only the two projected velocities, not absolute
   positions or global headings. Velocity is physical mm/s, so it is not
   invariant to changing body size while retaining the same pixel calibration.
3. **Balanced metric.** Center using balanced day-1 training clips. Divide
   all retained posture coefficients by the square root of their total
   variance, preserving relative mode weights. Divide each velocity channel
   by its training SD times sqrt(2). Both feature blocks then contribute
   total variance one. Save exact centers/divisors in `feature_scaling.csv`.
4. **One-second histories.** Concatenate posture coefficients and the two
   velocity channels at each of 13 timestamps across one second. Flatten
   this time-by-feature matrix and divide by sqrt(13). Squared Euclidean
   distance is then mean feature discrepancy across the sampled times.
5. **Motif dictionary.** MiniBatchKMeans learns shared prototype histories
   from balanced training ants in both colonies. Compare 12/24/48 centers
   using future joint-state prediction on held-out hours; select the smallest
   count within one standard error of the best. Refit the basis, scaling and
   dictionary on day 1 only, then assign all histories to the nearest center.
   Motif examples are actual measured histories nearest their centers.
6. **Ant profiles.** Compute each ant's motif-frequency distribution. Fit
   ant groups in square-root frequency space separately within each colony.
   P0/P1 label order follows lower/higher postural angular motion, not space.

Motifs and prediction targets both include velocity. This changes the
interpretation from posture alone to posture plus locomotion. Ablations
refit motifs and groups on the same cohort with instantaneous joint state,
posture histories only, velocity histories only, mean-removed joint
histories, half/double velocity amplitude, and camera-centered joint
histories. Camera-only frequency profiles provide a nuisance comparator.
Weights are not selected by spatial recovery.

## Sampling, missing data and validation

One independently seeded random 2.5-second clip is requested per minute
for each ant, exactly matching the earlier sampling schedule. Day 1 begins
July 24 at 10:00 and lasts 24 hours; day 2 is the following matched 24-hour
period. Sampling uses the full recording clock in parquet metadata rather
than the ant's observed span in speed metadata.

The initial >40% detection rule uses each ant's first-to-last observed span,
matching the existing cohort. Fitting additionally requires >=240 accepted
day-1 clips in >=24 half-hours. Day-2 availability does not select training
ants. Missing landmarks or anchor positions, duplicate detections, unknown
or switching pose/position cameras and degenerate geometry reject clips.
Geometry bounds are 0.1–1.25 mm for the body axis and 0.05–2 mm for analyzed
segments. A broad jump guard rejects clips with estimated velocity magnitude
above 20 mm/s. No interpolation or missing-as-immobile rule is used. Finished
tracks lack per-node confidence, and residual tracking noise remains possible.

Posture and velocity are sampled at raw frames 4,6,...,58. History ends at
frame 52 and predicts the joint state at frame 58 (0.25 s later). Neither
filter reaches into future data; input and target filter windows do not
overlap. Every fourth day-1 hour is withheld for internal validation. Tests
check velocity signs, calibration, rotation/translation invariance, missing
anchors, future-frame exclusion, training-only scaling, balanced block
variance and sampling boundaries.

K=2 is a prespecified comparison, not an enforced biological conclusion.
Compare K=2–6 with 100 whole-ant bootstraps. Candidates need smallest group
>=4 and median bootstrap ARI >=0.8. Select the smallest K within 0.02 of the
best eligible silhouette; K=1 means none passed, not evidence of unimodality.
Compare discrete prototypes with continuous profile projections on day 2.

`SPACE_BLIND_FROZEN.json` records joint feature definitions, normalization,
dimension and model/group hashes before occupancy maps or prior spatial
labels are opened. Absolute location is excluded from fitting; its local
derivative, projected into body coordinates, is an explicitly allowed input.
Spatial forecasts use other group members' day-1 maps to predict an ant's
day-2 occupancy, excluding that ant from both the group and colony baseline.
Both days require >=40% position coverage. Resample whole ants for uncertainty
and permute whole-ant labels for the null. Camera centering can remove real
context-dependent behavior as well as imaging bias; it is not a causal fix.

## Outputs

Six PNG/PDF pairs (numbered 02–07), a combined PDF,
`0724_postural_dynamics_velocity.pdf`, `REPORT.md`, an offline gallery and an
interactive mode/motif/ant explorer. The explorer includes signed velocity
traces for the recalculated motifs. Tables, models, measured examples,
normalization parameters, source/code fingerprints and software versions
accompany the figures. A completion marker is written only after all outputs
succeed. Extraction caches and generated results are data artifacts; commit
the source, tests and documentation, and publish results alongside the data.

Methodological inspiration: [Stephens et al., 2008](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000028)
and [Costa, Ahamed, Jordan and Stephens](https://arxiv.org/abs/2105.12811).
The implementation does not claim to reproduce the full transfer-operator
method or identify maximally predictive states.
