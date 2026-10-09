# Direct velocity and posture analysis of July 24

This workflow asks whether physical measurements of individual ants recover the
existing grid-occupancy partition. It measures body-axis velocity, antennal/body
geometry and short-timescale angular motion. Spatial labels are introduced only
after fitting; they never select features, PCA loadings, mixture components or K
within a fit. The overall analysis is exploratory, not a preregistered discovery.

The final representation uses the dominant PCA component of 45 standardized
physical variables: three hourly quartiles for each of 15 measurements. PCA learns
the shared variation across velocity and posture. An unequal-variance Gaussian
mixture is fit separately to each colony. K=1 remains possible. A multi-component
fit must improve BIC and pass ant-bootstrap stability, split-minute consistency
and minimum-size requirements. Day 2 is reserved for persistence checks.

The published run is
`20260724/block01/analysis_outputs/velocity_posture_20261009/index.html`.
It contains six numbered figures, individual PNG/PDF files, a combined PDF,
`REPORT.md`, all-ant coverage, physical measurements, assignments, effect estimates,
camera controls, fitted models and provenance. Read the report for results and
limitations. Ants are the resampling units; the analysis does not count frames as
independent biological replicates.

## Run from existing physical measurements

Use the repository's Python environment (`pyproject.toml`). The numerical analysis
needs NumPy, pandas, SciPy, scikit-learn, joblib and matplotlib. Raw extraction also
needs pyarrow. The published `reproduction/environment.json` records exact versions.

From the repository root, set paths appropriate to your machine:

```bash
block=/bucket/ReiterU/Ants/basler/20260724/block01
measurements="$block/analysis_outputs/velocity_posture_20261009"
result=/path/to/new_velocity_posture_result

python -m analysis.velocity_posture_mixture \
  --source "$measurements" --output "$result" \
  --representation common --bootstrap 100

python -m analysis.velocity_posture_report \
  --output "$result" \
  --spatial-reference "$block/analysis_outputs/grid_occupancy_0p25mm_20260916/track_cluster_ids.csv" \
  --occupancy-root "$block/stitched/grid_occupancy_histograms_arena_0p25mm" \
  --pose-cache /path/to/pose_cache
```

`--source` needs `physical_measurements.npz`, `all_ant_coverage.csv` and
`measurement_manifest.json`. Input and output directories must be different.
`--pose-cache` is optional for reporting; it enables within-camera, within-speed
posture comparisons. The report otherwise still includes dominant-camera and
observation-coverage controls. Output input-file symlinks should be dereferenced
when publishing (`rsync -aL`), so the published directory is portable.

## Rebuild physical measurements from tracked skeletons

The extraction is intentionally scoped to the 114 tracked identities in July 24
block01. It samples one random 2.5-second clip per minute for two complete,
clock-matched days starting July 24 at 10:00 JST. The source has 4,247,196 frames at
24 fps and uses 0.016 mm per panorama pixel. Other datasets require their own
clock/calibration protocol rather than silently reusing these constants.

If the unfitted `posture_velocity_v2` cache already exists, skip raw extraction:

```bash
python -m analysis.velocity_posture_features \
  --pose-cache /path/to/pose_cache --output /path/to/physical_measurements
```

To recreate that cache from `stitched/per_track/*.parquet`:

```bash
python -m analysis.postural_dynamics_extract \
  --block "$block" --prepare /path/to/extraction_tasks --all-tracks

# Run task indices 0 through 113, preferably as an HPC array job.
python -m analysis.postural_dynamics_extract \
  --tasks /path/to/extraction_tasks/tasks.json --task-index 0 \
  --output /path/to/pose_cache
```

The raw extractor is already tracked in this repository. It uses tracked
landmarks and body-axis projected velocity; it does not fit or load the old motif
dictionary or old ant clusters. Cache metadata fingerprints every source parquet.
`velocity_posture_features` rejects changed source files and the wrong sampling
schedule. It preserves missing observations and rejects invalid geometry, camera
transitions and implausible motion at the local-window level.

## Outputs and validation

- `physical_measurements.npz`: 114 ants × 2,880 minute clips × 15 features;
  hourly means, per-feature observation counts, even/odd-minute replicates and
  camera counts. Array order is given by `ants` and `feature_names`.
- `all_ant_coverage.csv`: all 114 identities, per-day eligibility and observation
  counts. At least 12 hours per feature are required; day-2 quality does not select
  the training cohort. Each hour requires at least five valid minute clips.
- `ant_profiles.npz`: hourly q25, median and q75 profiles for training and day 2.
- `UNSUPERVISED_FROZEN.json`: model-input, assignment and model-file hashes written
  before spatial evaluation. `evaluate` verifies them before loading spatial labels.
- `k_selection.csv`: every K candidate, BIC, group sizes and repeatability.
- `assignments_with_spatial_comparison.csv`: all eligible ants, including every
  spatial disagreement; label permutation is for comparison only.
- `physical_group_differences.csv`: group medians, median differences with 95%
  ant-bootstrap intervals, and Cliff's delta (group 1 minus group 0).
- `posture_at_matched_speed_*.csv`: equal-ant comparisons within clip-speed bins.
- `shared_camera_speed_strata.csv`: descriptive comparisons with at least three
  ants per group in each camera × speed stratum. Strata are not independent ants.

The default `common` representation is the final analysis. The older `global`
and `block` options and the KMeans entry point remain for auditing exploratory
history; they are not required to reproduce the final fit. The earlier models
used different weights and a stronger BIC gate. Do not describe this refinement
as prospective blind validation.

```bash
python -m pytest analysis/test_velocity_posture_features.py \
  analysis/test_velocity_posture_analysis.py -q
```

Tests cover synthetic geometry and signed velocities, missing-data and camera
transitions, hourly observation thresholds, frozen preprocessing, an unequal-
variance synthetic mixture, K=1 fallback, spatial label permutation and ant-level
conditional summaries.
