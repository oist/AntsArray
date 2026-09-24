# Activity-only discovery for 20260724/block01

This analysis follows the posture-plus-velocity and hourly-motif analyses. It adds a UMAP of one-second histories to Figure 3 and compares ten individual activity representations with Gaussian mixtures and KMeans. The selected representation, K and assignments are frozen before opening existing spatial labels. Spatial outcomes are posthoc evaluation endpoints, never fit features or method-selection criteria. Because earlier spatial results were known, this search is exploratory.

Entry points:

- `activity_discovery_features.py`: full-speed summaries for every identity, preserving gaps, global clock alignment and alternating five-minute measurement splits.
- `activity_discovery.py`: hourly posture/motif/speed feature banks, training-only transforms, unsupervised model comparison, ant bootstrap, held-out-day evaluation, UMAP and posthoc spatial comparison.
- `activity_discovery_plots.py`: six figures (02–07), PDF, report, gallery and offline interactive explorer.
- `test_activity_discovery.py`: regression tests for gaps, clock splits, missingness, absence of selection leakage, and recovery of synthetic groups.

Install the repository's analysis dependencies (`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pyarrow`, `joblib`, `umap-learn`, `pytest`). The published `run_manifest.json` records the versions actually used. Run modules from the repository root. These modules are specific to the July 24 clock windows; do not silently apply them to other dates.

## Reproduction

Set paths to the block, frozen preceding dictionary, preceding hourly publication, run directory and writable cache. The dictionary includes `SPACE_BLIND_FROZEN.json`, `space_blind_models.npz` and its prediction/reconstruction tables. The hourly publication supplies the camera-occupancy diagnostic only; it does not select activity groups.

```bash
block=/bucket/ReiterU/Ants/basler/20260724/block01
dictionary="$block/analysis_outputs/postural_dynamics_velocity_20260924"
hourly_reference="$block/analysis_outputs/postural_dynamics_hourly_20260924"
run=/path/to/writable/activity_discovery
mkdir -p "$run"
python -m analysis.postural_dynamics_extract --block "$block" --prepare "$run" --all-tracks
```

Extract each task index in `tasks.json` (114 indices for this recording). These independent jobs can be submitted as a scheduler array. Existing validated all-track pose caches can be reused. The speed source is the existing stitched speed vector, not new inference.

```bash
python -m analysis.postural_dynamics_extract --tasks "$run/tasks.json" --task-index 0 --output "$run/pose_cache"
python -m analysis.activity_discovery_features --tasks "$run/tasks.json" --task-index 0 --block "$block" --output "$run/speed_cache"
```

After **all** tasks finish, execute the stages in order. `fit` writes `ACTIVITY_ONLY_FROZEN.json` with feature/model/assignment/ranking hashes. `spatial` verifies those hashes before opening the spatial outcomes. UMAP does not enter method selection and can run independently after `features`.

```bash
for stage in features fit umap spatial; do
  python -m analysis.activity_discovery \
    --tasks "$run/tasks.json" --pose-cache "$run/pose_cache" \
    --speed-cache "$run/speed_cache" --dictionary "$dictionary" \
    --block "$block" --output "$run/results" --stage "$stage"
done
python -m analysis.activity_discovery_plots \
  --output "$run/results" --block "$block" --dictionary "$dictionary" \
  --hourly-reference "$hourly_reference"
python -m pytest analysis/test_postural_dynamics.py analysis/test_postural_dynamics_hourly.py analysis/test_activity_discovery.py -q
```

Plotting also requires the existing spatial atoms and `task_bins.parquet` in `analysis_outputs/long_timescale_0723_0724_0729_20260915`, the existing fine-grid spatial labels, arena annotations and grid metadata. Atom provenance is verified against the source block and source stamps. These spatial inputs are read only after the activity fits have been frozen. Raw data and generated models/plots belong in the dataset publication, not Git.

The serialized activity model contains `ActivityTransform`. CLI stages import/define this class when loading it. For ad hoc loading of CLI-generated joblib files, first import `ActivityTransform` into the Python entry-point namespace; only load trusted model artifacts.

## Definitions and limits

Day 1 starts July 24 at 10:00 and lasts 24 hours; day 2 is clock matched. All 114 identities are audited. Posture eligibility requires ≥16 hours with ≥5 accepted clips. Speed eligibility requires ≥16 hours with ≥50% observed seconds and ≥3 valid five-minute blocks per hour. A measured second requires ≥18 frames, and a valid five-minute block requires ≥180 measured seconds. The primary method comparison uses the common cohort; four locomotor representations also use the broader speed-only cohort.

Feature families are saved in `feature_definitions.json`: concatenated hourly motif frequencies; hourly motif-frequency quantiles; hourly posture means/amplitudes; hourly short-time postural/velocity dynamics; full-speed levels; movement/rest-like runs and autocorrelation; hourly speed timing; and three combined representations. Quantiles summarize hourly measurements rather than pooling all clips across a day. The plotted typical speed reverses log scaling after aggregation: it is a shifted geometric summary of block mean speeds, not the arithmetic mean of all observed frames.

Full-speed inputs already interpolate gaps up to five frames, smooth by two frames and exclude >5 mm/s estimates. Missing intervals and five-minute boundaries truncate runs. Runs are censored observed durations, not sleep diagnoses. Velocity in the motif dictionary retains signed forward and lateral body-axis components.

Each representation is robustly scaled on training data, with median imputation and equal total weight per feature block, then reduced to at most three PCs. GMM compares K=1–4 and three covariance structures using BIC; nontrivial fits need ≥10 BIC improvement over K=1. KMeans needs silhouette ≥0.25 and chooses the smallest eligible K within 0.02 of the best silhouette. Both require ≥4 ants/group, median 100-bootstrap ARI ≥0.8 and ≥80% A/B assignment retention; otherwise K=1. Bootstraps refit preprocessing. GMM uses standard KMeans starts plus activity-PC quantile starts to avoid spurious local optima; no spatial labels initialize the models.

The shared primary method favors supported splits in both colonies and then higher mean A/B group-centroid prediction gain. This deliberately favors reproducible partitions and does not prove discrete biological types. Prediction targets differ across representations. Day 2 is excluded from selection. Spatial agreement is measured only afterward, every method is reported, and no spatial-based retuning occurs. Existing spatial labels use the recording including day 2, so agreement with them is not independent validation. Camera/coverage correlations and broader-cohort sensitivity are included as limitations. Confirmation needs another recording or colony with the procedure fixed.

UMAP uses 30 neighbors, min_dist 0.1, seed 724, Euclidean original 143-dimensional history features and equal 100-history sampling per common-cohort ant. Colors show the frozen 12-motif assignment in the original history space. UMAP is a visualization and does not reliably preserve density ([official documentation](https://umap-learn.readthedocs.io/en/latest/faq.html)).
