# 0724 spatial behavioral landscape

Run the single-recording analysis from the repository root:

```bash
python -m analysis.colony_behavioral_landscape \
  --block /bucket/ReiterU/Ants/basler/20260724/block01 \
  --cache /bucket/ReiterU/Ants/basler/20260724/block01/analysis_outputs/long_timescale_0723_0724_0729_20260915 \
  --output /path/to/new/output \
  --bootstrap 100 --seed 24
```

Use the tracking/analysis environment (NumPy, pandas, PyArrow, SciPy,
scikit-learn and Matplotlib). No UMAP, browser service or network is needed.
The CLI deliberately restricts the source to `20260724/block01` because the
report narrative is scoped to that recording.

The prior analysis folder supplies **raw half-hour counts and measured behavior
summaries only**. The requested source index is checked. Models, reference
coordinates and labels fitted on other dates are never loaded. Existing 0724
fine-grid labels are a descriptive comparison, not a training target.

Source requirements:

- Finished per-track parquet files and speed metadata.
- Published 1 mm and 0.25 mm annotated-arena occupancy caches and label CSVs.
- `temporal_clusters_4h/atoms/<source index>` from
  `temporal_occupancy_cache.py`, validated against their original inputs.
- `task_bins.parquet` and `run_manifest.json` from the prior measured-behavior
  analysis. Their source fingerprints must still match the published block.

All model fits give ants equal weight and keep colony sides separate.
Whole-recording selection reproduces the saved >40% speed-detection rule.
Probability maps include observed outside-arena mass; missing data are never
treated as a behavioral category. PCA and KMeans use square-root probabilities.
For computation, KMeans uses **all** PCA dimensions, retaining full distances.

Outputs include a seven-page PDF, separate PNG/PDF figures, an offline HTML
gallery and ant explorer, quantitative report, individual tables, fitted PCA
arrays, K=2–6 silhouettes and ant-bootstrap stability, representation
sensitivity, and a disjoint clock-matched first-day/second-day validation.
Temporal points require >=40% position detection and >=95% recorded exposure.
Figures compare pooled temporal correlations with ant-centered correlations
and within-ant time permutations. Uncertainty never treats frames as independent
replicates.

These are spatial behavioral summaries, not a complete behavioral state-space
reconstruction. A Gaussian-mixture BIC and a matched-covariance Gaussian
silhouette reference are descriptive checks, not biological null tests. A
continuous PCA subspace is a flexible reconstruction benchmark, not a
parameter-matched probability model. The report explicitly separates these
claims from evidence for discrete tasks or metastable states.

Validation:

```bash
python -m pytest -q analysis/test_colony_behavioral_landscape.py
```

On Deigo, write analysis outputs on `/flash`, then publish them from a login
node after `COMPLETE.json` appears; compute nodes cannot write to `/bucket`.
