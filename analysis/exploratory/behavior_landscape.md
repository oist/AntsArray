# Occupancy versus a multidimensional behavioral landscape

`behavior_landscape.py` runs a separate, reproducible experiment on the 0724
recording. It reads completed `grid_occupancy` analyses and preserves the
canonical spatial cluster assignments.

```bash
/home/sam-reiter/miniforge3/envs/ants/bin/python \
  analysis/exploratory/behavior_landscape.py \
  --dataset /home/sam-reiter/bucket/ReiterU/Ants/basler/20260724/block01 \
  --output analysis_outputs/behavior_landscape_0724 --headless
```

Use an environment with NumPy, pandas, SciPy, matplotlib, scikit-learn,
pyarrow, igraph and leidenalg. The existing local `ants` environment contains
these dependencies. No UMAP embedding or additional model fitting service is
required.

The output contains `report.md`, five PNG figures, per-ant features and coverage,
all candidate cluster assignments and validation scores, continuous-neighborhood
forecasts, coverage sensitivity results, and a provenance manifest.

## Design

- Compare two contiguous, nonoverlapping 24-hour windows at identical clock
  times. Discard the remaining recording tail. Each colony is analyzed separately.
- Use four equal-weight feature families: square-root occupancy; activity level
  and sleep; within-ant daily timing; and undirected skeleton-contact behavior.
- Rebuild **time-split** occupancy from exact finished bodypoint-0 positions,
  using the original arena grid and detected-frame normalization. Require a
  reconstructed whole-recording histogram to match the saved full-frame grid.
  Cache the resulting small histograms per ant. Never give a full-recording
  histogram to a held-out forecast. A cached-position approximation is also
  audited, but predictions use exact frame-level histograms.
- Compare occupancy alone, individual added families, the full model, and
  tracking-coverage controls on identical ants. Missing observations are not
  inactivity. Preprocessing and feature selection use only the training window.
- Compare k-means at fixed k=2…6, then select among candidates using training
  separation and resampling stability. Require four ants per group; return one
  group when no candidate meets the criteria. Save a separate Leiden resolution
  sweep using the current grid workflow's graph construction.
- Predict other-day behavior using the mean of **other ants** in the same
  training-day cluster. Exclude the focal ant from its own forecast. Five-neighbor
  forecasts test continuous variation without requiring discrete groups.
- Forecast activity, timing, contacts, and space/resource use. The latter two
  fractions are not explicitly added as input features. Save the focal ant's
  own preceding-window behavior as a persistence reference.
- Report forward and reverse directions separately; paired-ant bootstrap
  intervals are conditional on the fitted models. Test outcomes never choose
  the number of clusters, feature weights, or preprocessing.
- Primary coverage is 70% per family per day and nine usable two-hour profile
  bins. A broader sensitivity cohort uses 50% and six profile bins. The strict
  left-colony cohort heavily underrepresents one original occupancy group.

The script requires the published `comparison_manifest.json` that identifies
the transferred context caches and their source fingerprints. It discovers
that manifest in the dataset's `analysis_outputs/`, or accepts
`--analysis-bundle PATH`. A manifest from another block, stale fingerprints,
changed annotations, an incomplete interaction run or an ambiguous contact
cache produces an error. Source data are read only. Repeated runs reuse extracted
features when their source fingerprints and extraction code still match.

## Interactive use

```python
from analysis.exploratory.behavior_landscape import run
from analysis.behavior_landscape_utils import ModelSettings

features, tables = run(
    output="analysis_outputs/behavior_landscape_0724",
    model_settings=ModelSettings(max_clusters=6, min_cluster_size=4),
)
tables["cluster_diagnostics"].query("side == 'right' and feature_set == 'full'")
tables["paired_gains"].query("direction == 'forward' and method == 'neighbors'")
```

Model settings live in `behavior_landscape_utils.py`; extraction and observation
rules live in `behavior_landscape_features.py`, with exact spatial extraction in
`behavior_landscape_spatial.py`. Do not select settings after
looking at the held-out scores and then call the same scores an independent
validation. The saved resolution sweep is exploratory.

## Interpretation

Additional predictive information and additional discrete behavioral types are
different conclusions. A richer continuous representation can improve forecasts
even if two or three clusters remain the best summary. A visually separated PCA
or UMAP cloud, a larger Leiden resolution or a higher training silhouette alone
does not demonstrate more reproducible types.

Contact rates use focal position availability and completed interaction
processing as exposure proxies. They still depend on skeleton/partner visibility
and local encounter opportunity. Partner diversity depends partly on sampling
effort. Bout presence includes the two-second merge tolerance and one-second
rounding; it is not an exact physical-contact duty cycle. Resource presence is
not evidence of feeding or drinking. Two clock-matched days do not establish an
endogenous circadian rhythm.

Tests:

```bash
/home/sam-reiter/miniforge3/envs/ants/bin/python -m unittest \
  analysis.test_behavior_landscape -v
```

The checks cover self-exclusion, missing outcomes, training-only scaling, equal
family weighting, a shuffled-future negative control, contact interval unions,
global-frame speed exposure, window separation, source remapping and training-only
model selection.
