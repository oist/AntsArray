# Focused 0724 activity landscape

`activity_landscape.py` and `activity_landscape_plots.py` present the frozen full-trajectory locomotor-summary / KMeans model from `activity_discovery.py`. The presentation uses a single method, shows the K=2–4 evidence, aligns independent spatial labels only for display, and numbers six figures from 1. Activity models, features and individual assignments are not refitted. The preceding method exploration remains archived separately.

1. Body-aligned eigenpostures and posture density.
2. One-second posture–velocity motifs in density-preserving UMAP.
3. K selection for the fixed 21-feature locomotor representation.
4. Identical activity coordinates colored by activity and spatial labels, with contingency tables and disagreements.
5. Day-2 occupancy and colony-region time of the activity groups.
6. Held-out-day persistence and physical activity interpretation.

The first two figures characterize posture at a short timescale. Ant groups use the full-speed summaries, not the posture motifs. Seven block-level measurements (mean/median/p90/p99 speed and fractions above 0.05, 0.2 and 1 mm/s) are aggregated hourly and summarized by three quantiles across hours. Robust scaling and three PCs precede KMeans. K=2 is retained because it has the highest silhouette in both colonies and meets silhouette ≥0.25 and the minimum-size, bootstrap and measurement-replication criteria. This is a useful partition, not proof of two intrinsic modes. K=1 is the fallback when no split qualifies.

## Reproduction

Run from the repository root using the analysis environment described in `activity_discovery.md`. The published exploration must have its frozen models, feature bank, cohort audit, candidate tests, UMAP inputs/old embedding and selected-model metrics.

```bash
block=/bucket/ReiterU/Ants/basler/20260724/block01
python -m analysis.activity_landscape \
  --source "$block/analysis_outputs/activity_discovery_20260925" \
  --output /path/to/writable/activity_landscape/results \
  --block "$block" \
  --dictionary "$block/analysis_outputs/postural_dynamics_velocity_20260924" \
  --stage all
python -m pytest analysis/test_activity_landscape.py -q
```

`--stage umap` prepares verified source copies and computes densMAP. `--stage render` then generates the report, six PNG/PDF figures, combined PDF, HTML gallery and offline explorer. The renderer reads the same validated spatial atoms, task bins and arena/grid metadata as the preceding analysis. Compute jobs should write to scratch; publish the completed directory from a node with bucket write access.

UMAP uses `densmap=True`, `n_neighbors=15`, `min_dist=0.01`, `dens_lambda=2`, `dens_frac=0.3`, `n_epochs=700`, Euclidean metric and seed 724. All 6,600 histories and original 143 features are retained, with 100 histories per common-cohort ant. No class labels supervise the embedding. The neighborhood-radius diagnostic compares RMS distances to the 15 nearest neighbors in original and embedded spaces, and reports their Spearman correlation. A fixed 1,500-history subsample also measures trustworthiness. Density preservation trades off some neighborhood fidelity; neither metric establishes the existence of discrete behavioral classes. See the [official densMAP documentation](https://umap-learn.readthedocs.io/en/latest/densmap_demo.html).

Spatial-class names/colors are aligned to activity groups by maximum overlap. This permutation changes only display names: it never changes the frozen activity assignments or coordinates. Every off-diagonal contingency entry remains visible. The same agreement counts are used in the PDF, report and explorer. `test_activity_landscape.py` checks this contract and the local-density diagnostic.

The report retains the cohort and confounding limitations: 30/36 of the 57/57 tracked ants meet both posture and speed criteria; day-2 retention uses 20/31 independently eligible ants; the left split fails the same rule on its broader 31-ant speed-only cohort; activity differences correlate with camera occupancy. All source hashes and parameters are stored with the results. Preserve the earlier publication when publishing this focused version.
