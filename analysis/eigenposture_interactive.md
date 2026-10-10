# Right-colony behavior classes, then spatial and sleep analysis

Open `analysis/eigenposture_interactive.py` and run its ten `# %%` cells in
VS Code/Jupyter. It replaces the old five-minute K-means/transition comparison.
The classification logic is visible in this one script. Existing repository
utilities read pipeline caches and draw the established grid-occupancy figures.
No HTML, parameter search, additional PCA, or spatial reclustering is involved.

The input is the published measurement/model packet
`20260724/block01/analysis_outputs/fine_behavior_20261010`, plus the block's
stitched occupancy, speed, motion/sleep and interaction caches. This is an
analysis of measured data, not a raw-video extraction script. See
[fine_behavior.md](fine_behavior.md) for extraction and validation provenance.
Use the environment from that analysis or install
`analysis/eigenposture_leiden_requirements.txt` in the repository's pipeline
environment (the downstream readers also require PyArrow and Shapely).

For a noninteractive run, with PNGs, a PDF and inspectable CSV/NPZ tables:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
python -m analysis.eigenposture_task_report --output /path/to/results
```

The default block is July 24, block01. Edit `BLOCK`, `SOURCE`, and `OUTPUT` in
cell 1, or set `ANTS_DATASET_ROOT` and `ANTS_ROLE_OUTPUT`. The default interactive
output is `block01/analysis_outputs/right_behavior_classes_20261010`.
Run from the repository root. Qt figures are used when an interactive kernel
supports them; the report runner selects Agg.

## What to inspect

1. **Inputs:** all 57 right-colony identities remain in the coverage table.
2. **Features:** four full-recording posture PCs, head/gaster articulation,
   unsigned forward/lateral speed, both head-relative antennal speeds, turning,
   and three interaction measurements. Reconstruct log transforms and family
   scaling; assert agreement with the validated 16-dimensional feature space.
3. **States:** reuse the shared eight-state Leiden dictionary, as requested;
   assign right-colony windows by five-neighbor distance-weighted voting.
   Exact graph membership overrides voting for dictionary rows. UMAP only
   visualizes assignments; the feature heatmap explains their differences.
4. **Ant representation:** state proportions within each observed hour, then
   equal-weight averages across hours. Missing hours are excluded, never zero.
5. **Classes:** square-root proportions in all eight dimensions. Tied-covariance
   GMM K=1–4; BIC below K=1, minimum five ants/class, median bootstrap ARI >=0.8,
   median alternating-hour ARI >=0.6. Prefer smaller K within two BIC units of
   the best qualifying fit; fall back to K=1. No spatial labels enter this step.
6. **Inspect classes:** per-ant state fractions and a sampled-minute ethogram.
7. **Spatial comparison:** join the fixed behavioral classes to grid caches by
   side, ID **and filename**; require every eligible ant. Plot mean/example
   occupancy and nest use; compare to the independently saved spatial labels.
8. **Speed/sleep:** established full-recording curves with contributing-ant
   counts and individual sleep heatmaps, grouped by behavioral class.
9. **Clock profiles:** per-ant activity/sleep across complete light–dark cycles.
10. **Returns/social response:** the established return definitions and
    matched sleeping-recipient comparisons, now plotted separately by class.
    Unclassified right ants remain available as contact/return context.

The minimal continuation covers the established occupancy, speed, sleep,
clock-profile and return/recipient analyses. The separate experimental trip
phenotyping, activity lead/lag and spatial-contact model workflows are not
embedded here. `behavior_class_ids.csv` is the explicit handoff for those
workflows; its historical `leiden_cluster` column contains the **ant class**.
The existing spatial `track_cluster_ids.csv` is never overwritten.

## Choices and limits

The 2-second windows overlap every 0.25 seconds within one randomly sampled
60-second segment per ant per hour. Twenty valid windows in at least 24 of the
48 hours is an eligibility rule, **not 24 hours of continuously observed motion**.
The expected result is 41 eligible right ants, with classes of 24 and 17.
State S4 is marked with an asterisk because its ant-removal reproducibility was
poor. Eight states are a useful vocabulary, not a proven biological state count.
The longer wavelet representation lost fast, fragmented tracks; the validated
short-window kinematic representation is deliberately retained here.

States were learned from training ants in both colonies; only right-colony ants
enter the class fit and downstream plots. Temporal/class bootstrap tests condition
on that fixed dictionary and the existing full-recording posture basis. Occupancy
agreement is post hoc and is not a new held-out validation. Sleep labels are
motion-defined, not an independent arousal assay. Returns and matched contact
responses are descriptive associations. Gray plot regions indicate missing data.

Downstream plots use the full recording, matching `grid_occupancy.py`; the state
sampling/class fit uses its central 48 hours beginning July 24 at 10:00.
Comments beside each cell explain the choices and assertions check normalization,
state reproduction, fraction sums, GMM convergence, and the spatial identity join.
