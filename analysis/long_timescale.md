# Long-timescale behavior across recordings

`long_timescale.py` combines completed `grid_occupancy` analyses into one
calendar timeline and an offline interactive dashboard. All recordings appear
together. It reads existing tracking-derived caches and does not run detection,
tracking, sleep classification, or contact extraction.

```bash
python analysis/long_timescale.py \
  --data-folder /bucket/ReiterU/Ants/basler \
  --dates 20260723 20260724 20260729 \
  --output /flash/ReiterU/ant_tmp/samuel-reiter/long_timescale \
  --workers 4 --headless
```

Dependencies are the existing grid analysis environment plus `plotly` (tested
with 6.9.0). On Deigo use a compute job and a writable flash output directory,
then copy the complete output folder to bucket from a login/transfer node.

Open **`index.html`** in a browser. The dashboard is entirely local: keep
`plotly.min.js` beside it. Select a colony, metric, and ant; click an individual
heatmap row; zoom calendar plots; restrict exploratory views to ants with
shared coverage across every recording. The phase selector changes the
standardized trajectory and drift panels. Whole-recording spatial maps always
show every tracked identity. The individual task panel follows one ant across all dates and links its saved figure. Saved PNGs and CSV/Parquet tables
are linked below the interactive plots.

Saved overview figures are separate for the left and right colonies, including
the task summaries, fixed-reference cluster comparisons, temporal occupancy
summaries, joint-cluster diagnostics, and trip-duration diagnostics. Filenames
include the colony side; individual-ant figures retain their existing names.

For cell-based work in IPython or an editor, the entry script has `# %%` cells:

```python
from analysis.long_timescale import run
from analysis.long_timescale_utils import Settings
tables, manifest = run(
    data_folder="/bucket/ReiterU/Ants/basler",
    dates=["20260723", "20260724", "20260729"],
    output="/flash/ReiterU/ant_tmp/samuel-reiter/long_timescale",
    settings=Settings(min_bin_coverage=0, min_phase_hours=0, min_shared_hours=6),
)
tables["slopes"].query("side == 'left' and metric == 'sleep' and included")
```

## Input selection and cache reuse

- Discover `block*` directories with `stitched/per_track/*.parquet` under each
  requested recording date. Skip blocks without finished tracking. Do not also
  include `continuous_stitched`, which would duplicate source blocks.
- Reject temporally overlapping windows. If a full block and a tracked window
  both exist, choose nonoverlapping inputs before analysis.
- Reuse a compatible existing combined-analysis bundle when available, or use
  `block_activity_utils.load_window` to load the existing clock/context caches.
  `--analysis-bundle PATH` selects a bundle explicitly. The earlier three-date
  comparison output is a valid bundle; none of its pairwise results is used.
- Validate source sizes and nanosecond modification times, annotation geometry,
  classifiers, frame rate, spatial scale, and clock settings. Missing or stale
  inputs raise an actionable error instead of silently rebuilding raw data.
- Resource presence uses `resource_presence_frames.parquet`; spatial maps use
  existing annotated-arena occupancy arrays. New aggregation results are cached
  only inside the new output folder. All identities are included; existing contexts and speed/sleep vectors are rebinned over every partial cycle. Resource caches for previously unselected ants are extended from finished tracks only when absent. Optional trip events and positions are reused as saved; trip analysis is never inferred as zero for an ant absent from its metadata.
- `--path-map OLD_ROOT=NEW_ROOT` can be repeated when the same shared files are
  mounted under different prefixes. The standard Deigo/Saion mounts have local
  defaults; remapping still requires matching file sizes and modification times.

## Figures

1. Observation coverage and contributing ants on the full calendar timeline.
2. Calendar-time speed, body/antenna motion, and sleep.
3. Calendar-time colony, outside, food, and water presence.
4. Clock-time profiles overlaying all recordings, separated by colony.
5. Light/dark means by actual calendar day.
6. Within-ant trajectories using common clock slots in every recording.
7. Individual timeline heatmaps for speed, sleep, outside, food, and water.
8. Individual deviations from the first recording across the entire series.
9. Sleep-bout length and censoring diagnostics.
10. Spatial occupancy maps for each recording and colony on shared grid scales.

The dashboard also includes a calendar-day actogram. It can export the current
interactive view as a PNG using Plotly's camera button.

## Interpretation

Calendar timestamps use original global frame offsets. Thus the available
`20260729/block01-w070-117` window is displayed on July 30–31, rather than
being shifted to July 29. Both July 23 blocks contribute to that recording's
clock profile, while retaining their distinct actual observation times.

Every metric uses its own coverage denominator. A zero requires observed
inactivity/absence; missing observations stay missing. Colony presence uses
the existing one-second majority state. Food and water percentages use unique
resource frames divided by detected position frames, and overlapping regions
of the same resource type are counted once. Presence is not proof of feeding.

The default display thresholds are **zero**: every observed value, partial day,
and boundary bin is included, even for ants omitted from upstream clustering.
The earlier complete-cycle selection is not reused as a calendar-time filter.
For example, the 0724 dataset contributes all 49 h 9 min of available tracking.
Daily task profiles weight observations by each metric’s observed exposure;
trip duration is weighted by trip count. Other overview means average observed
clock bins. No daily value extrapolates missing hours into a 24-hour budget.

The optional shared-clock view uses the same clock slots for an ant in **every
recording**, requiring six shared hours overall by default, or any shared
observations within a light/dark phase. This eligibility applies only to the
explicitly labeled shared-clock comparison; it never removes ants from the
calendar plots. The cohort is fixed across recordings for each metric/phase.
Recording-level clock profiles average repeated cycles within each ant first.

Per-ant slopes use actual elapsed days between duration-weighted recording
centers. They describe sparse observations, not a demonstrated linear process.
Bootstrap intervals resample ants within each colony; they do not provide
independent colony replication or establish a causal effect of age/time.

Sleep bouts use existing one-second classifier states. Runs touching missing
tracking or recording boundaries are flagged as censored; only runs bounded
by known awake states enter complete-bout duration summaries. Shorter bouts
can therefore reflect coverage changes, so read the censoring panel alongside
the duration panel. The original sleep classifier is unchanged.

`inventory.csv` retains the original selection flags for auditing. All tracked
identities enter the behavior summaries; unavailable metric observations remain
blank. Omission from an optional trip or shared-clock summary is not evidence
of death.

## Individual task-allocation analysis

`long_timescale_tasks.py` combines existing `grid_occupancy` optional trip
outputs with colony occupancy, activity, sleep, and food/water presence.
`long_timescale_task_plots.py` extends the single-recording colony-use versus
trip-investment plots and produces:

- An eight-panel calendar figure for **every** tracked identity in
  `individual_ants/`, also selectable interactively in the dashboard.
- Daily ant-by-behavior heatmaps, with a fixed identity order across metrics.
- Colony occupancy versus trip rate/duration paths through all observed days.
- Per-ant changes across all recordings on identical shared clock hours.
- Interactive daily profiles with separate lights-on/off views. Colors are
  within-ant, within-metric standardized values; hover shows the original units
  and observed hours. These colors are descriptive, not statistical tests.

Trip definitions are inherited from `trip_phenotyping_utils`: completed
colony–outside–colony trips lasting at least 30 s, 5 s colony anchors, at least
20% observed coverage, gap bridging up to 30 s, and 5 s border flicker cleanup.
These event definitions match the existing grid analysis. No minimum number of
trips or daily completeness filter is imposed on the combined summaries.

`trip_availability.csv` records which blocks have optional trip analyses and
how many ants were analyzed. Ants listed in the source trip metadata with no
events receive zero counts when observed. Ants omitted from that source
analysis have unknown trip metrics. `completed_trips.csv` preserves all saved
events with globally unique IDs and actual departure/return timestamps.
Trips are never joined across blocks or tracking gaps. Boundary-censored trips
are absent from these completed-trip caches, so completed-trip rates can be
sensitive to tracking continuity.

Departures are counted once in their departure bin; the full completed duration
is associated with that departure. Observed outside seconds within a completed
trip are allocated to their actual bins, including across midnight. Trip rate
is completed departures per observed position hour; trip investment is observed
seconds in completed trips divided by observed position seconds. Counts, rates,
and exposure are provided together. A high rate from little exposure is retained
and should be read alongside its coverage.

`task_daily.csv` retains actual calendar days and light/dark phases.
`task_standardized.csv` and `task_slopes.csv` additionally assess within-ant
change on common clock slots across all recordings. These can test whether
patterns survive clock matching, but cannot establish biological task categories,
causal aging, or changes during unrecorded days. Activity, colony use, and resource
presence are measured proxies for task allocation, not direct task labels.

## Trip-duration diagnostics

To investigate whether the same ants make longer excursions, run
`long_timescale_trip_diagnostics.py` on an existing combined output folder:

```bash
python analysis/long_timescale_trip_diagnostics.py \
  --analysis-folder /path/to/long_timescale_output \
  --output /writable/flash/path/trip_duration_diagnostics --workers 4
```

This reuses completed events, position exposure, and one-second sleep contexts.
It saves eight alternative figures and an interactive per-ant trip explorer:
mean/median/tail trajectories, matched departure-clock strata, whole light/dark
cycles, duration survival curves, trip-rate versus duration changes, coverage
and recording-edge sensitivity, complete nights within a recording, and sleep
during outside excursions. Primary views retain all saved events. Comparisons
and sensitivity subsets report their denominators separately.

Calendar activity figures now show each source block's tracked interval, daily
date labels, and a separate close-up of near-continuous recording handoffs.
Actual gaps are retained; recording summary centers are not recording edges.

## Fixed-reference occupancy cluster switching

`long_timescale_cluster_switching.py` uses saved first-reference cluster IDs to
classify later maps with KNN, separately for each colony. For this comparison,
the requested reference is July 23 **block02**, chronological window index 1:

```bash
python analysis/long_timescale_cluster_switching.py \
  --analysis-folder /path/to/long_timescale_output \
  --output /writable/flash/path/cluster_switching --reference-indices 1
```

The classifier uses the exact original square-root occupancy features and
Euclidean distance, with five inverse-distance-weighted neighbors. It preserves
the original reference labels, never fits later clusters, and never interprets
independently fitted cluster numbers as longitudinal changes. Histogram edges,
image-to-arena geometry, source fingerprints, and original labels are validated.
All later cached profiles are retained; only identities with an original label
can count as switches. Side plus tag ID defines identity.

The output includes transition counts, all-ant assignment heatmaps, reference
UMAP projections, neighbor support/distance diagnostics, actual ant/prototype
occupancy maps, and an offline interactive per-ant explorer with colony-use,
speed, trip-duration and trip-rate summaries from the same windows. KNN-weighted
projection into the reference-only UMAP is for display; classification and
novelty checks use the full feature space. Novel profiles still project inside
the reference map, so their distance flag must be considered.

`assignments.csv` retains all predictions. `neighbors.csv` lists their five
contributing reference ants, distances and weights. `reference.csv` and
`calibration.csv` report leave-one-ant-out label recovery. A supported switch
requires an unambiguous baseline, ≥80% later vote, agreement at k=3/5/10,
agreement after excluding the ant's own baseline, and distance within the
reference leave-one-out 95th percentile. Highlighted examples additionally have
≥40% detected frames and ≥1 detected hour at both endpoints. These descriptive
checks do not filter the primary tables or establish biological task identities.
Whole-block maps have unequal duration and clock coverage; switching alone
cannot distinguish task allocation from circadian occupancy changes.
`clock_matched_behavior.csv` provides a supplementary check using the existing
colony-use, speed and sleep bins: observed-frame weighting combines repeated
cycles, then shared clock slots receive equal weight in every compared block.
Matching is per ant/metric, without a coverage threshold. This does not refit or
clock-match the KNN occupancy maps. Missing optional trip analyses remain NA.

## Four-hour clustering and continuity

`temporal_occupancy.py` extends the fixed-reference analysis to calendar-aligned
four-hour windows, plus one-hour detail. It uses the same 0723/block02 labels
and square-root features. A UMAP fitted only to those reference maps must
reproduce the saved coordinates to 1e-5 before its `transform` maps later data.
UMAP is a display: change sizes and novelty are calculated in the original
grid features, where a distorted embedding cannot create a false jump.

Whole-block histograms cannot be split retrospectively. First prepare per-ant
tasks and extract exact half-hour count caches from finished tracking:

```bash
python analysis/temporal_occupancy_tasks.py \
  --analysis-folder /path/to/long_timescale_output --tasks /run/tasks.json

# Slurm array: 0 through N-1, with %8 to run eight ants concurrently.
python analysis/temporal_occupancy_cache.py --tasks /run/tasks.json \
  --task-index "$SLURM_ARRAY_TASK_ID" --output /flash/temporal_clusters_4h

python analysis/temporal_occupancy.py \
  --analysis-folder /path/to/long_timescale_output \
  --tasks /run/tasks.json --output /flash/temporal_clusters_4h --tables-only

# Run eight rendering shards after the numerical stage succeeds.
python analysis/temporal_occupancy_render.py \
  --analysis-folder /path/to/long_timescale_output --output /flash/temporal_clusters_4h \
  --shard "$SLURM_ARRAY_TASK_ID" --shards 8

# Finalize reports after every rendering shard succeeds.
python analysis/temporal_occupancy_render.py \
  --analysis-folder /path/to/long_timescale_output --output /flash/temporal_clusters_4h \
  --finish --shards 8
```

Each task uses the existing `load_track_xy` frame averaging and arena grid
geometry. Its atom counts must sum **exactly** to the saved whole-block map.
No tracking, pose inference, sleep classification or trip extraction is rerun.
Four-hour windows are aligned at 00:00, 04:00, etc. Source contributions across
near-continuous block handoffs are pooled; real gaps and partial windows remain.
All observed maps are retained, including those without original cluster labels.

Leiden is fitted independently in every four-hour window. Its labels are local
identifiers; adjusted Rand index compares consecutive partitions without
assuming the labels match. Fixed KNN labels, a continuous axis between reference
prototypes, original-feature step distances, observation coverage, and colony
use/activity traces distinguish label-boundary crossings from larger changes.

Both the old whole-recording novelty cutoff and a duration-matched cutoff are
saved. The latter uses complete windows inside the reference, excluding each
ant's own baseline map. One-hour models compare circadian-only behavior, linear
drift, and a single step. Model comparison requires >=40% detection and >=95%
recorded hours, with at least 12 observations; these conditions do not filter
the displayed data. BIC is descriptive because hours are autocorrelated. An
unrecorded transition cannot be classified as gradual or sudden.

The offline `interactive.html` has a four-hour selector, fixed UMAP, population
novelty/partition continuity, one-hour traces and neighboring occupancy maps.
UMAP points can be colored by fixed reference labels or independent local clusters.
`individual_ants/` contains one detailed figure per reference ant; `filmstrips/`
covers the switching cases. `profiles_4h.csv`, `profiles_1h.csv`, `local_clusters.csv`,
`partition_continuity.csv`, and `change_models.csv` retain the underlying results.

Figure 07 follows the ants with original July 23 block02 cluster labels across
all recorded periods. Each period has a scatter plot of time inside the colony
(%) versus mean movement speed (mm/s), on shared linear scales. Colors stay fixed
to each ant's original cluster, even when its later assignment changes. Thin
lines below follow individual ants' period means; bold lines show the original
groups' medians, with observed counts. Missing behavior stays missing and breaks
lines; hollow scatter markers indicate less than 40% tracking coverage. The short
July 23 block01 observation predates the reference and is labeled accordingly.
Period widths are categorical, and their durations and clock coverage differ.
The plotted values and original labels are exported to
`07_original_cluster_behavior_by_period.csv`.

`temporal_occupancy_plots.save_overview_figures` can redraw these summaries using
saved tables and maps without refitting clusters or regenerating every ant figure.
The sharded renderer's `--finish` step also refreshes these overview figures.

For fast rerendering from existing result tables, fan out
`temporal_occupancy_render.py --shard 0..7 --shards 8`, passing the same
`--analysis-folder` and `--output`, followed by `--finish --shards 8` after those
jobs succeed. This reuses the saved classifications and maps. Deigo run scripts
and the per-ant task manifest for the July analysis are in
`~/ants_runs/temporal_clusters_0723_0724_0729_20260915`.

## Joint occupancy clusters across time

`temporal_occupancy_joint.py` fits one shared set of occupancy groups per colony
across all four-hour profiles. It uses **the full square-root probability grids**,
with one extra component for observed positions outside the arena. UMAP, PCA,
time, ant identity and behavior summaries are not clustering features.

KMeans observations are weighted by detected hours; partial and poorly detected
maps remain included. Candidate K=2–8 fits run independently as a 14-task Slurm
array, with eight concurrent tasks. Twenty-four bootstrap replicates resample
whole ants, keeping each ant's dates together. Label alignment uses in-bag ants;
out-of-bag agreement quantifies assignment stability. Selection compares full
feature silhouette, bootstrap stability and representation across multiple ants.
It favors the simplest supported partition within 0.02 of the best silhouette.
Both observation-weight and 1/2 mm spatial-smoothing sensitivity checks are saved.

```bash
python analysis/temporal_occupancy_joint.py \
  --temporal-folder /flash/temporal_clusters_4h \
  --output /flash/temporal_clusters_4h/joint_clusters \
  --task-index "$SLURM_ARRAY_TASK_ID" --bootstraps 24

# After all 14 candidates finish:
python analysis/temporal_occupancy_joint.py \
  --temporal-folder /flash/temporal_clusters_4h \
  --output /flash/temporal_clusters_4h/joint_clusters --finish
```

The resulting J0/J1/etc. labels have one meaning across dates **within a colony**.
This is retrospective clustering using all dates, distinct from the original
July 23 KNN reference assignments. `assignments.csv` retains both label sets.
`model_selection.csv`, `sensitivity.csv`, cluster maps and bootstrap counts expose
the model's limits. Low silhouette values caution against interpreting a stable
partition as discrete biological task states. All 2,964 temporal rows remain,
including 44 windows without spatial observations and therefore without labels.

The new `interactive.html` emphasizes colony presence, movement and trips rather
than embedding geometry. Saved timeline figures place clusters beside actual
behavior and tracking coverage; cluster-profile figures show the spatial maps
and behavior distributions that distinguish groups. The plotting helper
`temporal_occupancy_behavior_figures.py` also provides hourly transition close-ups.

## Validation

```bash
MPLBACKEND=Agg python -m pytest -q analysis/test_long_timescale.py analysis/test_long_timescale_tasks.py
MPLBACKEND=Agg python -m pytest -q analysis/test_long_timescale_cluster_switching.py analysis/test_temporal_occupancy.py
MPLBACKEND=Agg python -m pytest -q analysis/test_temporal_occupancy_joint.py
```

Tests cover global timestamp offsets, all-recording clock matching, side/tag
identity, phase coverage, equal-cycle/ant weighting, actual-day slopes,
overlapping resource annotations, unknown observations, and bout censoring.
