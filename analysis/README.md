# Analysis workflows

## Supported interactive workflow

[`grid_occupancy.py`](grid_occupancy.py) is the single routine interactive
analysis. Its `# %%` cells form one end-to-end workflow:

1. load normalized spatial occupancy and speed-vector outputs;
2. filter poorly observed tracks;
3. cluster ants by spatial position, separately for each colony side;
4. save `track_cluster_ids.csv`;
5. inspect cluster maps and example ants;
6. plot speed through time for each spatial job cluster with light/dark context;
7. plot mean sleep fraction by cluster, individual-ant sleep heatmaps, and
   activity/sleep matrices by time of day;
8. use `panorama_regions.csv` to compare colony-restricted versus in/out clusters;
9. plot return-aligned motion/contact rates and sleeping-recipient responses
   against matched no-contact times and other contacts; and
10. optionally summarize roaming ants from complete colony-outside-colony trips,
   including trip investment, trip and resource timing, inside/outside time,
   correlations between colony use and trip investment, and day-to-day
   consistency of outside time and trip frequency.

Edit `DATASET_ROOT` near the top, then run the cells in
order. The required per-track inputs are produced by
`compute_track_grid_occupancy.py` and `compute_track_speed_vector.py`.
`ANTS_DATASET_ROOT` can override the dataset path without editing the script.
To analyze combined blocks, pass the date folder with `--continuous`, or pass
the exact `continous_stitched` / `continuous_stitched` folder directly:

```bash
python analysis/grid_occupancy.py /bucket/ReiterU/Ants/basler/20260515 \
  --continuous --occupancy-only --headless
```

This saves the single-ant histogram, both colony UMAPs, cluster mean maps,
and example-ant maps in the combined folder's `analysis_outputs/`, alongside
cluster assignments in its grid cache. `--figure-root PATH` overrides the
figure destination. Omit `--occupancy-only` to continue into the existing
speed, sleep, and later analysis cells. In a notebook, point `DATASET_ROOT`
directly at the combined folder, or set `ANTS_CONTINUOUS=1` with a date folder.
If the combination contains multiple disjoint block groups, select the exact
group subfolder; ambiguous groups are reported instead of choosing one.

Combined inputs use their own tracks, caches and copied panorama annotations.
The start date, clock, FPS and full frame span come from `block_combination.json`,
preserving unrecorded gaps. Combined recordings reuse the selected occupancy
arrays directly, without rereading tracks or rebuilding per-ant histograms.
The existing single-block default still prepares annotated arena grids;
`--grid-workers N` controls that preparation.
Interaction-dependent sections report missing combined interactions, since
source-block interaction caches cannot be treated as one continuous cache.

The pipeline's `stitched/grid_occupancy_histograms` is selected automatically;
a unique legacy grid folder is also supported. Set `GRID_OUTPUT_NAME` or
`ANTS_GRID_OUTPUT_NAME` to explicitly select a different grid. Bin sizes are
read from metadata, not inferred from pilot-study folder names.
For single blocks, `USE_ANNOTATED_ARENA_BOUNDS=True`: draw two `arena` rectangles in
the panorama GUI first. The analysis rebuilds per-ant occupancy from finished
tracking into `stitched/grid_occupancy_histograms_arena_0p25mm`, using **0.25 mm
bins by default**, the calibrated scale, no padding, and the exact per-side
arena bounds. This also rebuilds finer grids when source caches use older 1 mm
bins. Other requested spacings get their own cache folder, leaving existing
grids and their saved analyses available.
`--grid-size-mm FLOAT` or `ANTS_GRID_SIZE_MM` overrides the default. An optional
block-local `grid_occupancy_settings.json` containing `{"grid_size_mm": 0.5}`
sets a persistent preference; command-line/environment values take precedence.
Preprocessing, pipeline submission and newly rebuilt combined grids also default
to 0.25 mm (`--grid_size_mm` / `GRID_OCCUPANCY_GRID_SIZE_MM` for the pipeline).
Existing combined recordings reuse their stored grids; a conflicting explicit
`--grid-size-mm` is reported rather than silently ignored.
These grids drive clustering and spatial summaries; source grids are untouched.
Each ant's cache is invalidated by changes to its arena geometry or tracking.
Out-of-arena detections are excluded from bins and reported in metadata;
normalization remains fraction of all detected frames, not renormalized to
hide exclusions. Spatial plot limits follow the exact annotated borders.
Set the option to `False` only to explicitly use legacy preprocessing bounds.

Panorama annotations save as `<block>/panorama_regions.json` and
`panorama_regions.csv`, so the date folder need not be writable.
Run `python tracking/gui/panorama_region_annotator.py --block-dir <block>`
from the repository root and press Save. Add `--copy-regions-to <other-block>`
to save identical JSON/CSV copies in another chunk folder on every Save;
repeat the option for additional blocks using the same calibration/layout.
The annotator selects the recording date's calibration when there is a unique
match; `--hmats` overrides it. A panorama made with a different calibration
must be regenerated before annotation, rather than silently reusing its image.
The source panorama, its coordinate metadata, and annotated preview stay in
the selected block. Analysis prefers block-local annotations, falling back to
date-level files only when local files are absent. Region-dependent cache
fingerprints use the selected CSV, so copied edits invalidate each block's
caches on its next analysis run.

Region names do not need `_L` or `_R`. For unsuffixed labels, analysis infers a
left/right divider at the midpoint of the horizontal gap between the two
arena rectangles (colony rectangles when arenas are absent), then assigns all
regions using their tracking-coordinate x centers. The resolved sides and
divider are displayed in `grid_occupancy`. Arena-edge overlaps up to 3 pixels
are reported and preserved, with their midpoint used for side labels; larger
overlaps must be corrected. Other ambiguous layouts require an explicit
`REGION_X_SPLIT_PX` instead of silently using a pilot dataset's grid boundary.
Arena-bounded grid rebuilding still requires two valid arena rectangles.

Known redundant exports are explicitly listed in `EXCLUDED_TRACK_NAMES` rather
than counted as additional ants; source files and preprocessing caches remain
untouched. May 15 block02 excludes the 10-hour `TrackID_0008_all_142044_right`
export, whose pose rows were verified against the full `142047` recording.
Sleep labels follow the exact filenames selected for spatial clustering.

The sleep and response cells incorporate figures 1, 2, 4 and 7 from
`exploratory/return_sleep_analysis.py`, using the same utility functions and
parameters. They use the current cluster assignments, cached `sleep_motion_labels`
and body/antenna motion, and the published `interactions` fanout (with support
for the older `interactions_skeleton_0p1mm` folder name).
The raw hit rule is minimum finished-skeleton distance <=0.1 mm; analysis bouts
merge hits separated by at most 2 seconds. Missing inputs are reported with
their paths and affected figures; incompatible or incomplete caches raise an
error, with no fallback to old interactions. Set `RUN_RETURN_SLEEP_ANALYSIS`
to `False` to disable the two response figures explicitly.

Figures in the interactive workflow use separate left- and right-colony windows,
including sleep heatmaps and the later foraging summaries. Speed plots use
recording clock time (`HH:MM`) with light/dark shading; the chronological axis
continues across midnight. The return-activity and recipient-response figures
(formerly plots 16 and 17) each use a **1 × 3** layout per colony. Recipient
responses display **−30 to +30 seconds**, controlled by
`RECIPIENT_RESPONSE_XLIM_SECONDS`; complete follow-up and contributing-ant counts
remain in the saved tables. Splitting figures changes subsequent plot numbers;
figure titles and filenames identify the colony and analysis.

The return/sleep figures auto-save under `FIGURE_ROOT`. Sleep tables go to
`GRID_ROOT/sleep_motion_analysis/`; return and recipient tables, exact matched
trigger frames, matching diagnostics and settings go to its `return_response/`
subfolder. Position/motion and contact-bout caches are shared with the standalone
probe. All excursion types are pooled, with a separate figure per colony. Figure 7
is a descriptive response including later contacts, not a causal effect or the
censored wake-risk analysis. The standalone probe retains the other diagnostic
figures, including matching balance and censoring.

The sleep/wake posture cell adds one two-panel density figure per colony in the
earlier exploratory style: bodypoint 0 is centered, 0 -> 1 points upward, and
numbered median body/antenna skeletons overlay a logarithmic density map. Both
states and colonies share physical bounds and color limits. States come only
from `sleep_motion_labels` (unknown excluded), not the older posture classifier.
Each ant contributes equal density mass within its colony/state; the two states
may have different contributing ants. Up to `POSTURE_FRAMES_PER_ANT_STATE=1000`
frames are sampled uniformly per ant/state. Missing or degenerate 0/1 axes are
excluded and counted in `sampling_coverage.csv`. Missing inputs are reported.
Coordinates retain their calibrated mm scale, without body-size normalization.

Pose samples, coverage, median landmarks and state summaries are saved under
`GRID_ROOT/sleep_motion_analysis/posture/`. Per-ant caches are invalidated by
changed label arrays, source tracks, calibration or sampling settings. Changing
`POSTURE_DENSITY_BINS` or `POSTURE_EXTENT_MM` only redraws the cached points.
`probability_in_view` reports clipped density mass; clipping does not rescale the
remaining mass. Set `RUN_SLEEP_POSTURE=False` to skip these plots.

The activity/sleep clock-matrix cell uses every spatially clustered ant, not
just foragers. Each colony gets activity (mean speed, mm/s) and sleep (% of
classified frames) matrices for each lights-on-to-lights-on cycle,
plus an observed clock profile. Rows keep the same occupancy-cluster/ant-ID order;
colors have common limits across cycles and colonies, with no per-ant scaling
or sorting by apparent rhythmicity. The x axis is hours since lights-on, with
clock times below and the light/dark transition marked. Defaults are 30-minute
bins and at least 50% valid/classified frames per bin. The default July 23
recording is only about 14 hours, so partial cycles are explicitly labeled,
unrecorded hours stay gray, and the clock profile requires one usable cycle
per bin. This cannot establish a 24-hour rhythm or its absence. For longer
recordings, the defaults automatically exclude partial cycles and require two
usable cycles per clock bin when at least two complete light-dark cycles exist
(one if only one complete cycle exists). `CLOCK_MATRIX_INCLUDE_PARTIAL_CYCLES`
and `CLOCK_MATRIX_MIN_CYCLES` can still be edited explicitly.
Unknowns remain gray, including
missing sleep caches; older sleep predictions are not substituted silently.
Tables and input/coverage auditing go to `GRID_ROOT/activity_sleep_clock/`;
figures use `FIGURE_ROOT`. These descriptive colony-only plots do not establish
an endogenous circadian rhythm or a difference from isolated ants without a
matched isolation recording.

The panorama-region section measures the fraction of detected time inside and
outside each colony annotation from the existing occupancy histograms. Exact
food/water detections are then extracted from raw tracks for the resource-time
heatmap.

The final trip-summary experiment is deliberately removable. Set
`RUN_OPTIONAL_TRIP_PHENOTYPING = False` to skip it, or delete the marked final
block and `trip_phenotyping_utils.py` to remove it completely. Its cached
tables are kept under `optional_trip_phenotyping/`; all figures use the editable
`FIGURE_ROOT` (by default, the dataset's `analysis_outputs/`). The summaries treat
variation among roaming ants as continuous; there is no second-stage trip or
resource clustering.

The final daily-repeatability cell adds ant-by-day heatmaps and matched-ant
scatterplots for adjacent full light–dark cycles, separately for each colony.
Each "day" runs from `LIGHT_ON_HOUR` to the next `LIGHT_ON_HOUR` (05:30–05:30
with the current settings), so it contains one uninterrupted light phase and
one uninterrupted dark phase. Dates label the cycle's start, not midnight.
Outside time includes all observed positions outside the colony; trip counts
use completed-trip departures, assigned to the cycle they start in. Outside
time is split at lights-on, even when a trip spans that boundary. Both measures
are expressed per 24 hours of observed position-bin exposure. This controls
exposure, but cannot recover trips missed in tracking gaps. By default, each
ant-day needs at least 70% coverage in both light and dark as well as overall;
partial first/last cycles are excluded and zero-trip cycles are retained. All
candidate roaming ants are included, without the three-trip summary filter.
Spearman correlations describe rank consistency, with paired-ant bootstrap
95% intervals; the identity line separately shows absolute agreement. Raw
daily counts, coverage, exclusions, and correlations are saved alongside the
other foraging tables. The first run caches daily position summaries; source
file, annotation, or light-schedule changes invalidate the relevant caches
automatically. Older midnight-based daily caches are also invalidated.

## Early versus late activity

[`exploratory/early_late_activity.py`](exploratory/early_late_activity.py) is a
standalone `# %%` analysis of the 20260810 `block02-w000-031` and
`block02-w149-197` windows. It reads the completed grid analyses; it does not
rerun tracking, interactions, or sleep classification. Missing or stale inputs
raise errors with the offending paths rather than silently skipping ants.

Workers are identified by **colony side + tag ID**, not timestamped filenames
or cluster numbers. Each worker's early and late activity is averaged over the
same fully recorded 30-minute clock slots. Defaults require 50% valid coverage
per slot in both windows and six hours of shared clock slots per metric.
This means supported clock time, not six uninterrupted observed hours; both
clock-slot support and effective valid hours are exported. Missing data are
never counted as stillness. Sensitivity tables use 50%/75% coverage and
3/6/9-hour support requirements.

Figures include the complete ID/coverage audit, paired locomotor/body/antenna
motion and sleep/outside-colony fractions, per-ID changes, paired clock
profiles, spatial-cluster overlap, and an editable individual-ant inspector.
Scatter points are clickable to show IDs. Body and antenna motion average
existing per-frame group percentiles, not all-point arithmetic means.
Cluster overlap is secondary and time-unmatched; cluster labels are arbitrary,
not named tasks. Bootstrap intervals resample paired ants and are descriptive,
not independent colony replication or evidence of causal task reassignment.

Run cells interactively, or from the repository root:

```bash
MPLBACKEND=Agg python analysis/exploratory/early_late_activity.py
```

Outputs default to the early block's `analysis_outputs/early_late_comparison/`
because the date-level directory may not be writable. PNGs, per-ant CSVs,
matched-bin Parquets, a parameter/source manifest, and `comparison_report.md`
are saved there. Subsequent runs reuse small fingerprinted comparison caches.
Override paths with `ANTS_EARLY_BLOCK`, `ANTS_LATE_BLOCK`, and
`ANTS_COMPARISON_OUTPUT`, or edit the script's parameter cell.

## Long-timescale behavior across recordings

Start with the compact [session handoff](README_long_timescale_session.md) for
the July 23/24/29 analyses, current results, output locations, and how to resume.

[`long_timescale.py`](long_timescale.py) extends the grid summaries to all
recordings together: calendar timelines, day/night rhythms, sleep, spatial
presence, individual task profiles, and shared-clock within-ant trends. It combines
existing completed-trip tables with colony occupancy, trip rates/durations,
resource presence, activity, and sleep. Every tracked identity and partial day
is retained; missing optional trip analyses stay unknown. It saves an offline
interactive dashboard, per-ant figures, and combined tables. See
[usage and interpretation](long_timescale.md).

## Exploratory work

[`exploratory/`](exploratory/) contains older, overlapping, or still-tuning
interactive analyses. The selected sleep/return plots above are available in
the routine workflow, but remain observational analyses rather than established
causal conclusions. These files remain available so ongoing
work and parameter choices are not lost, but they are not routine entry
points.

## Other top-level files

Top-level `compute_*`, `export_*`, classifier, GUI, and utility modules are
supporting preprocessing or tooling rather than additional interactive
analysis workflows. `commands.sh` records the per-track preprocessing fanout
commands.
