# Session handoff: ant behavior across recordings

Updated 2026-09-15. Question: **do individual ants change their behavior/task
allocation over days, and are changes gradual, abrupt, or daily rhythms?**

Start with the **joint-cluster behavior explorer**. Results are completed and
published; resuming does not require detection/tracking reruns. Full usage is in
[long_timescale.md](long_timescale.md).

## Data and timing to keep straight

Times below are actual recording times, not inferred from folder names.

| Source | Tracked interval | Role |
|---|---|---|
| `20260723/block01` | Jul 23 11:42–14:57 | Included, short first recording |
| `20260723/block02` | Jul 23 19:31–Jul 24 09:29 | **Chosen fixed reference** for the original labels/UMAP |
| `20260724/block01` | Jul 24 09:31–Jul 26 10:40 | About **49 h**, all included |
| `20260729/block01-w070-117` | Jul 30 21:48–Jul 31 21:23 | Tracked subset; preserve its global frame offset |

- Identity is **colony side + tag ID**. There are 162 tracked identities in total;
  86 have original reference labels (41 left, 45 right).
- The July 23 afternoon gap, 14:57–19:31, is real. The July 24 recording handoff
  is only about **80 seconds**; it has consecutive 30-minute analysis bins.
- Main longitudinal displays retain partial days and all tracked observations;
  missing values stay missing. Optional comparison/model support flags do not
  remove primary rows. Misleading date labels and complete-cycle filters were fixed.
- July 23 contributes **456 ant × four-hour profiles**, including partial windows.

## Analyses tried

| Analysis | Purpose / method | Entry point and saved output |
|---|---|---|
| Block combination, initially tested on 0515 | Combine continuous finished tracks and compatible caches on a shared timeline, with per-ant Slurm fanout. Also enabled `grid_occupancy` on combined data. | [Block combination guide](../tracking/colony/block_combination.md); output spelling is **`continous_stitched`**. |
| Single-dataset grids and July comparison | Rerun/save the full grid analysis for available 0723/0724/0729 tracking; combine existing analysis tables for comparison. The July longitudinal work does **not** concatenate tracks across dates. | [grid_occupancy.py](grid_occupancy.py); Deigo run `grid_0723_0724_0729_20260915`. |
| All-recording behavior and individual task profiles | Calendar activity/sleep, colony/resource presence, day/night rhythms, per-ant daily changes, trips and shared-clock comparisons. | [long_timescale.py](long_timescale.py), [long_timescale_tasks.py](long_timescale_tasks.py); output root below. |
| Trip-duration follow-up | Test whether the same ants take longer excursions: means, medians, tails, clock matching, complete nights, coverage/edge sensitivity and sleep during trips. | [long_timescale_trip_diagnostics.py](long_timescale_trip_diagnostics.py); `trip_duration_diagnostics/`. |
| Whole-recording fixed-reference mapping | Five inverse-distance-weighted neighbors in square-root occupancy grids assign later maps to **0723/block02's original labels**. Check k=3/5/10, own-baseline exclusion and novelty. The initial UMAP display used neighbor-weighted reference coordinates. | [long_timescale_cluster_switching.py](long_timescale_cluster_switching.py); `cluster_switching/`. |
| Four-hour profiles and continuity | Exact finer spatial counts; actual fixed-reference **UMAP.transform**; KNN labels; independent Leiden clustering in each window; hourly trajectories and circadian/drift/step fits. Local Leiden IDs cannot be compared as temporal identities; partition continuity uses adjusted Rand index. | [temporal_occupancy.py](temporal_occupancy.py); `temporal_clusters_4h/`. |
| **Joint clustering — current primary view** | Fit shared groups across **all dates**, separately per colony, using exposure-weighted KMeans in **2,689 full occupancy features**. Compare K=2–8, bootstrap whole ants, check spatial smoothing/weight sensitivity, and show actual behavior beside assignments. | [temporal_occupancy_joint.py](temporal_occupancy_joint.py); `temporal_clusters_4h/joint_clusters/`. |

**Clustering never uses UMAP coordinates.** Joint clustering also excludes time,
identity and behavior summaries from its features; identity is used for bootstrap
resampling. J0/J1 are retrospective groups defined using all dates, not predictions
from July 23 and not interchangeable with the original labels.

**Reading old plot 5:** rows are reference ants, columns are four-hour windows;
the three panels are fixed-reference KNN assignment, novelty, and coverage.
The new joint timeline figures instead put assignments beside colony presence,
movement and tracking coverage, for all tracked ants.

## Findings worth carrying forward

- **Trips:** means increased in 12/13 left and 13/15 right ants with trips in every
  recording. Medians changed less; clock-matched/complete-night checks support
  lengthening. Trip caches cover only some ants and omit censored excursions;
  missing trip data are not zero.
- **left:036 is the strongest transition example:** around Jul 24 18:00–21:00,
  colony presence falls from nearly 100% to ~1%; joint clustering also supports
  the sustained switch. Detection at 18:00–19:00 is ~25%, limiting timing precision.
- **right:052:** behavior changes during the first hours observed on Jul 30.
  The first hour contains only 11 minutes. Its first four-hour assignment is
  ambiguous; J1 is clearer from Jul 31 00:00.
- **Novelty is broader than label switching:** the original 32/86 final whole-map
  outliers used a whole-recording reference threshold. With a duration-matched
  four-hour threshold, ~29% left / 30% right of well-observed Jul 31 profiles
  remain outside range. These percentages count profiles, not ants.
  Extreme examples `left:004`, `left:010`, `right:057` become nearly stationary;
  unusual occupancy need not mean a new task or normal sleep.
- **Joint clustering favors two broad groups per colony:** J0 is mostly
  colony-associated/lower movement; J1 spends more time outside and moves more.
  Median bootstrap ARI is ~0.97, but silhouette is only 0.167 left / 0.114 right.
  Extra groups are less stable; 1 mm smoothing changes membership (ARI 0.77 / 0.59).
  There are 168 adjacent-window label changes, 21 supported at both endpoints,
  and 13 also retaining the new label in the next window; these are events, not
  animals or proof of task reassignment.

## Where to resume

All July outputs are under this **Deigo** root:

```text
/bucket/ReiterU/Ants/basler/20260724/block01/analysis_outputs/long_timescale_0723_0724_0729_20260915
```

On Saion replace `/bucket/ReiterU` with `/home/sam-reiter/bucket/ReiterU`.

| Open / inspect | Relative to that root |
|---|---|
| Current simple ant explorer | `temporal_clusters_4h/joint_clusters/interactive.html` |
| Current saved figures + methods | `temporal_clusters_4h/joint_clusters/index.html` and `report.md` |
| Joint labels, groups, reliability and alternatives | Same folder: `assignments.parquet`, `cluster_summary.csv`, `model_selection.csv`, `sensitivity.csv`, `switch_events.csv` |
| Reference UMAP, hourly detail and novelty | `temporal_clusters_4h/index.html`, `interactive.html`, `individual_ants/`, `filmstrips/` |
| Trip follow-up | `trip_duration_diagnostics/index.html`, `individual_trip_explorer.html` |
| Broad behavior dashboard and source tables | Root `index.html`, `task_bins.parquet`, `task_daily.csv`, `completed_trips.csv`, `inventory.csv` |

The joint bundle retains **2,964 rows / 162 ants**: 2,920 nonempty maps assigned,
44 without a spatial assignment. See `run_manifest.json` and `publication.json`.

Deigo scripts, frozen code, environments and logs live under
`/home/s/samuel-reiter/ants_runs/` (Saion: `/home/sam-reiter/saionHome/ants_runs/`).
Run prefixes are `long_timescale`, `trip_duration_diagnostics`, `cluster_switching`,
`temporal_clusters`, and `joint_occupancy`, each suffixed `_0723_0724_0729_20260915`.

For reruns, follow [the commands](long_timescale.md). Reuse
`temporal_clusters_4h/atoms/`: exact half-hour counts reproduce all 456 original
maps and supply 26 four-hour / 93 hourly windows. Their initial extraction read
finished tracks because whole-block histograms cannot be split; subsequent
clustering/plotting reuse caches. `temporal_occupancy_render.py` fans out figures;
joint fitting uses 14 tasks, at most eight concurrent. Analyze on compute nodes,
stage on flash, and publish from a Deigo login node.

Validation: 6 reference + 5 temporal + 3 joint tests, exact map reconstruction,
nearest-centroid/source/timing checks, PNG and browser checks. Test commands are
in [long_timescale.md](long_timescale.md).

## Next useful questions

1. Separate repeated daily excursions from persistent changes using the same
   clock hours across days; inspect `left:036` and recurring right-colony switches.
2. Inspect video/tracks for the near-stationary novelty examples before giving
   them biological labels.
3. Decide whether a finer spatial scale or additional behavioral features gives
   reproducible distinctions before interpreting more clusters as more tasks.

Scripts currently include working-tree additions/changes on `main`; this handoff
does not create a commit or push. Linked method docs/manifests hold the details.
