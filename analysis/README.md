# Analysis workflows

## Behavioral analysis

[`eigenposture_interactive.py`](eigenposture_interactive.py) is the supported
interactive analysis. Run its `# %%` cells in order in VS Code/Jupyter. It uses
both colonies to discover behavioral states, then classifies right-colony ants
from their state proportions. The same script compares classes in space,
speed, sleep, trips, and return contacts. S0 alone is the putative sleep state;
return figures can use fresh motion labels calibrated from S0. The current audit
retains eight provisional state labels, flags weak states, and selects two ant
classes (26/18) for better stability. State-merging and class-count evidence is
in `state_class_reduction_summary.pdf` alongside the full report.

[`eigenposture_interactive.md`](eigenposture_interactive.md) documents inputs,
feature definitions, model selection, parameters, and the current results.
Install the pinned dependencies with:

```bash
python -m pip install -r analysis/eigenposture_leiden_requirements.txt
```

Set `ANTS_DATASET_ROOT` to the block, `ANTS_BEHAVIOR_SOURCE` to the prepared model
packet, and optionally `ANTS_ROLE_OUTPUT` to the report destination. To run the
same cells headlessly:

```bash
ANTS_DATASET_ROOT=/path/to/block MPLBACKEND=Agg \
  python -m analysis.eigenposture_task_report \
  --source /path/to/model_packet --output /path/to/report
```

The report includes summary figures and `right_behavior_classes.pdf`, with a
text description for every figure. It reuses prepared inputs; it is not a
replacement for tracking or feature extraction.

## Retained preprocessing and shared code

| Files | Purpose |
| --- | --- |
| `full_behavior_extract.py` | Full-recording inventory, aligned full-body posture PCA, motion and contacts. |
| `full_behavior_states.py` | Window features, state dictionary fitting, and assignment. |
| `full_behavior_profiles.py`, `full_behavior_roles.py`, `full_behavior_sleep.py` | State summaries, ant classes, and sleep labels consumed by the interactive script. |
| `s0_motion_sleep.py` | Calibrate motion thresholds from S0 and fan out fresh per-track labels. |
| `compute_track_*.py`, `compute_sleep_motion_labels.py` | Per-track speed, occupancy, colony presence, motion, and optional sleep-label caches. |
| `grid_occupancy.py`, `arena_grid_utils.py` | Prepare independent spatial reference labels and grid QC. |
| `sleep_classifier*.py`, `sleep_label_gui.py`, `export_*sleep*.py` | Preserve sleep-label annotation, crop export, and supervised preprocessing tools. |
| `plot_stitched_xy.py`, `exploratory/tag_detection_*.py`, `exploratory/validate_aruco_lossless_camera.py` | Tracking and detection checks. |
| `egg_seg/`, `hits_to_csv.py`, `video_info.py` | Image preparation/training and input inspection/conversion. |

Shared grid, motion, sleep, return, trip, interaction, and plotting utilities
remain because the supported analysis or tracking review tools import them.
Their regression tests remain as well. Superseded exploratory analyses, older
behavioral-model variants, dashboards, and their standalone reports/tests have
been removed. Previously committed versions are available in Git history.

## Spatial reference preprocessing

`grid_occupancy.py` now stops after preparing `track_cluster_ids.csv` and spatial
QC figures. It does not run a second set of behavioral analyses. For a block:

```bash
python -m analysis.grid_occupancy /path/to/block --headless
```

The preparation uses speed metadata to filter ants with more than 40% detection
coverage, then fits spatial clusters separately by colony.
These labels are used only for comparison after behavioral classification;
they do not train behavioral states or ant classes. The behavioral state cohort
retains its separate >60% detection rule.

Single blocks use annotated arena bounds and 0.25 mm grid bins by default.
`--grid-size-mm`, `--grid-workers`, and `--figure-root` control preparation and
QC output. `ANTS_GRID_OUTPUT_NAME` selects an existing input cache. A block-local
`grid_occupancy_settings.json` can specify `{"grid_size_mm": 0.5}`.
Annotate with `tracking/gui/panorama_region_annotator.py`; region lookup prefers
block-local annotations, then date-level annotations, then an earlier recording.

For already combined inputs, pass the exact `continous_stitched` or
`continuous_stitched` directory, or a date folder with `--continuous`:

```bash
python -m analysis.grid_occupancy /path/to/date --continuous --headless
```

Combined recordings reuse their stored grids and `block_combination.json`
timing. Select an exact group folder if several disjoint combinations exist. A conflicting
explicit grid spacing is rejected instead of silently rebuilding combined data.
`--occupancy-only` remains accepted for existing pipeline callers.

## Tracking and multi-dataset combination

Tracking and preprocessing launchers are preserved, including
[`../scripts/analysis_after_stitch.sh`](../scripts/analysis_after_stitch.sh),
[`../scripts/per_track_slurm_fanout.sh`](../scripts/per_track_slurm_fanout.sh),
and `commands.sh`.

[`../tracking/colony/combine_blocks.py`](../tracking/colony/combine_blocks.py)
and [`../tracking/stitch_tracks.py`](../tracking/stitch_tracks.py) combine
recordings/chunks and preserve timing, track identities, and compatible caches.
See [block combination](../tracking/colony/block_combination.md) for usage.
Source recordings, model packets, cached results, and generated reports are
outside this script cleanup.
