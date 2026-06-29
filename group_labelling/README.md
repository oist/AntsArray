# group_labelling

Tooling for the SLEAP problem-frame rescue labelling rounds: pick frames worth
relabelling, package them for human labelers, and merge the corrections back.

Labeler-facing instructions live in
[SLEAP_GROUP_LABELLING_GUIDELINE.md](SLEAP_GROUP_LABELLING_GUIDELINE.md). The
rest of this file is for whoever is preparing a round.

## Three-stage flow

```
detection_pipeline outputs (*.slp, *_aruco_tracks.h5)
        │
        ▼
1. build_training_inventory.py   →  inventory_master.parquet
        │   per-frame metrics: n_aruco, n_sleap, n_matched,
        │   n_unmatched_*, mean_kp_score, min_pair_dist,
        │   mean_speed, n_low_score_inst
        ▼
2. select_training_frames.py     →  selected_frames.csv
        │   stratified greedy sampling per camera with a
        │   temporal min-gap (default 500 frames). Per-camera
        │   QUOTAS and STRATA_WEIGHTS at the top of the file.
        ▼
3. build_training_chunks.py      →  salvage_input.csv + chunks/*.pkg.slp
        │   wraps sleap_salvage_project.py: builds the master
        │   project and splits it into per-labeler packages
        │   with embedded frames.
        ▼
hand chunks/*.pkg.slp to labelers (see SLEAP_GROUP_LABELLING_GUIDELINE.md)
```

`sleap_salvage_project.py` is invoked by `build_training_chunks.py` via
same-directory lookup (`Path(__file__).with_name(...)`), so the two must stay
co-located.

## Typical run

```bash
# 1. Per-frame inventory of an experiment's predictions
python group_labelling/build_training_inventory.py \
    --data-dir /bucket/ReiterU/Ants/basler/20260520/block01/data \
    --out-dir  /bucket/ReiterU/Ants/training/block01/inventory

# 2. Sample frames per camera according to quotas + strata
python group_labelling/select_training_frames.py \
    --inventory /bucket/.../inventory/inventory_master.parquet \
    --out       /bucket/.../selected_frames.csv

# 3. Build per-labeler chunks (master .slp + split .pkg.slp packages)
python group_labelling/build_training_chunks.py \
    --selected-csv /bucket/.../selected_frames.csv \
    --data-dir     /bucket/ReiterU/Ants/basler/20260520/block01/data \
    --out-dir      /bucket/.../labelling_round_NN \
    --chunks       5 \
    --prefix       problem_frames
```

See the `--help` of each script for the full option list.

## Reviewing returned corrections

When labelers return their corrected `*.pkg.slp` chunks, union them into one
self-contained package so the whole round can be checked from a single source:

```bash
python group_labelling/sleap_salvage_project.py combine \
    --corrected-dir /bucket/.../Group_labelling/<date> \
    --pattern       '*_chunk[0-9][0-9]_*.pkg.slp' \
    --out           /bucket/.../Group_labelling/<date>/<prefix>_ALL_corrected.pkg.slp
```

`combine` needs no master and overlays nothing (unlike `merge`): every labeled
frame from every input is carried over verbatim and its embedded images are
re-embedded into the output. The `--pattern` above selects only the
labeler-corrected chunks (which carry a name suffix after `chunkNN`) and skips
the bare originals.

Every labeled frame is also added as a SLEAP suggestion (disable with
`--no-suggestions`) so the GUI's suggestion navigation steps through exactly the
annotated frames. SLEAP has no per-frame "author" field, so provenance is kept
three ways: each suggestion gets a `group` integer = its source chunk (the GUI
clusters annotated frames by who returned them); the integer -> filename legend
is written to `provenance["sleap_salvage_project"]["suggestion_groups"]` inside
the package; and `combine_report.csv` (beside the output, with
`combine_summary.json`) maps every `(video, frame_idx)` to its source file and
group.
