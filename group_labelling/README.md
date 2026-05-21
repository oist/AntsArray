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
