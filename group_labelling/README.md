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

## Top-down retraining and instance-level review

A second, newer flow works on already-labelled chunks rather than raw predictions,
and targets the top-down centered-instance model. Its training unit is the
*instance* -- a 640px crop around one anchor -- so instances can be reviewed
individually, and unselected ants in the same frame simply become image context
carrying no supervision. Models are retrained incrementally: every review batch that
lands goes round the same loop.

```
topdown_common.py            every entry point imports this; nothing else is shared
        │
        ├── flatten_slp_videos.py       GUI re-save → loadable package
        ├── build_topdown_arms.py       camera-disjoint arms + frozen test set
        ├── make_topdown_configs.py     one warm-started sleap-nn config per arm
        ├── predict_gt_anchored.py      1 prediction per labelled instance   [GPU]
        ├── select_review_instances.py  the next review package + order CSV
        ├── extract_reviewed_batch.py   finished frames out of a part-review
        ├── score_node_error.py         per-node error + body-frame convention
        └── simulate_review_ch06.py     offline A/B — NOT part of the loop

submit_topdown_pipeline.sh   one driver, staged; runs all of the above
```

### The loop

```
1 flatten   raw/*.pkg.slp → flat/*.pkg.slp
              cuts the source_video chain; without it sleap_io.load_slp
              raises RecursionError and the file cannot be trained on at all
2 predict   flat/ → pred/*.pred.slp                            [GPU, SLURM]
3 select    → <name>.pkg.slp + .json + _order.csv              [login node]
              dense frames first, with SLEAP suggestions in priority order
4 (review happens in the SLEAP GUI)
5 extract   part-reviewed .slp → reviewed_batch.pkg.slp
6 arms      → arms.json   (--extra-arm folds the new batch in)
7 configs   → one warm-started .yaml per arm
8 train     one GPU job per arm                                [GPU, SLURM]
9 eval      GT-anchored inference → per-node error + convention  [GPU, SLURM]
```

```bash
bash group_labelling/submit_topdown_pipeline.sh --stages sync      # deploy first
bash group_labelling/submit_topdown_pipeline.sh --stages predict   # wait for the job
bash group_labelling/submit_topdown_pipeline.sh --stages select

REVIEWED_SLP=.../ch01to05_dense_n3000.slp \
ORIGINAL_SLP=.../ch01to05_dense_n3000.pkg.slp \
  bash group_labelling/submit_topdown_pipeline.sh --stages extract

EXTRA_ARM=rev_batch2=/work/.../reviewed_batch.pkg.slp \
  bash group_labelling/submit_topdown_pipeline.sh --stages arms,configs,train
```

Every stage is tunable by environment variable (`MODEL`, `CHUNKS`, `TOTAL_BUDGET`,
`MIN_FRAME_DENSITY`, `SELECT_MODE`, …); `--help` on the driver and on each script
lists the full set. `--dry-run` prints what would run and mutates nothing.

**`sync` is not optional on a fresh checkout.** Every entry point imports
`topdown_common` by module name, so `$SCRIPTS` must hold it alongside them; the
driver refuses to start a python stage otherwise rather than dying on `ImportError`
halfway through.

The selection JSON records camera, frame_idx, density bucket, per-node disagreement
and hidden-node counts for every selected instance, plus an effort estimate
tabulated over several seconds-per-keypoint assumptions, so a review budget can be
set from a measurement rather than a guess. Selection is deterministic: the same
arguments over the same inputs reproduce the same package.

### What lives in `topdown_common.py`, and why

Each of these was reimplemented in two or three scripts, and the copies drifted:

| primitive | the part that is easy to get wrong |
| --- | --- |
| `normalise` / `resolve_nodes` | node names are irregular **on purpose** — `'aruco '` has a trailing space, `'antenna L_2'` a space, `'antenna_L3'` no underscore. They are matched forgivingly and never rewritten: renaming a head channel silently breaks warm start. |
| `read_manifest` / `map_videos_to_cameras` | video → camera by exact labelled-frame set, never by position — order does not survive a GUI re-save. |
| `align_videos` vs `match_videos_by_overlap` | **two different relations, and conflating them once produced plausible nonsense.** Predictions are *contained* in their labels (they exist only where a label anchored them); two label versions merely *overlap* (a review may empty a frame or add one — Jaccard 0.990 on the real ch06 pair). |
| `match_frame` | Hungarian assignment on the anchor node, per frame, so a misalignment surfaces as unmatched instances instead of scoring one ant against another's prediction. |
| `classify_points` | truncation = near an edge **and** missing nodes. Position alone over-drops: 45% of near-edge instances carry a complete skeleton. |
| `compare_to_predictions` / `stats_of` | per-node error and PCK from GT-anchored predictions, so the centroid model contributes nothing to the score. |
| `body_frame_offsets` | along/perp offsets normalised by body length. **`perp` is the convention indicator**; `along` swings with posture by more than the effect being measured (6.6px sparse-vs-dense), so reading it chases biology instead of labelling drift. |

`test_topdown_common.py` covers all of these — 27 tests, nothing on disk. The
sleap-nn environment ships no pytest, so the file also runs standalone:

```bash
/apps/unit/ReiterU/sleap-nn/0.3.1/tools/sleap-nn/bin/python \
    group_labelling/test_topdown_common.py
```

### Traps worth knowing

* `--batch_size` in the predict stage counts **frames, not crops**. Every instance
  in a batched frame becomes its own 640px crop, so 16 frames of a 30-ant camera is
  450 crops and the decoder's bilinear upsample overflows `INT_MAX`. Keep
  `batch_size x max_instances_per_frame` under ~400 — the driver uses 4.
* `--peak_threshold 0.0` is mandatory for these models. At anything higher no peak
  clears the threshold and every frame comes back empty. The staged config must also
  have `data_pipeline_fw` reset to `torch_dataset` and `cache_img_path` to null,
  since the training cache lived on a compute node's `/scratch` and is long gone.
* Ground-truth anchoring means passing **only** the centered-instance model to
  `sleap-nn predict`. Adding a centroid model folds detection error into the number
  and breaks the 1:1 correspondence.
* `frame_idx` is the alignment key, never the video index. A Windows GUI save
  rewrites video paths to `Z:/...` (unreadable on Linux) and de-duplicates embedded
  videos by filename — 6 collapsing to 2 has been observed.
* Dense-frame selection uses a threshold of 16 instances/frame. That is a regime
  switch, not a gradient, and ranking by model disagreement is **not** a substitute:
  replayed offline it tracked the real corrections at r=0.03. It stays available as
  `--mode instance` for comparison and must not become the default again.
* Training batch size is settled — 6 for the centered instance, 8 for the centroid.
  The VRAM sweep that found them is retired; the largest batch that fits was never
  the fastest.
* Cluster: `/bucket` is read-only on compute nodes, `saion-gpu25` is excluded
  everywhere (it cannot resolve the login host, so jobs report COMPLETE while their
  output vanishes), and a largegpu slice is 16 CPUs + 128G per GPU.

The full record of what was measured, and on what, is in
`/bucket/ReiterU/Ants/SLEAP_files/Group_labelling/20260803_topdown/MODEL_COMPARISON_final_v2_vs_legacy.md`
(English: `..._EN.md`).

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
