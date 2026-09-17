# Combining recording blocks

This standalone workflow succeeds the former continuous-stitch stage embedded
in `submit_blocks_pipeline.sh`, which was removed from the block pipeline.

Run `combine_blocks.py` on a **Deigo login node** with a data folder containing
`blockNN/stitched/per_track/TrackID_*.parquet` files:

```bash
cd /home/s/samuel-reiter/AntsArray
PYTHONNOUSERSITE=1 /apps/unit/ReiterU/ant_tracking/venv/bin/python \
  tracking/colony/combine_blocks.py /bucket/ReiterU/Ants/basler/20260515
```

The command submits one Slurm worker per ant using
`scripts/per_track_slurm_fanout.sh`, followed by a validation job. A detached
login-side publisher waits for successful validation, copies the result from
flash using four disjoint transfer lists, verifies the transfer, and installs
`DATA_FOLDER/continous_stitched`.
The spelling `continous_stitched` is intentional. Add `--wait` to keep the
publisher attached to the calling shell.

Use `plan DATA_FOLDER` to inspect discovery and timing without writing outputs,
or `DATA_FOLDER --dry-run` to generate the frozen code, manifest, and worker
scripts without submitting jobs. Jobs default to 2 CPUs, 16 GB, and four hours
per ant; `--cpus`, `--mem`, `--time`, and `--partition` override these resources.
`--work-root` changes scratch placement; `--output` changes the destination.

## Discovery and timing

- Only numbered `blockNN` directories with readable, nonempty stitched per-ant
  parquets participate. Missing/untracked blocks break the sequence. Isolated
  tracked blocks are reported and skipped.
- One consecutive sequence writes directly into `continous_stitched`. Multiple
  sequences use separate children, for example
  `continous_stitched/block02_block03` and `continous_stitched/block05_block06`.
  They are never joined across an untracked block.
- Ant identity is `(colony side, TrackID)`. A tag used in both colonies produces
  two outputs. Ants observed in only some blocks remain in the result.
- Recording start times come from the timestamp used by the chunk stitcher,
  including its calendar date. A block can start days after the date-folder
  name. Tracks from older timestamped generations are excluded and recorded.
  Duplicate identities within the selected generation cause an error.
- `num_frames` parquet metadata supplies each recording's complete frame span,
  including undetected tails. FPS must agree across metadata; camera diagnostic
  JSON files provide a fallback for tracking-only blocks, and `--fps` supplies
  it when absent. Different frame rates or overlapping recording intervals
  cause an error instead of implicit resampling or duplicate frames.
- Global frames use `round((block_start - first_start) * fps) + local_frame`.
  This is a nominal-FPS clock, matching existing within-block tracking; it does
  not correct individual camera clock drift. Real gaps remain unknown.

If chunk files have been archived, source filenames inside a stitched parquet
can provide its start time. For recordings requiring an explicit timing
correction, place `block_combination_timing.json` in the block:

```json
{"start_datetime": "2026-05-18T08:59:22", "fps": 24.0}
```

## Tracks and caches

The usual `per_track`, `speed_vectors`, `sleep_motion`, `sleep_motion_labels`,
`sleep_predictions`, `colony_presence_vectors`, and `grid_occupancy_histograms`
layouts are retained. Track names use the first recording's start time.
Parquet columns are preserved, compatible Arrow types are promoted, and
`source_block` plus `block_frame` retain each row's provenance. Streaming is
implemented in `tracking/stitch_tracks.py` so a worker never loads the complete
multi-block pose table into memory.

Compatible temporal caches are reused:

- Speed and RF probabilities are concatenated with NaN for unobserved time.
- Presence, sleep, and RF classification vectors use `-1` for unknown time.
- Sparse bodypoint motion caches retain their frame indices and bodypoint
  columns; explicit frames receive the same block offsets as tracks.
- Prediction tables and sleep-bout endpoints receive those offsets too.
  Bout boundaries and within-block estimates are preserved. The combination
  does not infer motion, interpolate, or join sleep bouts across block edges.
- Counts, frame spans, summaries, identities, and output paths are updated.
  Incompatible temporal-cache parameters fail preflight. Missing cache families
  are explicitly reported; available segments retain unknown gaps.

When the blocks share panorama arena and colony annotations, presence and grid
occupancy are rebuilt from the combined coordinates using the existing
production calculators. This reconciles old caches that used different fixed
boundaries. Grid size defaults to 0.25 mm with 10 mm padding; use `--grid-size-mm`
and `--grid-pad-mm` to change it. The shared annotations and derived bounds are
included in the output. If annotations are unavailable, presence caches must
have matching parameters, and occupancy histograms must have identical edges
and coordinate origins; compatible histograms are weighted by detected frames.

Exploratory tables, manual labels, trained models, alternate grids, and
interaction caches keep their original block-specific meanings. They remain
accessible through relative `source_blocks/blockNN` links and are indexed in
the manifest. They are not presented as recomputed multi-block analyses. In
particular, legacy directed antenna/body interactions cannot be concatenated
with current undirected skeleton contacts as though they were the same measure.

## Provenance, monitoring, and retry

The printed run directory contains `plan.json`, frozen `code/`, `submission.json`,
`finalize.out`, `finalize.err`, and `publish.log`. Per-ant scripts, IDs, and logs
are under `<sequence>/tasks/jobs`. The manifest records input sizes/mtimes, code
hashes, selected and excluded tracks, timing, cache parameters, and geometry.
Inputs are checked again before and after workers and before publication.

Published results contain `combination_manifest.json`, `block_combination.json`,
`ant_combination_report.json`, per-family status files, and per-ant provenance
under `analysis_cache/block_combination/per_track`. `_SUCCESS.json` describes
the validated combined generation. `PUBLISHED.json` in the scratch run confirms
installation on bucket.

To retry failed workers in an existing run, use the frozen script and manifest:

```bash
/apps/unit/ReiterU/ant_tracking/venv/bin/python RUN/code/tracking/colony/combine_blocks.py \
  retry --manifest RUN/plan.json --wait
```

Retry refuses while ant jobs are active, retains completed ant outputs, and
resubmits unfinished workers and validation. For a transfer-only failure, run
`publish --manifest RUN/plan.json`. An output lock prevents competing runs from
publishing to the same destination. A fresh `--overwrite` run archives the old
combined folder only after its replacement passes validation. Source block
tracks and caches are never overwritten by this workflow.
