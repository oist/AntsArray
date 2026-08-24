# AntsArray detection_pipeline (v2)

Chunk-ordered ArUco + SLEAP detection/inference pipeline for PylonRecorder2 outputs.

Replaces the monolithic [transcode_sleap_aruco.sh](../transcode_sleap_aruco.sh).
Drops the re-encode stage (no longer needed: pylonrecorder2 emits clean
GOP-aligned `.mkv`/`.mp4`/`.avi` with sidecar diagnostics), switches SLEAP to
`sleap-nn/0.2.0` with optional TensorRT export, and schedules per-chunk
across all grid cameras (all `_000` before any `_001`) so early-time results
land in `<exp>/data/` first.

## Architecture

```
deigo-login (detection_pipeline/pipeline.sh)
  └── chunk_array (one task per grid video)
        └── chunk_finalize
              ├── backup (single datacp job; updates stable Backup archive)
              ├── aruco_array (cross-video, chunk-ordered, BATCH_SIZE chunks/task)
              │     │     ↳ each chunk: inline rsync h5 → bucket (via ssh deigo login)
              │     └── aruco_datacp (single safety-net job; idempotent rsync)
              ├── bridge (lazy TRT export check + ssh saion sbatch)
              │     ├── saion sleap_predict_array (largegpu; self-fetches via /deigo_flash)
              │     │     │     ↳ each chunk: inline rsync .slp → bucket (via ssh saion login)
              │     │     └── saion sleap_datacp (single safety-net job)
              │     └── saion cleanup (rm -rf /work)
              └── cleanup (polls bucket for all SLP files → rm -rf /flash)
```

Key design choices:

```bash
rsync -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw \
    /work/ReiterU/$USER/$EXP_NAME/output/*.slp \
    "$DATA_DIR/"
```


- **No deigo→saion chunk rsync.** Saion compute reads chunks directly from
  `/deigo_flash` (read-only cross-mount), `cp` to its own `/work` for isolation,
  then runs predict. Lets sleap start as soon as bridge exits (seconds, not hours).
- **Streaming bucket uploads.** Each array task ssh's its cluster's login
  immediately after producing an output file, so results appear in `<exp>/data/`
  as the work completes — not in one big batch at the end. The end-of-array
  `aruco_datacp` and `sleap_datacp` jobs remain as idempotent safety nets to
  catch any uploads that hit a transient SSH failure.
- **Single-job datacps** (one per leg) — keeps total queued jobs under the
  `AssocGrpSubmitJobsLimit` cap on both deigo and saion `datacp`.
- **Bucket is the cleanup sentinel.** Deigo cleanup polls `$DATA_DIR/*.slp` until
  every chunk has a result, then frees `/flash`. No cross-cluster Slurm deps needed.
- All cross-cluster SSH (TRT export trigger, saion sbatch, inline uploads) uses
  `ssh_retry` with 5 attempts + 10·n backoff — the lesson from block01's
  `kex_exchange_identification` reset wedging the whole pipeline.

## Layout

```
detection_pipeline/
  pipeline.sh                      # entry point (deigo-login)
  README.md                        # this file
  lib/
    backup_list.py                 # source-video + metadata list for Bucket Backup archives
    hosts.sh                       # SSH_CMD, ssh_retry, rsync_retry, host_resolves
    manifest.py                    # video discovery + sidecar/ffprobe cross-check
    perms.sh                       # best-effort chgrp/setgid helpers for shared outputs
    worklist.py                    # chunk-ordered (chunk_idx ASC, vname ASC) TSV builder
    pipeline_state.py              # data/PIPELINE_STATE.json: processing contract + wave ledger
  templates/
    backup.sbatch                  # update stable raw-video archive under /bucket/<unit>/Backup/<collection>
    chunk.sbatch                   # ffmpeg -c copy segment (no re-encode)
    chunk_finalize.sbatch          # build worklist + submit downstream sbatches
    aruco_array.sbatch             # run_aruco.py per chunk with --custom-dict
    aruco_datacp.sbatch            # datacp partition rsync flash→bucket
    bridge.sbatch                  # deigo→saion handoff, lazy TRT export
    sleap_predict_array.template.sh   # saion-side: sleap-nn predict (TRT/ONNX/PyTorch)
    sleap_datacp_array.template.sh    # saion-side: rsync /work→bucket via ssh saion
    cleanup.sbatch                 # rm -rf /flash, afterany
  scripts/
    export_sleap_trt.sh            # one-time TRT/ONNX export on saion-largegpu
    filter_done_aruco.py           # bucket-aware skip for the aruco leg
    filter_done_chunks.py          # bucket-aware skip for the sleap leg
```

## Quick start

From `deigo-login`:

```bash
bash detection_pipeline/pipeline.sh \
  --dir /bucket/ReiterU/Ants/basler/20260520/block02 \
  --sleap-model-centroid  /bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centroid \
  --sleap-model-instance  /bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centered_instance \
  --aruco-dict A \
  --chunk-sec 7200 \
  --sleap-runtime tensorrt
```

Monitor:

```bash
squeue -u $USER
ls /flash/ReiterU/$USER/jobs/<exp>/         # rendered sbatches + jid_*.txt + manifest.csv + worklist
ssh saion squeue -u $USER                   # saion side
```

Outputs land in `<exp>/data/`, per grid camera per chunk:

- `<vname>_NNN_aruco_tracks.h5`     dense `(frames, instances, 2)` arrays (deigo aruco array)
- `<vname>_NNN_aruco_detections.h5` DataFrame `(Frame, Instance, X, Y, Confidence)` (deigo aruco array)
- `<vname>_NNN.slp`                 SLEAP predictions (saion sleap predict)
- `<vname>_NNN_sleap_data.h5`       SLEAP DataFrame via `sleap2h5.py` (saion, inline post-process)

The colony tracking map stage consumes the `.h5` files (`_aruco_tracks.h5` / `_aruco_detections.h5`
and `_sleap_data.h5`); it does **not** read `.slp` directly.

## CLI

See `bash detection_pipeline/pipeline.sh --help` for the full option list. Notable
defaults:

| Flag                     | Default            | Notes                                                                                                 |
| ------------------------ | ------------------ | ----------------------------------------------------------------------------------------------------- |
| `--chunk-sec`          | `7200` (2 h)     | passed to `ffmpeg -segment_time`                                                                    |
| `--chunk-ext`          | `mkv`            | output container; sleap-nn 0.2 reads mkv via sleap-io                                                 |
| `--aruco-dict`         | `A`              | resolves to `custom_4x4_A100_d4_*.npz` (latest by name) under `/bucket/ReiterU/Ants/aruco_dicts/` |
| `--aruco-params`       | empty              | extra `run_aruco.py` detector flags copied from the curation GUI parameter test                       |
| `--sleap-runtime`      | `tensorrt`       | also `onnx`, `pytorch` (last = no export needed)                                                  |
| `--skip-trt-export`    | off                | fall back to `sleap-nn track` (raw model dirs, no export)                                           |
| `--saion-partition`    | `largegpu`       | A100 SM80                                                                                             |
| `--sleap-module`       | `sleap-nn/0.2.0` | saion module to `module load` for predict tasks                                                     |
| `--aruco-concurrency`  | `100`            | array `%N` cap; compute cpu cap 2000 / `-c 16` ≈ 125 max                                             |
| `--sleap-concurrency`  | `8`              | array `%N` cap                                                                                      |
| `--datacp-concurrency` | `4`              | array `%N` cap (deigo has 4 mover nodes)                                                            |
| `--group`              | `reiteruni`      | group owner for shared outputs; created dirs are chgrp'd and setgid where permitted                 |
| `--no-backup`          | off              | skip the automatic raw-video backup                                                                 |
| `--backup-root`        | `/bucket/<unit>/Backup/<collection>` | destination dir; `<collection>` = exp path minus date/block (e.g. `Ants_basler`) |
| `--backup-archive`     | `<date>_<block>_raw_videos.zip` | stable per-block archive filename; reruns update this same file                       |

## Auto-trigger colony tracking (optional)

Pass `--run-tracking` to chain [colony tracking](../tracking/colony/submit_blocks_pipeline.sh)
onto this block automatically. Because SLEAP runs on saion with no cross-cluster Slurm
dependency, detection cannot `afterok:` the sleap array; instead `pipeline.sh` starts a
login-side poller (`templates/track_trigger.sh`, `nohup`, survives logout) that watches
`<exp>/data/` until every expected `_aruco_tracks.h5` and `_sleap_data.h5` is present (the
expected count is the `aruco_worklist.txt` line count), then runs the tracking submit for
this one block:

```bash
bash detection_pipeline/pipeline.sh \
  --dir /bucket/ReiterU/Ants/basler/20260515/block02 \
  --sleap-model-centroid ... --sleap-model-instance ... \
  --run-tracking \
  --tracking-hmats /bucket/ReiterU/Ants/basler/cameraArray_calib/.../initial_H_mats.npz
```

Notes:

- **Gates on `_sleap_data.h5`, not `.slp`.** The map stage reads `_sleap_data.h5`; the inline
  `slp -> h5` conversion is best-effort, so a silent conversion failure (only `.slp` present)
  correctly counts as "not ready" instead of firing tracking on invisible SLEAP data.
- **Tracking needs the whole block, and refuses anything less.** `map_combine` groups whatever
  files exist per chunk and silently drops absent cameras, so a partial fire produces panoramas
  with holes rather than a shorter run — and produces them without an error. On deadline with a
  partial set the poller therefore *refuses* to submit and mails instead; override with
  `TRACKING_ALLOW_PARTIAL=1` only for a permanently dead camera.
- **The gate is the block's declared chunk total**, read from `data/PIPELINE_STATE.json`, not the
  worklist line count. Under `--chunk-range` the worklist is one wave, while the output counter
  globs all of `data/`, so gating on the worklist would fire on wave 2's first poll. This also
  means `--run-tracking` may be passed on any single wave: it waits for every wave. Pass it on
  one wave only — a second poller submits tracking twice.
- **Deadline is generous** (`--tracking-timeout`, default 48h). Raise it for multi-wave blocks:
  detection on a 98-hour block takes ~3 days, well past the default.
- **Cluster: deigo.** Tracking is CPU-only and reads `/bucket`; it runs on `compute` (the same
  login as detection). saion has no general CPU partition.
- **Conda-free by default.** Tracking jobs run under the unit `ant_tracking` venv
  (`submit_blocks_pipeline.sh` `DEFAULT_PYTHON_BIN`). Build it once (self-contained, no conda):

  ```bash
  module load uv
  # The uv module's default cache (/flash/ReiterU/tmp/uv-cache) is not group-writable;
  # override cache + managed-python install dir with paths you own. The standalone
  # python must live under the unit tree so the venv is self-contained (tracking jobs
  # run the venv python with NO `module load`, so its base interpreter cannot depend
  # on your home or a module's LD_LIBRARY_PATH).
  export UV_CACHE_DIR=/flash/ReiterU/$USER/uv-cache
  export UV_PYTHON_INSTALL_DIR=/apps/unit/ReiterU/ant_tracking/python
  mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
  uv venv --python 3.11 /apps/unit/ReiterU/ant_tracking/venv
  uv pip install --python /apps/unit/ReiterU/ant_tracking/venv/bin/python \
      numpy pandas pyarrow h5py tables opencv-python-headless tqdm
  ```

  Override per run with `--tracking-python-bin <path-to-venv>/bin/python`. `tables` (PyTables)
  is required — the map stage reads `_aruco_detections.h5` via `pd.read_hdf(key="detections")`.
- Progress log: `<exp>/hpc_logs/pipeline/track_trigger.log`.

Nothing is shared between `detection_pipeline/` and `tracking/` beyond invoking the tracking
entry script by path.

## Shared group permissions

The pipeline keeps bucket and scratch outputs group-shareable by default. Pass
`--group NAME` to change the target group; otherwise it uses `reiteruni`.
`pipeline.sh` exports this as `OUTPUT_GROUP` so deigo and saion jobs use the
same policy end to end.

Directory setup is best-effort: helpers in `lib/perms.sh` run `chgrp` and set
mode `2775` on pipeline-created directories so the setgid bit preserves group
inheritance. The startup preflight only warns if the experiment directory itself
is not group-owned, group-writable, or setgid, because that directory may contain
files owned by other users.

Upload stages also pass `rsync --chown=:$OUTPUT_GROUP` so `rsync -a` cannot
preserve a source-side group into the bucket by accident. Permission fixes never
abort a run if the current user cannot change a path they do not own.

## Bucket Backups

By default, normal pipeline runs submit one `datacp` backup job after chunking
finishes. The job updates a stable per-block archive grouped by collection under
the unit Backup folder, for example:

```bash
/bucket/ReiterU/Backup/Ants_basler/20260520_block02_raw_videos.zip
```

The `<collection>` subfolder (here `Ants_basler`) is the experiment's path under
the unit bucket with the trailing date/block stripped, so every basler block's
zip and `.txt` sidecar land together under `Backup/Ants_basler/` instead of flat
in `Backup/`. The archive contains the raw source videos listed in `manifest.csv` plus all
top-level `.txt` and `.json` metadata files in the experiment directory. A
matching `.txt` description file is written next to the archive with the
required `name:` and `project:` lines for OIST Bucket Backup.

Re-running the same block updates the same archive with `zip -0 -FS`; it does
not create timestamped duplicates. OIST's weekly Backup snapshots preserve older
versions remotely. Pass `--no-backup` for test runs where no Backup archive
should be updated.

## Processing contract (`data/PIPELINE_STATE.json`)

The first run of a block **declares** how it is processed; every later run must
agree or is refused before anything is submitted.

This exists because a chunk's filename encodes its *index*, not the settings that
produced it. `cam01_..._042` means "the 43rd chunk under this `--chunk-sec`", so
re-running a block at a different chunk length overwrites part of `data/` with
content that no longer lines up with the part it did not overwrite — and nothing
downstream can detect it afterwards, because every file stays individually valid
and the names are unchanged. The same applies to a model swap: detection counts
move ~4x between model generations, so a half-and-half block is quietly useless.

```jsonc
{
  "chunking":  { "chunk_sec": 1800, "chunk_ext": "mkv", "total_rows": 4950,
                 "videos": { "cam01_...": { "n_chunks": 198, "fps": 24.0, ... } } },
  "detection": { "aruco_dict": "...", "sleap_model_centroid": "...", ... },
  "waves":     [ { "wave": 1, "chunk_range": [0, 4], "rows": 125, ... } ]
}
```

| field group | role | on mismatch |
| ----------- | ---- | ----------- |
| `chunking`, `aruco_dict`, `aruco_params`, `sleap_model_*` | changes the *content* of an identically-named output | **refused** (exit 2) |
| `sleap_module`, `sleap_runtime`, `saion_partition` | changes how the work *ran* | warned, allowed |
| `videos[].n_chunks` / `frame_count` | the source recording was replaced or repaired | **refused** |

Keys the run does not exercise are not compared: an `--only-sleap` rerun supplies
no ArUco settings, and absence means "not exercised", never "clear it". A key the
contract never recorded is filled in rather than rejected.

To deliberately re-process a block under new settings, pass
`--new-processing-run`. It archives the contract to `PIPELINE_STATE.<utc>.json`;
it does **not** delete the old outputs — move them aside yourself first.

Blocks processed before contracts existed get one with:

```bash
python detection_pipeline/catalog.py state-init 20260716/block01 --chunk-sec 1800 --dry-run
```

`--chunk-sec` is required and cross-checked against the archived `manifest.csv`.
It is not inferred: guessing it is the weakness the contract removes.

## Wave processing (`--chunk-range`)

A 98-hour block at `--chunk-sec 1800` is 4,950 chunks — over `compute`'s 2,016
submit cap (Slurm counts each array task individually) and ~6 TiB held on
`/flash` until the run ends. `--chunk-range A-B` processes one window of chunk
indices instead of the whole block:

```bash
bash detection_pipeline/pipeline.sh --dir ... --chunk-sec 1800 --chunk-range 0-4    # wave 1
bash detection_pipeline/pipeline.sh --dir ... --chunk-sec 1800 --chunk-range 5-6    # wave 2
```

Waves slice the **worklist**, never the source videos. Because a chunk's index is
a pure function of `(video, --chunk-sec)`, a window names the same span of
wall-clock in every run, re-running a window overwrites only its own outputs, and
the block stays one contiguous experiment for tracking. Splitting the block
directory instead would restart chunk numbering per piece and break track
continuity at the seam.

- Ranges are clamped per video, so a shorter camera contributes fewer rows.
- `--chunk-sec` must be a whole number of GOPs. `chunk.sbatch` seeks with `-ss`
  and then verifies the first chunk's packet count against the expected frame
  cap, failing rather than emitting silently offset chunks.
- Every wave is appended to `waves[]`. Coverage is derived from what is actually
  in `data/`, never from the ledger — a killed job cannot leave the ledger
  claiming work that does not exist.
- `--run-tracking` fires when *this wave* completes. Pass it on the final wave only.

### Overlapping waves

Waves may overlap in time: wave *N+1* can be submitted while wave *N* is still on
the GPUs, so the queue never drains. What makes that safe is that each wave owns
its **control files** and shares only its **data files**:

| path | scope | holds |
| ---- | ----- | ----- |
| `/flash/.../jobs/<exp>/wave_<A>-<B>/` | per wave | `pipeline.env`, `aruco_worklist.txt`, `jid_*.txt`, rendered templates |
| `/work/.../<exp>/jobs/wave_<A>-<B>/` | per wave | uploaded worklist + rendered saion arrays |
| `/flash/.../<exp>/` | per block | chunks — filenames carry the index, and `cleanup` frees only its own wave's |
| `/work/.../<exp>/input`, `output` | per block | same; sharing `input/` is what lets `prefetch` skip an already-staged chunk |

The worklist is the reason for the split: `prefetch`, `sleap_predict`,
`sleap_datacp` and the verify gate all re-read it **at task start** and index it
by row. A second wave overwriting it under a shared path would hand a running
array a different wave's rows — every task would still report `COMPLETED`, with
the wrong chunks processed. For the same reason `saion_cleanup` deletes only the
chunks its own worklist names (mirroring `cleanup.sbatch` on `/flash`) instead of
the whole `/work` root; the root disappears when the last wave leaves.

Two runs that would land in the *same* jobs dir are refused before anything is
written — the guard reads that dir's `jid_*.txt` and checks both clusters' queues.
Override with `--force-submit`. A `--chunk-range` that overlaps a range already in
`waves[]` only warns: recomputing a window for a new model, or to rescue
half-landed chunks, has to stay possible.

Practical limits when overlapping: the GPU cap and `GrpSubmit` (2,016 array tasks)
are per user, so waves share them — divide `--sleap-concurrency` accordingly — and
`/flash` holds every live wave's chunks at once.

Inspect a block at any time:

```bash
python3 detection_pipeline/lib/pipeline_state.py show --data-dir <exp>/data
```

The catalog reports `expected_source`, `chunks_declared`, `waves_done` and
`unclaimed_chunks`, and flags `WAVE_GAP` when a window between two submitted
waves was never claimed. A trailing tail is normal progress and is not flagged.

## Re-runs skip work already on the bucket

Both legs consult `data/` before running, so a re-run only redoes the gaps:

| leg | filter | verified by |
| --- | ------ | ----------- |
| SLEAP | `scripts/filter_done_chunks.py` (in `bridge.sbatch`) | h5 `expected_frames` attr |
| ArUco | `scripts/filter_done_aruco.py` (in `chunk_finalize.sbatch`) | `aruco_tracks` dataset `shape[0]` |

Before the ArUco filter existed, its only skip was flash-local — so once
`cleanup` freed `/flash`, any later run recomputed the entire block's ArUco.

Both keep a row on any uncertainty: an unreadable bucket dir, a corrupt h5 or a
missing `h5py` all mean recompute. "Cannot verify" is never a verdict of "done".
Force a full redo with `--sleap-force-recompute` / `--aruco-force-recompute`.

## Phase isolation (for testing)

```bash
bash pipeline.sh --dir ... --only-chunk    # stop after chunk submission
bash pipeline.sh --dir ... --only-aruco    # skip bridge / saion
bash pipeline.sh --dir ... --only-sleap    # skip aruco array+datacp
```

## Pre-flight checks (run once before first real run)

1. **mkv readable by sleap-nn**: on saion-gpu24, `sleap-nn predict <export_dir> <some_chunk.mkv> --runtime tensorrt --n-frames 100` succeeds.
2. **TRT export of legacy `best_model.h5` models**: run
   `detection_pipeline/scripts/export_sleap_trt.sh --centroid <dir> --instance <dir> --out /tmp/exporttest --runtime tensorrt` and confirm `model.trt` is created. If
   the export errors on legacy weights, set `--skip-trt-export` and the
   `sleap_predict_array` task falls back to `sleap-nn track` (PyTorch).
3. **ArUco dict A npz schema**: on deigo,
   `python3 -c "import numpy as np; d=np.load('/bucket/ReiterU/Ants/aruco_dicts/custom_4x4_A100_d4_20260410_103938.npz'); print(list(d.keys()), d['bytesList'].shape, int(d['max_correction_bits']))"`
   confirms `bytesList` + `max_correction_bits` are present (what `run_aruco.py`'s
   `load_custom_aruco_dict` expects).

## What changed vs the old monolith

|                      | old `transcode_sleap_aruco.sh`                                            | new `detection_pipeline/`                                    |
| -------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Re-encode step       | yes (split → libx264 encode → encfin)                                     | **dropped** — `ffmpeg -c copy` only                   |
| Per-video scheduling | full pipeline submitted per `vname`                                       | cross-video, chunk-ordered worklist                            |
| SLEAP module         | `sleap/1.4.1` or `sleap/1.5.2` (old `sleap-track`/`sleap-nn-track`) | `sleap-nn/0.2.0` (`sleap-nn predict` + TRT export)         |
| ArUco dict           | default `DICT_4X4_1000` slice                                             | custom `--custom-dict` (A100 or B300)                        |
| Cleanup dep          | `afterok:$bridge:$arucofin` — wedged on transient SSH                    | `afterany:...` — flash freed regardless of saion outcome    |
| SSH cross-cluster    | bare ssh, no retry                                                          | `ssh_retry` / `rsync_retry` with 5 attempts                |
| Container            | `.avi` only                                                               | `.mkv`/`.mp4`/`.avi` discovered; chunks default `.mkv` |
| Global cam           | skipped via `^global_cam` filter                                          | skipped via `^global_` filter (matches new naming)           |

## deigo / saion limits (Reiter unit, `stephensuni` account)

Per-user association limits as of 2026-05-20:

| Partition   | MaxWall | GrpSubmit    | cpu cap | mem cap | Notes                                                                                  |
| ----------- | ------- | ------------ | ------- | ------- | -------------------------------------------------------------------------------------- |
| `compute` | 4 days  | 2016         | 2000    | 7500 G  | bridge + aruco_array live here                                                         |
| `short`   | 2 h     | 4016         | 4000    | 6500 G  | chunk, cleanup — anything that fits in 2 h                                            |
| `datacp`  | (none)  | **20** | 4       | 19 G    | aruco_datacp lives here. Submit count is tight,**keep this leg as single jobs.** |

Practical implications:

- aruco_array at `-c 16 --mem=24G` per task → ceiling is `2000/16 ≈ 125` concurrent tasks (cpu-bound). Default `ARUCO_CONCURRENCY=100` uses ~1600 cpu and leaves headroom for the bridge (also on `compute`); raise toward ~125 only if running standalone — higher just queues `AssocGrpCpuLimit`.
- Bridge must use `compute` (rsync to saion can take hours; 2 h cap on `short` is fatal).
- Anything multiplicative — never submit per-chunk arrays on `datacp` (20-job cap is trivial to blow). Use single jobs.

### saion (Reiter unit, `stephensuni` account)

| Partition      | cpu cap | gpu cap      | mem cap  | MaxWall              | Notes                                                                                                                                                                         |
| -------------- | ------- | ------------ | -------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `largegpu`   | 128     | **8**  | 1 T      | 12 h                 | A100 80GB.**8-GPU cap is the binding limit** for sleap_predict. With `-c 16 --mem=128G --gres=gpu:1` per task and 8 concurrent, all three caps are saturated exactly. |
| `short-a100` | 256     | **32** | 2048 G   | **2 h (1 h non-preempt)** | Same A100 nodes (gpu23-26) as `largegpu`, low priority (tier 1). **4x the GPUs** for sleap_predict, but preemptible→requeued after 1 h. See preset below.            |
| `gpu`        | 72      | 8            | (none)   | 2 days               | V100/P100 mix; usable for aruco GPU detector if quality validation passes                                                                                                     |
| `test-gpu`   | 18      | 2            | (none)   | 8 h                  | small testing partition                                                                                                                                                       |

Each `largegpu` node has 128 cpus / 2 TB RAM / **8x A100 80GB**. Four nodes total → 32 A100s in the partition. On `largegpu` a single user can only hold 8 (`gres/gpu=8`), so `SLEAP_CONCURRENCY=8` is the max there. The new `short-a100` partition (same 4 nodes) lifts the per-user cap to **32 GPU / 256 cpu / 2048 GB** — the whole partition — at the cost of a 2 h walltime and low priority.

Each `largegpu` sleap task at `-c 16 --mem=128G --gres=gpu:1` uses 1/8 of the quota. The TRT inference is GPU-light (~30 % average utilization per A100, bursty between forward pass and CPU postproc); the bottleneck is CPU-side postprocess (peak finding, instance assembly), which scales with the `-c` count. That's why we give each task the max 16 cores allowed by `cpu/8` math.

#### `short-a100` preset (4x concurrency)

Selecting the partition **auto-sizes** concurrency, cpus, mem, and walltime from the
per-user caps (see `saion_caps` in `pipeline.sh`) — so the minimal invocation is just:

```bash
./pipeline.sh --dir ... \
  --saion-partition short-a100 \      # -> conc=32, -c 8, --mem=64G, -t 0-2 (auto)
  --chunk-sec 1800                    # smaller chunks: each task fits the wall AND >=32 tasks fill the slots
```

The auto-derived knobs are equivalent to spelling out
`--sleap-concurrency 32 --sleap-cpus 8 --sleap-mem 64G --sleap-wall 0-2`. Override any
one only to hold resources back — e.g. `--sleap-concurrency 16` (uses 16 of 32 GPUs but
keeps the full `-c 16 --mem=128G` per task), or `--sleap-wall 0-1` to stay strictly inside
the 1 h non-preemptible window once you've confirmed chunks finish in time.

Notes / caveats:

- **At 32 GPUs the cpu cap binds:** auto-sizing gives `256/32 = 8` cpu and `2048/32 = 64 GB` per task (vs. 16/128G on largegpu). Half the cores per task means CPU postproc is somewhat slower per chunk, but 4x the GPUs still nets ~2-3x throughput.
- **Calibrate `--chunk-sec` first.** No TRT (`sleap-nn/0.2.0`) throughput has been measured yet — the `[FPS]` log line exists precisely to capture it. The default `CHUNK_SEC=7200` (2 h of video/chunk) likely will *not* finish inside a 1 h wall at `-c 8`, and a colony may produce <32 such chunks (under-filling the 32 slots). Run a tiny array first, read the `[FPS]`/`Elapsed` lines, then pick a `--chunk-sec` that lands each task at ~30-45 min.
- **Preemption is handled by idempotency.** `PreemptMode=REQUEUE` + the `[[ -f "$out_slp" ]] && continue` skip means a requeued/re-run task resumes without redoing finished chunks. Keep the default (do **not** pass `--no-requeue`). Staying at `--sleap-wall 0-1` avoids preemption entirely.
- **Availability is opportunistic.** `short-a100` is low priority on the *same* nodes as `largegpu`/`gpu-a100`; you get up to 32 GPUs only when they're physically free. The TRT engine is SM80-identical to largegpu's, so it runs unchanged — but switching `--saion-partition` triggers one harmless re-export (the engine cache key includes the partition name).

**Gotcha — `ssh saion-gpu26 nvidia-smi` only shows ONE GPU.** Saion uses pam_slurm_adopt, so ssh sessions get wrapped in one of your running jobs' cgroup and `nvidia-smi` is filtered by `CUDA_VISIBLE_DEVICES`. To check all GPUs on a node, use either:

```bash
# Authoritative: how many physical GPUs Slurm sees on the node
scontrol show node saion-gpu26 | grep -E 'Gres|CfgTRES'

# Per-task GPU status — run nvidia-smi inside a specific job's allocation
ssh saion srun --jobid=4628221_44 nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader

# All your jobs on a given node and their resource allocations
ssh saion squeue -h -w saion-gpu26 -o '%i %u %T %c %m %G'
```

### V100 is NOT viable for sleap-nn 0.2.0 (verified 2026-05-21)

We tried installing `sleap-nn/0.2.0-cu128` to extend saion sleap throughput to the V100s on `gpu` partition. Two stack-level blockers make this impossible without custom-building PyTorch:

- PyTorch 2.11 wheels (cu128 and cu130) compiled for SM ≥ 7.5 only; V100 is SM 7.0.
- cuDNN 9.19 (pulled transitively by torch) requires SM ≥ 7.5.

`sleap-nn/0.1.2` works on V100 but has a different CLI (`sleap-nn-track`, no `sleap-nn predict`, no TRT export). Mixing versions across partitions would require split-architecture pipeline support — too much complexity for ~40 % throughput gain. **Stay A100-only for sleap.**

The cu128 install is left in place at `/apps/unit/ReiterU/sleap-nn/0.2.0-cu128` in case a future PyTorch wheel relaxes the SM list.

Query commands to verify on a new account:

```bash
sacctmgr show assoc user=$USER format=Cluster,Account,User,Partition,QOS,GrpJobs,GrpSubmit,MaxJobs,MaxSubmit,MaxWall -p | column -t -s'|'
for p in short compute datacp largejob; do scontrol show partition=$p | grep -E "MaxTime|QoS|TRES"; done
```

## Cross-cluster diagnostics (read-only mounts)

deigo and saion mount each other's scratch read-only — useful for monitoring
without `ssh`:

| Path                              | Visible from               | Backing FS       |
| --------------------------------- | -------------------------- | ---------------- |
| `/deigo_flash/ReiterU/$USER/…` | saion compute + login (RO) | deigo `/flash` |
| `/saion_work/ReiterU/$USER/…`  | deigo compute + login (RO) | saion `/work`  |

```bash
# From saion-login: watch deigo's pipeline progress
ls /deigo_flash/ReiterU/$USER/jobs/<exp>/
tail /deigo_flash/ReiterU/$USER/jobs/<exp>/aruco_*.out

# From deigo-login: watch saion's sleap output land
ls /saion_work/ReiterU/$USER/<exp>/output/   # .slp files appear here
ls /saion_work/ReiterU/$USER/sleap_export/<model_id>__largegpu/  # TRT engine
```

Mounts are **read-only**. Saion predict tasks read chunks from `/deigo_flash`
and copy them into local `/work` at task start; no bulk deigo→saion rsync is
needed. `cleanup.sbatch` polls bucket for final SLEAP outputs before deleting
`/flash` — if outputs are missing, it exits non-zero and preserves the data.

## Run logs (`hpc_logs/`) — survive mid-run failures

Job logs used to live only on scratch (`/work` on saion, `/flash` on deigo) and were
destroyed by cleanup — `saion_cleanup` removes the `jobs/` dir that holds the sleap
`.out`/`.err`. So a walltime kill, node failure, mass `scancel`, or maintenance drain
left nothing to diagnose. The pipeline now captures logs to bucket under
`<exp>/hpc_logs/` in four layers (defense in depth):

```
<exp>/hpc_logs/
  sleap/     sleap_<A>_<a>.out|.err|.status, prefetch_<A>_<a>.out|.err,
             sacct_sleap_<jid>.tsv                                   (saion)
  aruco/     aruco_<A>_<a>.out|.err|.status, sacct_aruco_<jid>.tsv   (deigo)
  pipeline/  chunk_*, bridge_*, aruco_datacp_*, cleanup_*, manifest.csv, pipeline.env
```

- **Layer 1 — live streaming** (`lib/ship_logs.sh`): each array task ships its own
  Slurm `.out`/`.err` to bucket every `LOG_SHIP_INTERVAL` (default 300s, ship-only-on-change,
  per-task jitter) and on a `TERM`/`EXIT` trap. `#SBATCH --signal=TERM@60` makes Slurm
  deliver SIGTERM ~60s before the walltime SIGKILL, so the final lines + a `.status`
  marker (`reason=signal …`) reach bucket *before* the task dies. This does not depend
  on any downstream job running.
- **Layer 2 — `sacct` post-mortem**: authoritative `State/ExitCode/Reason/Elapsed/MaxRSS`
  from slurmdbd (survives the scratch wipe). sleap sacct is taken in `saion_cleanup`
  (after the array is terminal); aruco sacct in deigo `cleanup`.
- **Layer 3 — archive-before-delete**: `saion_cleanup` and deigo `cleanup` rsync all
  task logs to bucket *before* any `rm`/scratch reclamation (idempotent safety net).
- **Layer 4 — fps line**: each sleap chunk prints `[FPS] <chunk> frames=N elapsed=Ns fps=F`,
  giving a durable, greppable throughput history for the TRT path.

Compute nodes cannot write `/bucket`; every ship rsyncs over SSH to the cluster login
alias (`deigo:` / `saion:`), the same mechanism the `.slp`/`.h5` uploads use. Payloads
are KB–MB text, so node load is negligible; the only real cost is SSH connections,
kept low by the coarse interval, change-gating, and jitter (and `rsync_retry` backoff
for the documented `kex_exchange_identification` resets).

Quick triage after a failed/odd run:

```bash
grep -h '^\[FPS\]' /bucket/.../<exp>/hpc_logs/sleap/*.out | sort   # throughput per chunk
cat /bucket/.../<exp>/hpc_logs/sleap/*.status                       # which tasks died and why
column -t -s'|' /bucket/.../<exp>/hpc_logs/sleap/sacct_sleap_*.tsv  # TIMEOUT/OOM/NODE_FAIL/CANCELLED
```

## Open verifications (before relying on this for new experiments)

- Sidecar JSON field names from pylonrecorder2 ([VIDEO_AI_HANDOFF.md](../../PylonRecorder2/docs/VIDEO_AI_HANDOFF.md)) — `manifest.py` accepts `fps`/`framerate`/`FPS` and `frames_encoded`/`frame_count`/`frames`/`frames_emitted`. Run the manifest builder once on a real recording dir and check the warnings.
- saion `~/.ssh/config` has a `Host saion` alias that resolves to a login with `/bucket` write (used by `sleap_datacp_array` to upload SLP files).
- saion has the `sleap-nn/0.2.0` module loaded by the install we did 2026-05-19 (`module load sleap-nn/0.2.0` on saion-gpu24 should print no error).
- saion partition for SLEAP datacp can run a small CPU task that ssh's the login — defaults to `test-gpu` (no GPU consumed); override with `SAION_DATACP_PARTITION=...` if it changes.
