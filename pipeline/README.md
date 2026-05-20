# AntsArray pipeline (v2)

Chunk-ordered ArUco + SLEAP pipeline for PylonRecorder2 outputs.

Replaces the monolithic [transcode_sleap_aruco.sh](../transcode_sleap_aruco.sh).
Drops the re-encode stage (no longer needed: pylonrecorder2 emits clean
GOP-aligned `.mkv`/`.mp4`/`.avi` with sidecar diagnostics), switches SLEAP to
`sleap-nn/0.2.0` with optional TensorRT export, and schedules per-chunk
across all grid cameras (all `_000` before any `_001`) so early-time results
land in `<exp>/data/` first.

## Architecture

```
deigo-login (pipeline.sh)
  └── chunk_array (one task per grid video)
        └── chunk_finalize
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
pipeline/
  pipeline.sh                      # entry point (deigo-login)
  README.md                        # this file
  lib/
    hosts.sh                       # SSH_CMD, ssh_retry, rsync_retry, host_resolves
    manifest.py                    # video discovery + sidecar/ffprobe cross-check
    worklist.py                    # chunk-ordered (chunk_idx ASC, vname ASC) TSV builder
  templates/
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
```

## Quick start

From `deigo-login`:

```bash
bash pipeline/pipeline.sh \
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

Outputs land in `<exp>/data/`:
- `<vname>_NNN_aruco_tracks_.h5` (from deigo aruco array)
- `<vname>_NNN.slp`               (from saion sleap predict)

## CLI

See `bash pipeline/pipeline.sh --help` for the full option list. Notable
defaults:

| Flag | Default | Notes |
|---|---|---|
| `--chunk-sec` | `7200` (2 h) | passed to `ffmpeg -segment_time` |
| `--chunk-ext` | `mkv` | output container; sleap-nn 0.2 reads mkv via sleap-io |
| `--aruco-dict` | `A` | resolves to `custom_4x4_A100_d4_*.npz` (latest by name) under `/bucket/ReiterU/Ants/aruco_dicts/` |
| `--sleap-runtime` | `tensorrt` | also `onnx`, `pytorch` (last = no export needed) |
| `--skip-trt-export` | off | fall back to `sleap-nn track` (raw model dirs, no export) |
| `--saion-partition` | `largegpu` | A100 SM80 |
| `--sleap-module` | `sleap-nn/0.2.0` | saion module to `module load` for predict tasks |
| `--aruco-concurrency` | `16` | array `%N` cap |
| `--sleap-concurrency` | `8` | array `%N` cap |
| `--datacp-concurrency` | `4` | array `%N` cap (deigo has 4 mover nodes) |

## Phase isolation (for testing)

```bash
bash pipeline.sh --dir ... --only-chunk    # stop after chunk submission
bash pipeline.sh --dir ... --only-aruco    # skip bridge / saion
bash pipeline.sh --dir ... --only-sleap    # skip aruco array+datacp
```

## Pre-flight checks (run once before first real run)

1. **mkv readable by sleap-nn**: on saion-gpu24, `sleap-nn predict
   <export_dir> <some_chunk.mkv> --runtime tensorrt --n-frames 100` succeeds.
2. **TRT export of legacy `best_model.h5` models**: run
   `pipeline/scripts/export_sleap_trt.sh --centroid <dir> --instance <dir> --out
   /tmp/exporttest --runtime tensorrt` and confirm `model.trt` is created. If
   the export errors on legacy weights, set `--skip-trt-export` and the
   `sleap_predict_array` task falls back to `sleap-nn track` (PyTorch).
3. **ArUco dict A npz schema**: on deigo,
   `python3 -c "import numpy as np; d=np.load('/bucket/ReiterU/Ants/aruco_dicts/custom_4x4_A100_d4_20260410_103938.npz'); print(list(d.keys()), d['bytesList'].shape, int(d['max_correction_bits']))"`
   confirms `bytesList` + `max_correction_bits` are present (what `run_aruco.py`'s
   `load_custom_aruco_dict` expects).

## What changed vs the old monolith

| | old `transcode_sleap_aruco.sh` | new `pipeline/` |
|---|---|---|
| Re-encode step | yes (split → libx264 encode → encfin) | **dropped** — `ffmpeg -c copy` only |
| Per-video scheduling | full pipeline submitted per `vname` | cross-video, chunk-ordered worklist |
| SLEAP module | `sleap/1.4.1` or `sleap/1.5.2` (old `sleap-track`/`sleap-nn-track`) | `sleap-nn/0.2.0` (`sleap-nn predict` + TRT export) |
| ArUco dict | default `DICT_4X4_1000` slice | custom `--custom-dict` (A100 or B300) |
| Cleanup dep | `afterok:$bridge:$arucofin` — wedged on transient SSH | `afterany:...` — flash freed regardless of saion outcome |
| SSH cross-cluster | bare ssh, no retry | `ssh_retry` / `rsync_retry` with 5 attempts |
| Container | `.avi` only | `.mkv`/`.mp4`/`.avi` discovered; chunks default `.mkv` |
| Global cam | skipped via `^global_cam` filter | skipped via `^global_` filter (matches new naming) |

## deigo / saion limits (Reiter unit, `stephensuni` account)

Per-user association limits as of 2026-05-20:

| Partition  | MaxWall   | GrpSubmit | cpu cap | mem cap   | Notes |
|---         |---        |---        |---      |---        |---|
| `compute`  | 4 days    | 2016      | 2000    | 7500 G    | bridge + aruco_array live here |
| `short`    | 2 h       | 4016      | 4000    | 6500 G    | chunk, cleanup — anything that fits in 2 h |
| `datacp`   | (none)    | **20**    | 4       | 19 G      | aruco_datacp lives here. Submit count is tight, **keep this leg as single jobs.** |

Practical implications:
- aruco_array at `-c 4 --mem=8G` per task → ceiling is `2000/4 = 500` concurrent tasks (cpu-bound). Default `ARUCO_CONCURRENCY=128` leaves headroom for other work; bump to ~400 if running standalone.
- Bridge must use `compute` (rsync to saion can take hours; 2 h cap on `short` is fatal).
- Anything multiplicative — never submit per-chunk arrays on `datacp` (20-job cap is trivial to blow). Use single jobs.

Query commands to verify on a new account:
```bash
sacctmgr show assoc user=$USER format=Cluster,Account,User,Partition,QOS,GrpJobs,GrpSubmit,MaxJobs,MaxSubmit,MaxWall -p | column -t -s'|'
for p in short compute datacp largejob; do scontrol show partition=$p | grep -E "MaxTime|QoS|TRES"; done
```

## Cross-cluster diagnostics (read-only mounts)

deigo and saion mount each other's scratch read-only — useful for monitoring
without `ssh`:

| Path | Visible from | Backing FS |
|---|---|---|
| `/deigo_flash/ReiterU/$USER/…` | saion compute + login (RO) | deigo `/flash` |
| `/saion_work/ReiterU/$USER/…`  | deigo compute + login (RO) | saion `/work` |

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

## Open verifications (before relying on this for new experiments)

- Sidecar JSON field names from pylonrecorder2 ([VIDEO_AI_HANDOFF.md](../../PylonRecorder2/docs/VIDEO_AI_HANDOFF.md)) — `manifest.py` accepts `fps`/`framerate`/`FPS` and `frames_encoded`/`frame_count`/`frames`/`frames_emitted`. Run the manifest builder once on a real recording dir and check the warnings.
- saion `~/.ssh/config` has a `Host saion` alias that resolves to a login with `/bucket` write (used by `sleap_datacp_array` to upload SLP files).
- saion has the `sleap-nn/0.2.0` module loaded by the install we did 2026-05-19 (`module load sleap-nn/0.2.0` on saion-gpu24 should print no error).
- saion partition for SLEAP datacp can run a small CPU task that ssh's the login — defaults to `test-gpu` (no GPU consumed); override with `SAION_DATACP_PARTITION=...` if it changes.
