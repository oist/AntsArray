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

Outputs land in `<exp>/data/`:

- `<vname>_NNN_aruco_tracks_.h5` (from deigo aruco array)
- `<vname>_NNN.slp`               (from saion sleap predict)

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

```bash
./pipeline.sh --dir ... \
  --saion-partition short-a100 \
  --sleap-concurrency 32 \
  --sleap-cpus 8 --sleap-mem 64G \   # 256/32 cpu, 2048/32 GB — saturates all three caps at 32 tasks
  --sleap-wall 0-1 \                  # 1 h wall => task never outlives the non-preemptible window
  --chunk-sec 1800                    # smaller chunks: each task fits 1 h AND >=32 tasks fill the slots
```

Notes / caveats:

- **At 32 GPUs the cpu cap binds:** `256/32 = 8` cpu and `2048/32 = 64 GB` per task (vs. 16/128G on largegpu). Half the cores per task means CPU postproc is somewhat slower per chunk, but 4x the GPUs still nets ~2-3x throughput.
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
destroyed by cleanup — saion's `rm -rf "$REMOTE_ROOT"` deletes the `jobs/` dir that
holds the sleap `.out`/`.err`. So a walltime kill, node failure, mass `scancel`, or
maintenance drain left nothing to diagnose. The pipeline now captures logs to bucket
under `<exp>/hpc_logs/` in four layers (defense in depth):

```
<exp>/hpc_logs/
  sleap/     sleap_<A>_<a>.out|.err|.status, sacct_sleap_<jid>.tsv   (saion)
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
