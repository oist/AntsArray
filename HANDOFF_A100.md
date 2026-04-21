# Handoff: Custom 4×4 A100 ArUco dictionary on Deigo pipeline

Portable primer for continuing this work on another PC. Delete this file before the eventual merge to `main`.

**Branch:** `feature/nn-aruco-detection`
**Tip commit:** `6ed82c7` — "WIP checkpoint: A100 custom-dict support + NN-aruco experiments"
**Last session:** 2026-04-21

---

## Quick start on the second PC

```bash
# If you already have an AntsArray clone:
cd <path-to>/AntsArray
git fetch origin
git checkout feature/nn-aruco-detection
git pull

# Fresh clone:
git clone https://github.com/oist/AntsArray.git
cd AntsArray
git checkout feature/nn-aruco-detection
```

Read order: this file → `CLAUDE.md` (repo overview) → `run_aruco.py` → `transcode_sleap_aruco.sh`.

---

## What was built

The repo ArUco pipeline previously hard-coded OpenCV's built-in `DICT_4X4_1000`. A custom 4×4 dictionary of 100 tags ("A100") with minimum Hamming distance 4 was generated on 2026-04-10 for lower false-positive rates, and wired through the pipeline as an **opt-in** flag.

### Plumbing changes

**[run_aruco.py](run_aruco.py)**
- `load_custom_aruco_dict(npz_path)` at [run_aruco.py:67-74](run_aruco.py#L67-L74) — loads NPZ into a `cv2.aruco.Dictionary`.
- `detect_aruco_in_video()` at [run_aruco.py:100-108](run_aruco.py#L100-L108) now takes an explicit `aruco_dict` arg; no more hardcoded `DICT_4X4_1000`.
- New CLI flag `--custom-dict PATH` at [run_aruco.py:236-241](run_aruco.py#L236-L241).
- `main()` at [run_aruco.py:292-306](run_aruco.py#L292-L306) picks dictionary before dispatch and refuses `--custom-dict` combined with any NN detector (NN decoders were trained on DICT_4X4_1000).

**[transcode_sleap_aruco.sh](transcode_sleap_aruco.sh)**
- New `--custom-dict PATH` CLI arg + `ARUCO_CUSTOM_DICT` env fallback at [transcode_sleap_aruco.sh:183-203](transcode_sleap_aruco.sh#L183-L203).
- New placeholder `__ARUCO_CUSTOM_DICT_ARG__` in the SLURM array template at [transcode_sleap_aruco.sh:660](transcode_sleap_aruco.sh#L660), substituted at [transcode_sleap_aruco.sh:1019](transcode_sleap_aruco.sh#L1019). When omitted it collapses to empty, preserving the DICT_4X4_1000 default.

---

## External resources (cluster-side, same from any PC)

### Custom dictionary on /bucket
- NPZ: `/bucket/ReiterU/Ants/aruco_dicts/custom_4x4_A100_d4_20260410_103938.npz`
- JSON manifest: `/bucket/ReiterU/Ants/aruco_dicts/custom_4x4_A100_d4_20260410_103938.json`

Source of the NPZ is in the repo at `nn-aruco-detection-test/results/custom_dicts/` (gitignored — NPZ on bucket is canonical).

### Deigo temp workspace
- `~/tmp_aruco_A100_20260421/` on Deigo contains the **patched** orchestrator and run_aruco.py used for the current test run. Patches vs repo:
  - `flash_root` and `jobs_root` overridden from `/flash/ReiterU/$USER/` → `/flash/StephensU/makoto-hiroi/`.
  - `saion_root` and `saion_jobs_root` overridden from `/work/ReiterU/$USER/` → `/work/StephensU/makoto-hiroi/`.
  - Backup of the pre-patch copy: `transcode_sleap_aruco.sh.bak`.
- These overrides are **temp-test-only**. The repo copy is unmodified so production ReiterU runs still work.

### Per-user scratch roots (created 2026-04-21, 2775 setgid)
- `/flash/StephensU/makoto-hiroi/` (Deigo side)
- `/work/StephensU/makoto-hiroi/` (Saion side)

---

## How to kick off the A100 test pipeline

**From a Deigo login node**, not via `sbatch`:

```bash
ssh deigo
cd ~/tmp_aruco_A100_20260421
bash transcode_sleap_aruco.sh \
    --dir /bucket/ReiterU/Ants/basler/20260414_20260417_CustomAruco/block01/ \
    --custom-dict /bucket/ReiterU/Ants/aruco_dicts/custom_4x4_A100_d4_20260410_103938.npz
```

Why `bash` and not `sbatch`: the orchestrator's initial `ensure_dir "$DIR/data"` writes inside `/bucket`, which is read-only from Deigo `compute` nodes but writable from login nodes. The orchestrator body is light — it just submits the first `split` job and exits; the real pipeline runs via SLURM dependencies.

### Monitor

```bash
squeue -u makoto-hiroi
ls /flash/StephensU/makoto-hiroi/jobs/block01/
cat /flash/StephensU/makoto-hiroi/jobs/block01/pipeline.jobs
```

### Verify A100 actually took effect

After the first aruco array task finishes, on Deigo:

```bash
grep "custom dict" /flash/StephensU/makoto-hiroi/jobs/block01/aruco-*_0.out
# Expected: "[INFO] Using custom dict /bucket/.../A100.npz (100 markers)"

python3 -c "import h5py; f=h5py.File('/bucket/ReiterU/Ants/basler/.../block01/data/block01_000_aruco_tracks.h5'); print(f['aruco_tracks'].shape)"
# Expected: (N_frames, 100, 2)   <- 100, not 300 or 1000
```

---

## Known gotchas

1. **Windows CRLF line endings.** If you scp the scripts from a Windows PC to Deigo, run `sed -i "s/\r$//" transcode_sleap_aruco.sh run_aruco.py` on Deigo afterwards — otherwise SLURM rejects with `DOS line breaks`.
2. **Bucket is read-only on compute nodes.** Run the orchestrator via `bash` from a login node, not `sbatch`, when `--dir` is a `/bucket` path. (See Phase above.)
3. **`datacp` is not an SSH host from Windows.** It's a Deigo-side SLURM partition used inside the orchestrator's rsync-to-bucket helper jobs. From outside the cluster, SSH via `deigo` or `saion` aliases in `~/.ssh/config`.
4. **`--custom-dict` + NN detector is refused by design.** The `yolo`/`rtdetr`/`deeparuco*` detectors assume DICT_4X4_1000. Mixing would silently return garbage IDs.

---

## Where we left off

- Test pipeline invocation prepared on Deigo but not yet successfully run (last attempt failed on the `sbatch → compute → /bucket read-only` gotcha). Next action: retry with `bash` from login node as documented above.
- `benchmark/results/` on PC1 is intentionally unpushed — it has local sweep outputs that haven't been cleaned for commit.

---

## Housekeeping notes for the other PC

- `.claude/`, `.omc/`, `.omx/`, `feature-nn-aruco-detection-handoff.md` are now in `.gitignore` — AI-tool state is per-machine.
- Your primary group on Deigo should be `stephensuni` (gid 2020). Check with `id` on Deigo if paths permission-fail.
