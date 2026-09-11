# basler catalog

A one-command metadata catalog for every filming session under
`/bucket/ReiterU/Ants/basler`. Walks the tree, classifies each folder, reads
video sidecars for recording health, scans `detection_pipeline` outputs for
processing status, parses stimulation session logs, and emits flat CSVs you can
open in Excel.

## Run

```bash
# full catalog into <root>/_catalog/
python detection_pipeline/catalog.py all \
    --root Z:/ReiterU/Ants/basler \
    --outdir Z:/ReiterU/Ants/basler/_catalog

# refresh only some sessions (rest are preserved from cache)
python detection_pipeline/catalog.py all --only 20260624,20260623

# re-emit CSVs from the cache without walking the tree
python detection_pipeline/catalog.py build
```

Flags: `--force` (ignore cache), `--allow-ffprobe` (probe sidecar-less `.avi`,
slow), `--parquet` (also write `.parquet` mirrors, needs pandas+pyarrow),
`--check-sizes` (stat data files to flag truncated artifacts), `--workers N`.

The default `--root` is `/bucket/ReiterU/Ants/basler` (Linux/HPC); on Windows
pass the `Z:` mount. Runs auto-log to `<outdir>/logs/catalog_<UTC>.log`.

## Outputs (in `_catalog/`)

| file | grain | what |
|------|-------|------|
| `catalog.csv` | one row per **block** (a flat session = one implicit block) | the "one-go" sheet: identity, stim summary, camera counts, recording health, pipeline status + completeness |
| `videos.csv` | one row per grid video | per-camera health: fps, frames, missed frames, clean-close, PC/drive |
| `trials.csv` | one row per vibration pulse | each `CSV_PULSE` with camera frame range + IMU + temperature |
| `calibrations.csv` | one row per homography stack under `cameraArray_calib/` | the calibration registry: date, `valid_from`, file, sha256, how many blocks expect / were tracked with it |
| `catalog.html` | — | self-contained browser viewer (data embedded; see below) |
| `catalog_run.json` | — | run summary + ignored/unknown entries |
| `.scan_cache.jsonl` | — | incremental cache (do not edit) |

## View in a browser

Open `_catalog/catalog.html` directly in any browser (double-click, or
`file:///bucket/ReiterU/Ants/basler/_catalog/catalog.html`). It is fully
self-contained — the data is embedded, so no server is needed and it works
offline. It regenerates on every `catalog.py` run.

Features: KPI tiles (blocks, complete/partial/not-started, stim, flagged);
Catalog / Videos / Trials tabs; click any column header to sort; a text filter
plus dropdown facets (kind, pipeline status, stim, health) and a "flagged only"
toggle; colour-coded status/health/hazard chips (green=good, amber=warning,
red=critical — always with a text label); a light/dark toggle. To share it as a
standalone file, just copy `catalog.html` — it needs nothing else.

Key = `session_id` + `block` (present in all three files, so they join).

## How folders are classified

* **session** — date-named (`20260624`, `20250321_2_test`, date ranges,
  `YYYYMMDD-HHMMSS`, `2025_Sep_...`), or an unrecognized folder that contains
  videos / a `sess` file / a `data/` footprint.
* **block** layout — a session containing `block01`, `block02`, … (one row each);
  otherwise **flat** (one implicit block, `block` blank). Non-block sibling dirs
  that hold videos (calibration datasets) become their own `NONBLOCK_VIDEO_DIR`
  rows.
* **pure_analysis** — a session with an analysis footprint but zero raw videos
  (`per_track/`, `predictions/`, …).
* **aux** — known non-experiment folders (`Anouk`, `cameraArray_calib`,
  `pipelineTest`, …); a thin row, not deeply scanned.
* `single_ants/` and `2025_Sep_no_pertubation/` are recursed one level so their
  sub-sessions each get a row. Loose files and `_`/`.`-prefixed dirs are
  recorded under `ignored` in `catalog_run.json`.

## Recovery (reprocess only the missing files)

Every pipeline stage is a SLURM array over a worklist and skips outputs that
already exist, so recovering a partial block = feeding the stage a **sub-worklist
of only the missing chunks**. The catalog already knows that set.

```bash
python detection_pipeline/catalog.py recover 20260623/block03
```
This reports the missing set per stage, writes the exact sub-worklist to
`_catalog/recover/<session>_<block>.<stage>.worklist.txt` (pipeline format
`vname⇥NNN⇥expected_frames`, with `--chunk-sec` inferred from fps/frames), and
prints the resubmit command(s). In the dashboard, every partial block shows a
**recover** button (Catalog tab) that opens the same missing counts + commands.

Recovery types, cheapest first:
- **upload** (`SILENT_PARTIAL`): outputs computed but not uploaded → rescue-copy from the saion login side.
- **slp2h5** (`SLEAP_H5_MISSING`): `.slp` present, `_sleap_data.h5` missing → `slp2h5_array.sh` over the bucket `.slp` (CPU only, no re-chunk).
- **aruco**: aruco behind → aruco array over the sub-worklist.
- **sleap** (`.slp` missing, e.g. block03's 287 chunks): re-chunk those chunks → SLEAP → slp2h5 → upload. The command output gives both the simple `pipeline.sh --only-sleap` re-run and the minimal per-chunk resubmit. The simple re-run is bucket-aware by default: it re-chunks the whole block but runs SLEAP only on the missing/incomplete chunks (a chunk is complete when its `.slp` + `_sleap_data.h5` are on the bucket and the h5's `expected_frames` attr matches the worklist). Pass `--sleap-force-recompute` to redo every chunk, e.g. when applying a new model.

### Consistency — models auto-filled per block

`pipeline.sh --only-sleap` also needs the SLEAP model paths. The catalog reads
them from the block's **own** records, in priority order: `data/PIPELINE_STATE.json`
(the processing contract, blocks from 20260722 on), then `hpc_logs/pipeline/bridge_*.out`
(the pipeline echoes `centroid:` / `instance:`), then `hpc_logs/pipeline/pipeline.env`
(the `export SLEAP_MODEL_*` configuration every run was submitted with — the only
source for blocks whose bridge jobs predate the echo). So the recovery command
reuses the *exact* models that block was processed with — different blocks
legitimately used different models, so this is per-block, not a global guess. The
model set also shows in the `sleap_models` catalog column for at-a-glance
consistency checks. `pipeline.env` likewise supplies the saion partition when
nothing better recorded it. The real `aruco_worklist.txt` (also under `hpc_logs`)
supplies the authoritative per-chunk `expected_frames`, so the sub-worklist is
exact, not inferred.

If a block has no logs, models fall back to an optional
`_catalog/recover.config.json`:

```json
{"sleap_model_centroid": "/bucket/…/250408_141245.centroid",
 "sleap_model_instance": "/bucket/…/250408_141245.centered_instance"}
```

Caveat: `sleap`/`aruco` recovery reads chunk videos from `/flash`, which is
cleaned after a run — for an old block those chunks must be re-chunked first
(the emitted steps say so).

## Tracking provenance (`tracks/TRACKING_STATE.json`)

Which homography stack (`--hmats`, i.e. which camera calibration) and which
panorama split (`--x_threshold`) made a block's tracks. Detection always
recorded its settings (`pipeline.env`, `PIPELINE_STATE.json`); tracking did
not, and a block tracked against a stale calibration is indistinguishable on
disk from one tracked against the right one.

`tracking/colony/pipeline.py` now writes the record before mapping (and adds
the stitch settings before stitching). It lives in `tracks/` because that
directory already travels to the bucket with the tracking transfer manifest,
and every window view has its own `tracks/`. Re-mapping with a different hmats
or split moves the displaced record into `history`, so retracks stay visible.

Catalog columns: `tracking_hmats` (the calibration directory name, e.g.
`20260623_calib_elevated_by_2mm_from_arenafloor`), `tracking_hmats_path`,
`tracking_x_threshold`, `tracked_at`, `tracking_source` (`pipeline` |
`backfill`). The viewer shows `hmat used` next to `tracking`; hover for the
full path, split and time.

Blocks tracked before the record existed show `TRACKING_UNRECORDED`. Backfill
them from evidence (the rendered sbatch under `/flash`, a lab note):

```bash
python detection_pipeline/catalog.py track-init 20260716/block01 \
    --hmats /bucket/ReiterU/Ants/basler/cameraArray_calib/20260623_calib_elevated_by_2mm_from_arenafloor/frame0/aruco_stitch/aruco_H_mats.npz \
    --x-threshold 2475 --tracked-at 2026-07-20 --note "rendered sbatch on /flash"
```

`--dry-run` prints without writing. A record the pipeline wrote itself is never
replaced by a backfill unless `--overwrite` (the displaced record stays in
`history`); a block without a `tracks/` directory is refused unless
`--allow-missing-tracks`. The hmats file is hashed when readable, so two records
are comparable even after a calibration is recomputed in place.

## Calibration registry (`calibrations.csv`)

Which camera-array calibration (homography stack) applies to which block. Every
`aruco_H_mats.npz` / `initial_H_mats.npz` / `refined_H_mats.npz` /
`aruco_board_H_mats.npz` under `cameraArray_calib/<calib_id>/…` becomes a row,
keyed by the calibration directory name and dated from it (the npz stores no
date; its mtime is the *computation* date, which for the 20260414 set is two
months later). `n_cams` is read from the npy header without numpy.

A block's **expected** calibration (`calib_expected`, `calib_expected_from`) is
the enabled row with the latest `valid_from` on or before the block's
`date_start`. `valid_from` defaults to the calibration date; adjust it, retire a
calibration, or annotate it in `<outdir>/calibration_overrides.csv`
(`calib_id,valid_from,enabled,note`; see `calibration_overrides.example.csv`).
Blocks with a fuzzy date, and the calibration filmings themselves, get no
expectation.

Where a block has a tracking record (`tracks/TRACKING_STATE.json`) naming a
different calibration, the row gets `HMAT_MISMATCH` and the viewer shows an
amber *mismatch* chip in the `calib` column: that block was tracked with a
stale or wrong homography and should be re-tracked. The `calibrations` tab lists
the registry with per-calibration counts of blocks expecting it and blocks
tracked with it, and the KPI row counts `hmat mismatch` blocks.

On the **timeline**, each enabled calibration is a dashed vertical marker at its
`valid_from` (labelled `calib MM/DD`, hover for the id, filming date and counts),
in a colour that is reused everywhere: each block bar carries a bottom strip in
the colour of its *expected* calibration and a top strip in the colour it was
*tracked with* (grey when tracked but unrecorded); a mismatch also outlines the
bar in dashed red. Markers more than 45 days outside the shown blocks are
omitted so an old calibration cannot stretch the axis. The legend in the
timeline toolbar maps colours to calibration dates.

The registry is applied after the scan cache, like the label overlay. So after a
new calibration: put its `*_H_mats.npz` under `cameraArray_calib/<date>_…/`,
optionally add an override row, and run `catalog.py all` (or `build`) — every
block from that date on switches its expectation, and anything already tracked
with the older stack is flagged. No rescan, no code change.

## Completeness (honest by design)

Newer colony blocks carry no chunk-count ground truth on disk (the pipeline's
`--chunk-sec` varies per run), so completeness is computed **internally**: the
deepest processing stage reached (usually aruco) sets the expected chunk count,
and `completeness_state=internal` marks the percentage as relative to that, not
to an absolute duration. `verified` would require an external chunk source;
`unverifiable` / `n/a` mean no honest number is possible. Completeness never
opens `.h5`/`.slp` files — it counts filenames only.

## Hazard flags (`hazard_flags`, `|`-joined)

`SLEAP_H5_MISSING` (slp present, sleap_data.h5 absent) ·
`STAGE_SKEW` (aruco≠slp counts) · `ARUCO_MISSING` (aruco behind slp) ·
`SILENT_PARTIAL` (outputs but no HPC logs — gpu25 upload-loss) ·
`DEAD_SYMLINK` (block00-style dangling video) · `NONBLOCK_VIDEO_DIR` ·
`CAM_NAMING_LEGACY` (`camN_..._camM` order) · `PIPELINE_FORMAT_LEGACY`
(`.npy`/`.csv`/trailing-`_` outputs; h5 flags suppressed) ·
`NAME_DATE_MISMATCH` · `NO_SESS_FILE` · `NO_SIDECAR` · `CAM_COUNT_OFF`
(≠25 colony cams) · `CHUNK_INTERNAL_ONLY` / `CHUNK_UNVERIFIABLE` ·
`TRUNCATED_ARTIFACT` (needs `--check-sizes`) · `RAW_CHUNKED` ·
`TRACKING_UNRECORDED` (`tracks/` landed but no `tracks/TRACKING_STATE.json`
says which homography made them — the `track-init` backfill queue) ·
`TRACKING_STATE_CORRUPT` (the record exists but cannot be read — inspect it,
do not backfill over it) · `HMAT_MISMATCH` (tracked with a different
calibration than the registry expects for the block's date — retrack candidate).

## Example questions

* Fully-processed stim blocks → filter `is_stim=true`, `pipeline_status=complete`.
* Blocks missing SLEAP outputs → `hazard_flags` contains `SLEAP_H5_MISSING`.
* Filmed but never processed → `session_kind=session`, `pipeline_status=not_started`.
* Compute-done-but-upload-lost → `hazard_flags` contains `SILENT_PARTIAL`.
* Every vibration pulse + frame range for a session → open `trials.csv`, filter
  `session_id`, read `cam_frame_start`/`cam_frame_end`.

## Design notes

* Sidecar-first: fps/frames come from `*.diag.json`; `ffprobe` runs only with
  `--allow-ffprobe` on sidecar-less `.avi`. Never decodes video at scale.
* Reuses `detection_pipeline/lib/manifest.py` (imported unmodified) for sidecar
  discovery/parsing.
* Incremental: a block is re-read only when its fingerprint (video count,
  `data/` file count, sess-file mtime/size, …) changes; unchanged blocks are
  reused from `.scan_cache.jsonl`. Bump `SCAN_VERSION` in `const.py` after
  changing scan logic.

## Known limitations

* `data_old/` and repo/env trees are not walked (see `WALK_BLACKLIST`); a
  session whose only outputs live under `data_old/` reads as `not_started`.
* `expected_cams` is hard-coded to 25 for colony sessions; single-ant/calib
  sessions therefore raise `CAM_COUNT_OFF` (informational).
* Aruco empty-result (`(N,0,2)`) files are indistinguishable from real ones
  without opening the `.h5`; only truncated/tiny files are flagged, and only
  with `--check-sizes`.
