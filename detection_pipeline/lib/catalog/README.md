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
- **sleap** (`.slp` missing, e.g. block03's 287 chunks): re-chunk those chunks → SLEAP → slp2h5 → upload. The command output gives both the simple `pipeline.sh --only-sleap` re-run and the minimal per-chunk resubmit.

### Consistency — models auto-filled per block

`pipeline.sh --only-sleap` also needs the SLEAP model paths. The catalog reads
them from the block's **own** `hpc_logs/pipeline/bridge_*.out` (the pipeline
echoes `centroid:` / `instance:`), so the recovery command reuses the *exact*
models that block was processed with — different blocks legitimately used
different models, so this is per-block, not a global guess. The model set also
shows in the `sleap_models` catalog column for at-a-glance consistency checks.
The real `aruco_worklist.txt` (also under `hpc_logs`) supplies the authoritative
per-chunk `expected_frames`, so the sub-worklist is exact, not inferred.

If a block has no logs, models fall back to an optional
`_catalog/recover.config.json`:

```json
{"sleap_model_centroid": "/bucket/…/250408_141245.centroid",
 "sleap_model_instance": "/bucket/…/250408_141245.centered_instance"}
```

Caveat: `sleap`/`aruco` recovery reads chunk videos from `/flash`, which is
cleaned after a run — for an old block those chunks must be re-chunked first
(the emitted steps say so).

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
`TRUNCATED_ARTIFACT` (needs `--check-sizes`) · `RAW_CHUNKED`.

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
