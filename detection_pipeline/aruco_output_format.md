# ArUco output format: preserve repeated IDs within a camera/frame

Both detector entry points and the colony panorama loader implement this format.
Updating repository code does not deploy an HPC release or replace previously
computed detections/tracks. No detector parameters or tracker assignment logic
changed.

## What changed

`scripts/run_aruco_mp.py` and `../run_aruco.py` retain every raw detection rather
than rebuilding the detection table from a dense, single-position-per-ID array.
Both use `scripts/aruco_output.py` to pack and atomically publish complete files.

- `*_aruco_tracks.h5` contains a new plain-HDF5 compound dataset,
  `aruco_detections`: one row per detection with `Frame`, `Instance` (tag ID),
  `X`, `Y`, and `Confidence`. Repeated `(Frame, Instance)` pairs are intentional.
- The old `aruco_tracks` and `aruco_confidences` arrays remain unchanged for
  compatibility. They are **lossy summaries**, not authoritative detections.
  File metadata records the number of extra detections absent from the summary.
- The pandas HDF table and CSV also retain every detection. Both HDF forms
  record the total frame count, including empty/trailing frames.
- `detect_aruco_mp()` and `detect_aruco_in_video()` now return three values:
  legacy tracks, legacy confidence, and the lossless DataFrame. All in-repo
  callers were updated.
- `tracking/colony/map_combine.py` prefers native lossless records, including
  when discovery selects an old/unreadable pandas sidecar. Marked-v2 files with
  missing/unreadable lossless data cannot silently fall back to dense data.
  Legacy files still load, with an explicit warning that duplicates may be lost.
- Native record files can also be loaded without the legacy arrays, provided
  they retain explicit `num_frames` metadata. Conflicting frame spans are rejected.

The existing left/right split and tracker then receive all candidates. Tests
verify that an established track chooses the closer eligible duplicate,
independently of input order and whether candidates have the same camera ID.
Initialization without track history is not redesigned here.

Third-party/direct readers of the dense dataset still see only the compatibility
summary. They must use `aruco_detections` to obtain lossless detection counts.

A production rollout must include **both the exporter and mapper updates**,
including the new shared `scripts/aruco_output.py` module (do not copy a detector
script alone). Existing data requires ArUco-only recomputation followed by
remapping/retracking and rebuilding dependent analyses; SLEAP recomputation is
unnecessary. Existing `.aruco.ok` sentinels and `--skip_existing` outputs do not
invalidate themselves on code changes, so do not reuse those caches for a
requested rebuild. This patch does not trigger that rebuild automatically.

## Camera03 test

Dataset: `20260810/block02-w000-031`. Camera03 straddles both arenas.
Replayed original frames **600–719** (120 frames, 5 seconds), using the recorded
custom dictionary and detector parameters and the dataset's homographies.
Test outputs rebase that interval to frames 0–119; `summary.json` records the
original offset and source/code fingerprints.

The test runs the actual production detection worker, packing/export functions,
panorama mapper and unchanged tracker with existing SLEAP detections. A control
uses the **same replay detections** with the old last-write-per-ID rule, so the
comparison isolates the storage change rather than OpenCV-version changes.

- Lossless detections: **2,023**, versus **1,958** under the old rule.
- **65** additional detections across **64/120** frames; duplicate IDs: 7, 17, 75.
  These are detector records, not 65 unique ants or independently verified IDs.
- At original frame637, **both ID75 positions are saved, assigned to opposite
  colonies, and tracked**. The old rule discards the right-colony detection.

| ID75 frames, out of 120 | Left old | Left patched | Right old | Right patched |
| --- | ---: | ---: | ---: | ---: |
| Tag detection retained | 22 | 68 | 80 | 97 |
| Track present | 45 | 66 | 77 | 91 |

Tracking can bridge missed tags, so track presence and tag detections differ.
This is a selected, single-camera functionality test, not an estimate of the
bug's frequency or tracking accuracy across the whole recording.

Local OpenCV was **4.11.0**, versus production **4.9.0**. Legacy replay detection
presence exactly matched the saved arrays throughout this interval; 35 positions
differed slightly (maximum 0.321 native pixels). The controlled old/new storage
comparison uses identical coordinates. The prior production-OpenCV replay had
already reproduced the duplicate at frame637.

Outputs are in the dataset's
`analysis_outputs/aruco_lossless_patch_cam03/`: `validation.png`, `summary.json`,
`id75_by_frame.csv`, `duplicate_detections.csv`, and separate raw, panorama and
tracking test files. The `aruco_lossless_draft_*` directories are earlier tests.

## Reproduce

From the repository root, in the local `ants` environment:

```bash
python -m pytest tracking/colony/test_aruco_lossless.py -q
python analysis/exploratory/validate_aruco_lossless_camera.py \
  --dataset /home/sam-reiter/bucket/ReiterU/Ants/basler/20260810/block02-w000-031 \
  --output /path/to/a/new/test-output-directory
```

The validator refuses an existing output directory. Regression tests also run
both detector CLIs on synthetic videos with repeated IDs using one and two MP
workers, including an underestimated frame-count hint, and cover empty outputs,
trailing empty frames, standalone tables, legacy loading and lossless fallback.
They also run actual mapping and tracking CLIs with non-identity homographies,
within/across-camera duplicates, same-ID ants in opposite colonies, a tag gap,
and a camera handoff. Native-only, pandas-only and dual-file inputs are tested.
Failed writes must preserve existing output, and missing PyTables must not lose
native records.

## Relevant history (before this patch)

- `ad05b18`, 2026-06-09 11:17 JST, author **Machiroi**: added the active MP
  exporter with the dense overwrite. That file had no later committed edits.
- `4587a6d`, 2026-06-09 13:48 JST, author **samRI**, “good tracking!”: added the
  existing tracker's nearest-previous-position duplicate selection.
- `50b376d`, 2026-06-11, author **samRI**: refined tracker matching.
- Latest commit under `aruco_detection/`: `a829b20`, 2026-06-23, author **samRI**;
  this added a grid PNG, not a detector/export fix. That directory is not the
  active OpenCV MP exporter used by this dataset.

Git records commit attribution; it does not establish who physically typed the
changes. The earlier tracking fix remains present, but could not recover raw
detections already lost during export.
