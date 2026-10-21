# print_test

Scores printed Custom_A tag sheets in video, to compare prints (e.g. the black ink-spread
compensation `--black-shrink-mm` of `aruco_detection/custom_dicts/generate_cuttable_sheet.py`).
Results of the 2026-10-20 eufyMake E1 test are in `aruco_detection/custom_dicts/README.md`; the
logs and per-tag rates (no videos) are in `/bucket/ReiterU/Ants/trials/tests_misc/ink_spread_test_20261020/`.

## Recording that works: arena crossover

Tape two sheets to one holder, put it where two arena cameras see one sheet each, record ~10 s
still, turn the holder 180 deg, record ~10 s still. Each camera then sees each sheet once, so
camera and position differences cancel. Keep hands out of view during the still parts.

Handheld phone video does not work: the phone focuses on one sheet, and which sheet is in focus
moved the rates by 0.1-0.4, far more than the print difference.

## Steps

Run from the repo root with the venv Python. The scripts use multiprocessing, so run them as
files (a `python -` stdin script hangs on Windows).

```powershell
$P = "makoto_play\print_test"; $OUT = "D:\print_test\arena_20261020_1902"
# 1. which cameras see the sheets
.\.venv\Scripts\python.exe $P\survey_arena_cams.py $OUT "Y:/cam*_2026-10-20-19-*.mkv"
# 2. sheet poses once a second (find the still phases) and footers ("Black shrink: ...") cut out upright
.\.venv\Scripts\python.exe $P\cut_footers.py $OUT 2,20 cam19=Y:/cam19_...mkv cam20=Y:/cam20_...mkv
# 3. write configs\<name>.json from 2, then score
.\.venv\Scripts\python.exe $P\score_arena_crossover.py $OUT $P\configs\arena_20261020_1902.json
```

A config names the two videos, a search window in seconds for each still phase, and which sheet
fills each camera in each phase (from the footers):

```json
{"cams": {"cam19": "Y:/cam19_....mkv", "cam20": "Y:/cam20_....mkv"},
 "phases": {"A": [0, 5], "C": [16, 24]},
 "main": {"cam19:A": "0.08mm", "cam19:C": "0.05mm", "cam20:A": "0.05mm", "cam20:C": "0.08mm"}}
```

Phone video: `score_phone_video.py VIDEO OUT_DIR --left-label 0.03mm --right-label 0mm`.

Every script writes `<script>_log_<timestamp>.txt` into OUT_DIR. The reads are cached there as
`<video stem>_detections.pkl` (with the dictionary name); delete it after changing the detector.
The frame rate is read from each video.

## How it scores

- `sheet_grid.py`: the production detector settings (`run_aruco_mp.py`) with 1-bit correction
  (`prod_ecr1`, as production) and without (`prod_ecr0`). A sheet is a homography from each read
  ID's print-grid position (tag pitch 3.3 x 3.9 mm) to the image, fitted with RANSAC; reads within
  0.8 mm of their predicted centre belong to the sheet.
- Still phase: the longest run of fitted frames whose main-sheet pose (the largest fit) stays
  within 3 px and 1 deg, with no gap over 1 s. Check the main-sheet centre it prints against
  `cut_footers.py`. Tags are placed with the phase's median homography, refined on a frame's own
  reads when at least 8 match, so frames whose own fit failed still count their misses. Dropping
  those frames instead hides a sheet's worst frames and biased the first phone analysis
  (0.715 vs 0.577 became 0.549 vs 0.557).
- Only tags whose whole marker is in the frame count. A tag-frame whose mean grey level is 25
  levels off that tag's phase median (a hand above it) is dropped.
- Wrong ID: a read on no sheet's grid that sits (within 1.2 mm) on a counted tag of this sheet with
  another ID. The tag's own ID read 0.8-1.2 mm off is reported as `off-pos` and counted as a miss.
- Reported: rate (tags read / tags in view), tags read in >= 99 % / >= 90 % of frames, wrong IDs
  as a share of all reads, and the same tag on both sheets in each camera.
