# custom_dicts

Custom 4×4 ArUco dictionaries (A=100 markers, B=300 markers) and the scripts that produced them.

## Files

- `create_custom_4x4_dicts.py` — searches the full 2^16 pattern space and emits the dictionary `.npz` / `.json` plus contact sheets.
- `generate_cuttable_sheet.py` — turns an `.npz` into a printable, cuttable PNG (and matching SVG).
- `ink_spread.py` — black ink-spread pre-compensation applied to each rendered marker.
- `custom_4x4_{A100_d4,B300_d3}_*.npz` / `.json` — dictionary payloads.
- `custom_4x4_{A100_d4,B300_d3}_cuttable.{png,svg}` — print-ready sheets with crop marks. They
  predate ink-spread compensation: 600 dpi, no shrink, anti-aliased grey text.

## Reproduce the cuttable sheets

Output is deterministic. The committed sheets above came from the commands below before the
2540 dpi / black-shrink defaults; add `--dpi 600 --black-shrink-mm 0` for their geometry.

```powershell
# Dict A (100 markers)
.\.venv\Scripts\python.exe aruco_detection\custom_dicts\generate_cuttable_sheet.py `
    --npz aruco_detection\custom_dicts\custom_4x4_A100_d4_20260410_103938.npz `
    --output aruco_detection\custom_dicts\custom_4x4_A100_d4_cuttable.png `
    --name Custom_A

# Dict B (300 markers)
.\.venv\Scripts\python.exe aruco_detection\custom_dicts\generate_cuttable_sheet.py `
    --npz aruco_detection\custom_dicts\custom_4x4_B300_d3_20260410_103938.npz `
    --output aruco_detection\custom_dicts\custom_4x4_B300_d3_cuttable.png `
    --name Custom_B
```

A matching `.svg` is written next to the `.png` automatically (pass `--no-svg` to skip).

Defaults: `--marker-mm 1.5 --margin-mm 0.3 --dpi 2540 --black-shrink-mm 0.03 --cols 10`.
Each run also writes `generate_cuttable_sheet_log_<YYYYMMDD_HHMMSS>.txt` next to the output.

## Ink-spread compensation (UV printer)

On the eufyMake E1 (3 white layers + 1 black layer) the black spreads slightly, so white modules
print smaller than designed. `--black-shrink-mm d` moves every black edge inward by `d` before
printing (a black line gets `2d` thinner); the spread then brings it back to the design.

- Applied right after `generateImageMarker`, to each marker's whole black area at once: no seams
  between neighbouring black modules. IDs, bits, module positions, margins, tag size (2.1 mm) and
  sheet size do not change; only black turns white.
- Exact disc erosion: a pixel stays black only if its centre is at least `d` from all white. Straight
  edges move `d` to the nearest pixel (a half rounds down); white corners get radius `d`, which the spread squares off again.
- Why 2540 dpi: 1 px = 0.01 mm, so 0.25 mm modules and 0.03/0.05/0.08 mm shrinks are whole pixels.
  At 600 dpi, 0.03 and 0.05 mm both round to one 0.042 mm pixel (the script warns when a shrink
  lands more than 0.005 mm off). The DPI is stored in the PNG (`pHYs`), so DPI-aware software
  prints it at the right size; check against the 10 mm ruler on the sheet.
- The PNG holds only black (0) and white (255): text and crop marks are drawn without
  anti-aliasing, so no grey halftone is printed. The SVG draws the same shrink as vectors.
- Every PNG is read back with OpenCV (default parameters, `errorCorrectionRate=0`) and the script
  warns about tags that do not decode as their own ID. This checks the design, before any spread.
  Dict A: 100/100 decode at 0.03 and 0.05 mm; at 0.08 mm only 17/100 do, because the border and
  lone black modules are down to 0.09 mm. Those tags only read if the printer really spreads
  that far, and the E1 does not (print test below).
- The footer prints the shrink (`Black shrink: 0.030mm`), so printed test sheets identify themselves.

Comparison set (one sheet per shrink, plus a zoomed before/after preview):

```powershell
.\.venv\Scripts\python.exe aruco_detection\custom_dicts\generate_cuttable_sheet.py `
    --npz aruco_detection\custom_dicts\custom_4x4_A100_d4_20260410_103938.npz `
    --output D:\uv_test\A100.png --compare-shrink
```

This writes `A100_shrink{0,0.03,0.05,0.08}mm.{png,svg}`, `A100_shrink_preview.png` (red = black
removed, blue grid = design modules) and a table of the shrink in px and mm, the width of a lone
black/white module, and how many tags decode. Any other values: `--black-shrink-mm 0 0.04 0.06`.

### Print test 2026-10-20 (eufyMake E1, arena cameras)

Custom_A sheets printed at 0, 0.03, 0.05 and 0.08 mm (film, 3 white + 1 black layer) were taped two
at a time to one holder seen by cam19 and cam20 (backlit, ~56 px markers), recorded still, turned
180 deg by hand and recorded still again, so each camera saw each sheet once. Detector settings as
in production (`detection_pipeline/scripts/run_aruco_mp.py`); rate = tags read / tags in view over
the still frames, mean of the two cameras:

| black shrink | rate, ecr=1 (production) | rate, ecr=0 | tags read in >= 99 % of frames (cam19 / cam20) | wrong IDs among reads, ecr=1 |
|---|---|---|---|---|
| 0 mm | 0.951 | 0.777 | 71 / 77 | 0.06 % |
| **0.03 mm** | **0.992** | **0.868** | **88 / 96** | **0.01 %** |
| 0.05 mm | 0.912 | 0.602 | 69 / 64 | 0.22 % |
| 0.08 mm | 0.355 | 0.116 | 4 / 5 | 15.5 % (6.5 % at ecr=0) |

- 0.03 mm is therefore the default. Within each recording the better sheet won in both cameras
  and on most tags (0.03 vs 0 mm: 34:16 and 29:8 tags; 0.05 vs 0.08 mm: 95:1 and 93:2).
- Shrinking too much costs more than the spread: 0.05 mm already reads worse than no shrink, and
  0.08 mm reads wrong IDs even without bit correction. Do not print it.
- One print per shrink, and 0 / 0.03 mm and 0.05 / 0.08 mm were separate recordings, so 0.03 vs
  0.05 mm is compared across recordings. A 0.02 / 0.03 / 0.04 mm crossover would narrow the optimum.
- Handheld phone videos could not separate the sheets: which sheet the phone focused on moved the
  rates by 0.1-0.4. Use fixed arena cameras.
- Print scale: with a 50 mm width setting the printed 10 mm ruler measured 10 mm, although the
  image is 51.8 mm wide. Set the print width by the printed ruler, not by the image size.
- Scripts and how to repeat the test: `makoto_play/print_test/`. Logs and per-tag rates (videos not
  kept): `/bucket/ReiterU/Ants/trials/tests_misc/ink_spread_test_20261020/`
  (`Z:\ReiterU\Ants\trials\tests_misc\ink_spread_test_20261020` on the lab Windows PCs).

## Re-tagging sheets (only some IDs)

`--ids "3,13,57-59"` prints just those IDs, `--copies N` repeats each one N times in a row
(columns default to ~10, a multiple of N), `--title` labels the sheet. Without these options the
output is the full sheet above, unchanged. `aruco_detection/tag_roster/` finds which IDs a block's
colonies are missing and writes these sheets for you.

## Rebuild the dictionaries from scratch

Slow (full 2^16 search + simulated annealing). Only needed if changing `--dict-a-count`, `--dict-b-count`, or distance constraints.

```powershell
.\.venv\Scripts\python.exe aruco_detection\custom_dicts\create_custom_4x4_dicts.py `
    --dict-a-count 100 --dict-b-count 300 --joint --algorithm full
```

Output defaults to this folder.
