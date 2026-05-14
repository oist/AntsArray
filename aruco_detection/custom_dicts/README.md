# custom_dicts

Custom 4×4 ArUco dictionaries (A=100 markers, B=300 markers) and the scripts that produced them.

## Files

- `create_custom_4x4_dicts.py` — searches the full 2^16 pattern space and emits the dictionary `.npz` / `.json` plus contact sheets.
- `generate_cuttable_sheet.py` — turns an `.npz` into a printable, cuttable PNG (and matching SVG).
- `custom_4x4_{A100_d4,B300_d3}_*.npz` / `.json` — dictionary payloads.
- `custom_4x4_{A100_d4,B300_d3}_cuttable.{png,svg}` — print-ready sheets with crop marks.

## Reproduce the cuttable sheets

Output is deterministic (verified byte-identical for dict A on Python 3.14, cv2 4.11, numpy 2.4).

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

Defaults: `--marker-mm 1.5 --margin-mm 0.3 --dpi 600 --cols 10`.

## Rebuild the dictionaries from scratch

Slow (full 2^16 search + simulated annealing). Only needed if changing `--dict-a-count`, `--dict-b-count`, or distance constraints.

```powershell
.\.venv\Scripts\python.exe aruco_detection\custom_dicts\create_custom_4x4_dicts.py `
    --dict-a-count 100 --dict-b-count 300 --joint --algorithm full
```

Output defaults to this folder.
