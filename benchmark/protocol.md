# ArUco Detection Benchmark Protocol

## Objective

Identify what limits ArUco detection performance on ant-mounted markers and
find optimal parameters for reliable detection.

Two parallel investigation tracks:

- **Track A** — Physical tag size (1.5, 2.0, 2.5 mm)
- **Track B** — OpenCV detector parameter tuning

---

## Track A — Tag Size Experiment

### Materials

- Tag sheet generated with `generate_tag_sheet.py` at 1200 DPI
- Laser printer with matte paper (avoid glossy — causes specular reflections)
- Same adhesive mounting method as current tags

### Procedure

1. **Generate tag sheet:**
   ```bash
   python benchmark/generate_tag_sheet.py --dpi 1200 --sizes 1.5,2.0,2.5 --ids 0-29
   ```

2. **Print & verify:** Print at "actual size" / 100% scale. Measure the 10 mm
   ruler bar on the printout — it must be exactly 10 mm. If not, adjust
   printer scaling.

3. **Cut tags:** Separate tags by size group. Each ant gets one tag of one
   size. Use at least 5 ants per size group for statistical power.

4. **Film:** Identical camera, lighting, and arena conditions for all groups.
   Minimum 10 minutes per group. Record which ant ID maps to which size.

5. **Run detection (baseline):**
   ```bash
   python run_aruco.py --video-file <video> --output-path results/ --dictionary-size 300
   ```

6. **Run detection (optimized):** Use best parameters from Track B.

7. **Compare metrics:**
   - Detection rate = frames with detection / total frames, per ant
   - ID stability = fraction of frames with correct ID (no swaps)
   - Position jitter = std of detected center across 5-frame windows (stationary ants)

---

## Track B — Parameter Tuning

### Setup

Select 2–3 cameras with varying lighting conditions from existing recordings.

### Procedure

1. **One-at-a-time sweep (start here):**
   ```bash
   python benchmark/aruco_benchmark.py \
       --video <path_to_video> \
       --sample-frames 100 \
       --sweep-mode one-at-a-time \
       --output-dir benchmark/results_oat/
   ```
   This varies each parameter independently from the current baseline (~30
   configurations). Produces CSV + per-parameter plots.

2. **Identify top parameters:** Check `benchmark_results.csv` and the plots.
   Look for parameters where changing the value significantly increases
   `mean_detections_per_frame` with minimal increase in `rejected_candidates`.

3. **Focused full sweep:** Create a JSON config with only the top 2–3 values
   per parameter and run:
   ```bash
   python benchmark/aruco_benchmark.py \
       --video <path_to_video> \
       --sample-frames 100 \
       --sweep-mode full \
       --sweep-config focused_grid.json \
       --output-dir benchmark/results_full/
   ```

4. **Validate:** Run the best configuration on a different camera / time
   period to check generalization.

---

## Expected High-Impact Parameters

| Priority | Parameter                  | Why                                                                 |
|----------|----------------------------|---------------------------------------------------------------------|
| 1        | `minMarkerPerimeterRate`   | Default 0.03 may filter out valid tags (perimeter threshold ~360 px for 4K, tags are ~300 px) |
| 2        | CLAHE preprocessing        | Dark tags on dark ant bodies — low local contrast                   |
| 3        | `cornerRefinementMethod`   | APRILTAG method may handle small markers better than CONTOUR        |
| 4        | `adaptiveThreshWinSizeMin` | Smaller windows (3–5) may better resolve small features             |
| 5        | `errorCorrectionRate`      | High rate (1.0) may cause false IDs; lower values improve precision |

---

## Output Files

```
benchmark/
├── tag_sheet.png                    # Printable multi-size tag sheet
├── results_oat/                     # One-at-a-time sweep results
│   ├── benchmark_results.csv
│   └── sweep_*.png                  # Per-parameter plots
├── results_full/                    # Full sweep results
│   ├── benchmark_results.csv
│   └── sweep_*.png
└── protocol.md                      # This file
```
