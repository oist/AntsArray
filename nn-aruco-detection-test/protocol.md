# NN ArUco Detection — Experiment Protocol

## Goal

Find NN-based ArUco detectors (DICT_4X4_1000) that outperform OpenCV's
built-in detector, especially for tags missed due to perspective distortion
near arena walls.

## Video Sources

| Name | Path | Setup | Resolution | FPS | Frames |
|------|------|-------|-----------|-----|--------|
| nest_cam04 | `Z:\...\20251020_1_30min_vibration\cam04_cam3_2025-10-20-13-46-00.avi` | 5x5 grid, dense nest | 4024x3036 | 24 | 225,672 |
| nest_cam05 | `Z:\...\20251020_1_30min_vibration\cam05_cam4_2025-10-20-13-46-00.avi` | 5x5 grid, dense nest | 4024x3036 | 24 | 225,672 |
| nest_cam09 | `Z:\...\20251020_1_30min_vibration\cam09_cam0_2025-10-20-13-46-01.avi` | 5x5 grid, dense nest | 4024x3036 | 24 | 225,672 |
| nest_cam10 | `Z:\...\20251020_1_30min_vibration\cam10_cam1_2025-10-20-13-46-01.avi` | 5x5 grid, dense nest | 4024x3036 | 24 | 225,672 |
| single_ant_* | `Z:\...\ARUCO_size_comparison_15_20_25\20260331_01\cam*.avi` | 1 ant/camera, 19 cams | 4024x3036 | 24 | 148,145 |

Pre-existing data in `data/` directory: chunk videos + ArUco CSV + SLEAP CSV
(chunks: `*_000.avi`, `*_001.avi`, `*_002.avi` per camera).

## Candidate Detectors

| Detector | Type | Status | Notes |
|----------|------|--------|-------|
| OpenCV | Classical | **Baseline** | 87-100% detection rate, ~14 FPS on 4K |
| YOLOv8/v11 | NN (fine-tuned) | Pending train | Two-phase: detect bbox + classify ID |
| RT-DETR | NN (transformer) | Pending train | No NMS, better multi-scale |
| DeepArUco++ | NN (purpose-built) | Pending setup | 3-stage pipeline, pre-trained available |

## Pipeline Phases

### Phase 1: Baseline Benchmark (OpenCV)
**Status: DONE** (2026-03-31)

```powershell
python nn-aruco-detection-test/benchmark_real.py `
    --video "Z:\ReiterU\Ants\basler\20251020_1_30min_vibration\cam*.avi" `
    --detectors opencv --export-details
```

**Results (200 frames/video, uniform sampling):**

| Video | Det | UIDs | Prec | Rec | F1 | FPS |
|-------|-----|------|------|-----|-----|-----|
| cam04 | 1597 | 61 | 1.000 | 1.000 | 1.000 | 14.7 |
| cam05 | 2294 | 60 | 1.000 | 1.000 | 1.000 | 14.5 |
| cam09 | 1452 | 61 | 1.000 | 1.000 | 1.000 | 8.9 |
| cam10 | 2005 | 62 | 1.000 | 1.000 | 1.000 | 13.4 |

**Note:** Precision/recall = 1.0 because OpenCV is both the detector AND
the ground truth here. The real value comes from comparing NN detectors
against this baseline — they should find additional detections that OpenCV
misses.

---

### Phase 2: Harvest Training Data
**Status: DONE** (2026-04-02)

```powershell
python nn-aruco-detection-test/harvest_training_data.py `
    --data-dir "Z:\ReiterU\Ants\basler\20251020_1_30min_vibration\data" `
    --max-chunks 3 --sample-every 10
```

**Results** (3 chunks from cam04, sample_every=10):

| Chunk | Classification crops | IDs | YOLO frames | Hard negatives |
|-------|---------------------|-----|-------------|----------------|
| _000  | 22,965              | 124 | 8,650       | 5,000          |
| _001  | 22,965              | 136 | 8,650       | 5,000          |
| _002  | 16,816              | 101 | 5,268       | 5,000          |
| **Total** | **62,746**      | **177** | **22,568** | **14,994**  |

**Outputs** in `nn-aruco-detection-test/training_data/`:
- `classification/` — 62,746 crops across 177 marker IDs
- `yolo/` — 22,568 full 4K frames + bbox labels
- `hard_negatives/` — 14,994 crops (SLEAP found ant, OpenCV missed tag)

---

### Phase 2b: Prep Training Data
**Status: DONE** (2026-04-02)

Resized 4K PNGs to 640px, created 80/20 train/val split.

```powershell
python nn-aruco-detection-test/prep_training.py --imgsz 640
```

| Split | Images | Size | Disk |
|-------|--------|------|------|
| train | 14,444 | 640x640 | 871 MB |
| val   | 8,124  | 640x640 | 490 MB |

---

### Phase 3: Train Models
**Status: DONE** (2026-04-02)

```powershell
python nn-aruco-detection-test/hybrid_pipeline.py train `
    --training-data-dir nn-aruco-detection-test/training_data `
    --epochs 100 --batch-size 16 --imgsz 640
```

**YOLO detector results** (RTX 5070 Ti, ~14 it/s, ~1.5 hr total):

| Metric | Value |
|--------|-------|
| mAP50 | 0.961 |
| mAP50-95 | 0.945 |
| Precision | 0.888 |
| Recall | 0.944 |
| Best epoch | ~73 |

Weights: `runs/detect/nn-aruco-detection-test/models/yolo_aruco/weights/best.pt`

**ResNet50 classifier results** (100 epochs):

| Metric | Value |
|--------|-------|
| Best val accuracy | 94.98% |
| Classes | 177 marker IDs |

Weights: `nn-aruco-detection-test/models/classifier_best.pth`
Class names: `nn-aruco-detection-test/models/classifier_classnames.npy`

**Problem found (2026-04-03):** YOLO got 0 TP in benchmark — two root causes:

1. **Scale mismatch:** Training used full 4K frames resized to 640px (markers ~12px).
   Inference tiles at 1280px (markers ~35-40px). YOLO learned to detect 12px blobs.
   *Confirmed:* disabling tiling → YOLO finds 11/12 markers matching OpenCV.
2. **Classifier framing mismatch:** Classifier trained on tightly-centered crops but
   inference gets YOLO bbox crops with different framing. Always predicted ID 987.

**Fix:** Re-harvest as 1280x1280 tiles (matching inference scale) + jittered classifier
crops. See Phase 3b below.

---

### Phase 3b: Re-train with Tiled Data (scale fix)
**Status: PENDING**

```powershell
# 1. Delete old training data
Remove-Item -Recurse nn-aruco-detection-test/training_data

# 2. Re-harvest as tiles (1280x1280)
python nn-aruco-detection-test/harvest_training_data.py `
    --data-dir "Z:\ReiterU\Ants\basler\20251020_1_30min_vibration\data" `
    --max-chunks 3 --sample-every 10

# 3. Create train/val split
python nn-aruco-detection-test/prep_training.py --imgsz 1280

# 4. Delete old caches
Remove-Item nn-aruco-detection-test\training_data\yolo\labels\train.cache -ErrorAction SilentlyContinue
Remove-Item nn-aruco-detection-test\training_data\yolo\labels\val.cache -ErrorAction SilentlyContinue

# 5. Re-train YOLO (imgsz=1280 to match tiles) + classifier
python nn-aruco-detection-test/hybrid_pipeline.py train `
    --training-data-dir nn-aruco-detection-test/training_data `
    --epochs 100 --batch-size 8 --imgsz 1280
```

#### RT-DETR detector
```powershell
yolo detect train data=nn-aruco-detection-test/training_data/yolo/data.yaml `
    model=rtdetr-l.pt epochs=100 imgsz=1280 cache=disk workers=4 `
    project=nn-aruco-detection-test/models name=rtdetr_aruco
```

#### Classifier (ResNet50 for marker ID)
Trained automatically during `hybrid_pipeline.py train`.

---

### Phase 4: YOLO+Classifier Benchmark (retrained with tiles)
**Status: DONE** (2026-04-03)

```powershell
python nn-aruco-detection-test/benchmark_real.py `
    --video "Z:\ReiterU\Ants\basler\20251020_1_30min_vibration\cam*.avi" `
    --detectors opencv yolo `
    --yolo-weights "runs/detect/nn-aruco-detection-test/models/yolo_aruco/weights/best.pt" `
    --classifier-weights "nn-aruco-detection-test/models/classifier_best.pth" `
    --class-names "nn-aruco-detection-test/models/classifier_classnames.npy" `
    --export-details
```

**Dense nest results (4 cameras, 200 frames each):**

| Metric | OpenCV | YOLO+Classifier |
|--------|--------|-----------------|
| Recall | 100% (by def.) | 97.1% |
| Precision | 100% | 69.8% |
| F1 | 100% | 81.2% |
| FPS | 4.9 | 5.7 |

YOLO detection is good (~97% recall) but classifier assigns many wrong IDs
(~30% FP). Root cause analysis:

**FP breakdown (cam04, 602 FPs):**
- 252 (42%): YOLO at right location, **classifier assigned wrong ID**
- 343 (57%): no OpenCV detection nearby (59% at frame edges — potential real recoveries)
- 7 (1%): location drift

**Classifier failure causes:**
1. Known OpenCV FP IDs (37, 190, 683) were in training data — 1,067 crops
   for ID 37 alone. Classifier learned to predict garbage IDs.
2. 103 of 177 classes had <10 training crops — severe imbalance.
3. Training crops (centered) didn't match inference crops (YOLO bbox framing).

---

### Phase 4b: Single-ant validation
**Status: DONE** (2026-04-06)

Tested on single-ant cameras where true ID is known (IDs 17, 25).

**YOLO location-only performance (ignoring classifier IDs):**

| Camera | True ID | OpenCV rate | YOLO finds ant | YOLO recovers misses |
|--------|---------|-------------|----------------|---------------------|
| cam01 | 25 | 179/200 (90%) | 164/179 (92%) | **14/14 (100%)** |
| cam12 | 25 | 140/200 (70%) | 140/140 (100%) | **22/22 (100%)** |
| cam17 | 17 | 182/200 (91%) | 171/182 (94%) | **4/4 (100%)** |

**Key finding:** YOLO recovers **100% of frames that OpenCV misses** (the
arena-wall perspective distortion failures from `benchmark/protocol.md`).
The detection stage works — the classifier is the bottleneck.

**Decision:** Replace the ResNet50 classifier with **OpenCV ArUco decoding**
on the YOLO bbox crop. OpenCV is perfect at decoding IDs from a good crop;
it just struggles to find markers under perspective distortion (which YOLO
handles).

---

### Phase 5: YOLO+OpenCV Hybrid Benchmark
**Status: DONE** (2026-04-06)

New hybrid approach: YOLO detects bbox → pad crop (50%) → OpenCV decodes
4x4 bit pattern. No NN classifier needed.

Detector: `yolo-hybrid` in `aruco_detection/nn_detection/yolo_opencv_hybrid.py`

#### Single-ant results (known true IDs)

```powershell
python nn-aruco-detection-test/benchmark_real.py `
    --video ".../cam01_cam0_*.avi" ".../cam12_cam3_*.avi" ".../cam17_cam8_*.avi" `
    --detectors opencv yolo-hybrid `
    --yolo-weights "runs/detect/nn-aruco-detection-test/models/yolo_aruco/weights/best.pt" `
    --export-details
```

| Camera | True ID | OpenCV det | Hybrid TP | Missed | Wrong ID | Undecoded (id=-1) |
|--------|---------|-----------|-----------|--------|----------|-------------------|
| cam01 | 25 | 200 | 171 | 29 | 9 | 77 (63 at real pos) |
| cam12 | 25 | 141 | **140** | **1** | 2 | 81 (57 at real pos) |
| cam17 | 17 | 198 | 175 | 23 | 4 | 66 (62 at real pos) |

**cam12 highlight:** OpenCV alone detects in 141/200 frames (70.5%).
Hybrid gets 140 TP + only 1 miss = **99.3% recall**. YOLO finds the marker
in all 23 frames that OpenCV misses — the arena-wall perspective distortion
cases identified in `benchmark/protocol.md`.

Wrong-ID FPs nearly eliminated: 2-9 per camera (vs 143-241 with NN classifier).

The undecoded (id=-1) detections are YOLO finding marker-like regions where
the padded crop wasn't clear enough for OpenCV bit-pattern decoding. These
can be safely filtered.

#### Dense nest results (4 cameras, ~60 ants)

```powershell
python nn-aruco-detection-test/benchmark_real.py `
    --video "Z:\ReiterU\Ants\basler\20251020_1_30min_vibration\cam*.avi" `
    --detectors opencv yolo-hybrid `
    --yolo-weights "runs/detect/nn-aruco-detection-test/models/yolo_aruco/weights/best.pt" `
    --export-details
```

| Camera | OpenCV | Hybrid TP | FP | Missed | Recall | Precision | F1 |
|--------|--------|-----------|-----|--------|--------|-----------|------|
| cam04 | 1597 | 1564 | 664 | 33 | 97.9% | 70.2% | 81.8% |
| cam05 | 2294 | 2231 | 1119 | 63 | 97.2% | 66.6% | 79.1% |
| cam09 | 1452 | 1412 | 725 | 40 | 97.2% | 66.1% | 78.7% |
| cam10 | 2005 | 1943 | 933 | 62 | 96.9% | 67.6% | 79.6% |
| **avg** | | | | | **97.3%** | **67.6%** | **79.8%** |

**FP breakdown (all 4 cameras):**

| Type | Count | Notes |
|------|-------|-------|
| Undecoded (id=-1) | 3231 | YOLO found region, OpenCV couldn't decode bit pattern |
|   — at real marker positions | 1425 | Near an OpenCV detection (crop quality issue) |
|   — no OpenCV nearby | 1806 | Arena features or markers OpenCV also can't see |
| Wrong ID | 210 | Real decoding errors |

**If undecoded (id=-1) detections are filtered out:**

| Metric | OpenCV | Hybrid (filtered) |
|--------|--------|--------------------|
| Recall | 100% (by def.) | **97.3%** |
| Precision | 100% | **~97.1%** |
| F1 | 100% | **~97.2%** |
| FPS | 4.2 | 3.0 |

Wrong-ID rate drops from 3089 (NN classifier) to **210** — a **15x reduction**.

---

---

### Phase 6: DeepArUco-PT (PyTorch reimplementation)
**Status: IN PROGRESS** (2026-04-07)

Reimplemented the DeepArUco++ concept in pure PyTorch for DICT_4X4_1000:
- Stage 1: YOLO detection (existing model)
- Stage 2: U-Net corner refinement (64x64 crop → 4 corner heatmaps → soft-argmax)
- Stage 3: CNN bit decoder (32x32 rectified → 16 logits → Hamming match vs 1000 IDs)
- Fallback: OpenCV decode when NN Hamming distance > 2

Detector: `deeparuco-pytorch` in `aruco_detection/nn_detection/deeparuco_pytorch.py`

#### Phase 6a: Synthetic training
**Status: DONE** (2026-04-07)

```powershell
# Generate corner refiner data (50K samples with real backgrounds)
python -m aruco_detection.nn_detection.training.datagen_corners `
    --output-dir nn-aruco-detection-test/training_data/corners `
    --background-dir benchmark/sample_frames --num-samples 50000

# Generate decoder data (100K samples)
python -m aruco_detection.nn_detection.training.datagen_decoder `
    --output-dir nn-aruco-detection-test/training_data/decoder --num-samples 100000

# Train corner refiner (U-Net, ~11M params)
python -m aruco_detection.nn_detection.training.train_corner_refiner `
    --data-dir nn-aruco-detection-test/training_data/corners `
    --output-dir nn-aruco-detection-test/models --epochs 100

# Train bit decoder (small CNN, ~200K params)
python -m aruco_detection.nn_detection.training.train_bit_decoder `
    --data-dir nn-aruco-detection-test/training_data/decoder `
    --output-dir nn-aruco-detection-test/models --epochs 50
```

#### Phase 6b: Initial benchmark (synthetic-only models)
**Status: DONE** (2026-04-07)

3-way comparison on `benchmark/sample_frames/` (single-ant cameras, 100 frames each):

| Detector | cam01 TP | cam09 TP | cam12 TP | Avg Prec | Avg Rec | Avg F1 |
|----------|---------|---------|---------|----------|---------|--------|
| **OpenCV** | 104 | 101 | 89 | 100% | 100% | 100% |
| **YOLO+OpenCV** | 92 | 98 | 85 | 62-75% | 88-97% | 73-85% |
| **DeepArUco-PT** | 4 | 12 | 14 | 2-8% | 4-16% | 3-11% |

**DeepArUco-PT diagnosis:**
- **Locations are correct** — 85-98% of detections are near OpenCV positions
- **Decoder assigns wrong IDs** — sim-to-real gap: synthetic training data
  (clean renders) doesn't prepare for noisy, perspective-distorted real crops

**Bug found and fixed:** `soft_argmax_2d` was applying softmax to raw U-Net
output (range ~[0, 1] from MSE training) — the probability mass spread
uniformly and all corners collapsed to image center (~31.5, 31.5).
Fix: scale raw heatmap by temperature=100 before softmax, skip sigmoid.
After fix: 0.0px error on ideal test crop.

#### Phase 6c: Fine-tuning on real crops
**Status: PENDING**

Fine-tune both models on real classification crops (62,746 crops, 177 IDs).
Uses OpenCV to detect corners on each crop as ground truth, then:
- Corner refiner: real 128x128 crop → resize to 64x64, GT = OpenCV corners
- Decoder: perspective-rectify with OpenCV corners → 32x32, GT = known bit pattern

```powershell
python -m aruco_detection.nn_detection.training.finetune_real `
    --cls-dir nn-aruco-detection-test/training_data/classification `
    --output-dir nn-aruco-detection-test/models `
    --corner-refiner-weights nn-aruco-detection-test/models/corner_refiner.pth `
    --decoder-weights nn-aruco-detection-test/models/bit_decoder.pth `
    --epochs 50
```

Benchmark with fine-tuned models:
```powershell
python nn-aruco-detection-test/benchmark_real.py `
    --image-dir "benchmark/sample_frames/cam01" "benchmark/sample_frames/cam05" `
               "benchmark/sample_frames/cam09" "benchmark/sample_frames/cam12" `
    --detectors opencv yolo-hybrid deeparuco-pytorch `
    --yolo-weights "runs/detect/nn-aruco-detection-test/models/yolo_aruco/weights/best.pt" `
    --corner-refiner-weights "nn-aruco-detection-test/models/corner_refiner_ft.pth" `
    --decoder-weights "nn-aruco-detection-test/models/bit_decoder_ft.pth" `
    --export-details
```

#### Training data quality note

Hard negatives (`training_data/hard_negatives/`) were checked: 81/14,994
(0.5%) contain detectable ArUco markers. These are cases where two ants are
close together — SLEAP detects both body points but ArUco only detects one
ant's tag. The second ant's crop captures the first ant's tag at the edge.
This contamination is low and doesn't affect DeepArUco-PT (which trains on
synthetic data + classification crops, not hard negatives).

---

---

### Phase 7: Fair benchmark — single-ant videos with true ground truth
**Status: DONE** (2026-04-07)

Previous benchmarks used OpenCV as ground truth, creating circular bias
(OpenCV always scores 100%). Single-ant videos have exactly one known
marker ID per camera — the true detection rate is simply "did the detector
find the correct ID in this frame?"

```powershell
python nn-aruco-detection-test/benchmark_single_ant.py `
    --video ".../cam01*.avi" ".../cam11*.avi" ".../cam12*.avi" ".../cam13*.avi" ".../cam17*.avi" `
    --detectors opencv yolo-hybrid `
    --yolo-weights "runs/detect/.../best.pt" --n-frames 500
```

**Results (500 frames/video, true ground truth):**

| Camera | True ID | Tag size | OpenCV | YOLO+Hybrid | Diff |
|--------|---------|----------|--------|-------------|------|
| cam01 | 25 | 2.0mm | **86.2%** | 84.6% | -1.6% |
| cam11 | 25 | 2.5mm | **92.2%** | 30.2% | -62.0% |
| cam12 | 25 | 1.5mm | 76.6% | 76.6% | 0.0% |
| cam13 | 17 | 1.5mm | **97.2%** | 95.4% | -1.8% |
| cam17 | 17 | 2.0mm | **91.6%** | 87.2% | -4.4% |
| **avg** | | | **88.8%** | **74.8%** | **-14.0%** |

**OpenCV wins on every camera** when measuring correct ID detection rate.

**cam11 anomaly:** YOLO produces 708 detections (1.4/frame) but only 151
have the correct ID (30.2%). YOLO finds things but OpenCV can't decode
ID 25 from the YOLO crops. The YOLO model was trained on cam04 nest video
data and doesn't generalise to cam11's different viewpoint/lighting.

**Key insight:** The earlier benchmark (Phase 4b) was misleading — "YOLO
found markers in 100% of missed frames" meant YOLO detected *something*,
not that it detected the *correct marker*. With true ground truth, the
hybrid never beats OpenCV on any camera.

---

### Phase 8: Decode Cascade, Whitelist, and SLEAP Rescue
**Status: DONE** (2026-04-08)

Three frame-level rescue strategies evaluated:

#### 8a: YOLO+Cascade decode (multi-strategy OpenCV)
Detector: `yolo-cascade` in `aruco_detection/nn_detection/yolo_cascade_hybrid.py`

Tries 3 paddings × 3 preprocessors × 2 OpenCV parameter profiles per
YOLO bbox (early exit on first decode).  "Aggressive" profile validated
via parameter sweep: CORNER_REFINE_APRILTAG, relativeCornerRefinmentWinSize=0.5,
perspectiveRemovePixelPerCell=8.

Dense nest: +39 TP, -39 misses vs hybrid (modest).
Single-ant (corrected labels): +5-6pp on hard cameras (cam01, cam06, cam12).

#### 8b: Session whitelist matcher
Tested offline on 1685 disagreement crops.  100% of crops had a quad
candidate (rejected or accepted), so the bottleneck is **not** quad
detection.  But soft whitelist only accepted 35/1685 (2.1%) — safe
(0 wrong IDs) but near-zero yield.  Root cause: most rejected quads
rectify to garbage (flat/empty images).

#### 8c: SLEAP-crop rescue
Three arms tested on 6674 rescue candidates (SLEAP ants with no nearby
OpenCV detection):

| Arm | Decoded | Rate |
|-----|---------|------|
| A: center crop | 591 | 8.9% |
| B: body-axis oriented | 546 | 8.2% |
| C: offset/scale sweep | 897 | 13.4% |

Cross-check verdict: **the 897 decodes are noise**.
- Tracklet consistency: 0/897 (no SLEAP instance decoded same ID twice)
- YOLO agreement: 0/897 (YOLO found nothing at those positions)
- Confidence buckets: HIGH=0, MEDIUM=0, LOW=152, NONE=745

**Conclusion:** Frame-level rescue strategies are exhausted.  The remaining
~8.5% misses are genuinely unreadable markers (face-down, occluded, blur).
The missing information has to come from **time**, not another frame-local model.

---

### Phase 9: Conservative Temporal Gap-Fill
**Status: DONE** (2026-04-08)

Conservative interior-gap filling along SLEAP tracklets.  Only fills
gaps flanked by the **same ID on both sides**, max 10 frames, never
corrects existing detections.

Module: `tracking/temporal_id.py` (integrated into `tracking_utils.py`
via `temporal_propagation=True`).

Key fix: SLEAP instance IDs are per-frame (not persistent), so the
tracklet builder uses nearest-neighbor spatial linking across frames.

#### Synthetic gap recovery (4 nest cameras, 5000 frames each)

| Drop rate | Recovery | Wrong fills | Precision |
|-----------|----------|-------------|-----------|
| 10% | 97.1% | 11 | 99.9% |
| 20% | 96.8% | 24 | 99.9% |
| 30% | 96.3% | 41 | 99.9% |
| 50% | 94.4% | 56 | 99.9% |

#### Natural gap fill (un-degraded data)

| Camera | Original | Gap-filled | Increase |
|--------|----------|------------|----------|
| cam04 | 42,347 | +2,867 | +6.8% |
| cam05 | 61,059 | +5,553 | +9.1% |
| cam09 | 36,201 | +5,347 | +14.8% |
| cam10 | 44,715 | +4,289 | +9.6% |

---

### Phase 10: Bit-Level Audit
**Status: DONE** (2026-04-08)

8 cameras in the single-ant benchmark had **wrong ground truth labels**.
Bit-level audit confirmed: the markers physically on the ants are
different IDs than assumed.  Corrected mapping in `benchmark_single_ant.py`.

| Camera | Was | Corrected to | Evidence |
|--------|-----|-------------|----------|
| cam02 | 25 | 17 | 29/29 frames |
| cam06 | 3 | 25 | 23/23 frames |
| cam07 | 17 | 25 | 27/28 frames |
| cam14 | 17 | 3 | 29/31 frames |
| cam15 | 17 | 3 | 28/30 frames |
| cam16 | 17 | 25 | 29/30 frames |
| cam18 | 25 | 17 | 29/29 frames |
| cam19 | 25 | 17 | 28/28 frames |

After correction, OpenCV average detection rate: **91.5%** (was 52.6%).

---

### Phase 11: OpenCV Parameter Sweep
**Status: DONE** (2026-04-08)

7 parameter profiles benchmarked in full-frame and YOLO crop-rescue modes.

Full-frame: baseline/conservative best (most detections, fastest).
Crop rescue: **aggressive** wins by +7pp (82.2% vs 75.2% decode rate).

Aggressive profile: CORNER_REFINE_APRILTAG, adaptiveThreshWinSizeMax=80,
relativeCornerRefinmentWinSize=0.5, perspectiveRemovePixelPerCell=8.

---

---

## Final Conclusions

### Production architecture

**OpenCV full-frame + conservative temporal gap-fill on SLEAP tracklets.**

- Frame-level: OpenCV at **91.5%** detection rate (19 cameras, corrected GT)
- Temporal gap-fill adds **+7-15% coverage** at **99.9% precision**
- Optional: YOLO+cascade on known hard cameras for +5-6pp frame-level
  (evaluate post-temporal to see if still needed)

### What we learned

1. **OpenCV is the best frame-level detector** for DICT_4X4_1000.
   On a fair benchmark with corrected ground truth, it beats all
   NN-based approaches on average.

2. **The remaining ~8.5% frame-level misses are genuinely unreadable.**
   No frame-level rescue strategy (cascade, whitelist, SLEAP crop,
   DeepArUco) reliably recovers them.  The markers are face-down,
   occluded, or too distorted for any decoder.

3. **Temporal gap-fill is the real remaining gain.** 97% recovery at
   99.9% precision.  This is a tracking-level solution, not a
   detection-level one.

4. **Ground truth quality gates everything.** 8/19 cameras had wrong
   labels, inflating apparent failure rates.  The bit-level audit
   script paid for itself immediately.

5. **SLEAP instance IDs are per-frame**, not persistent.  Tracklet
   building must use nearest-neighbor spatial linking, not instance
   ID grouping.

6. **Aggressive OpenCV parameters help crop rescue** (+7pp) but hurt
   full-frame detection.  Use two profiles: conservative for full-frame,
   aggressive for rescue crops only.

### What didn't work and why

| Approach | Result | Root cause |
|----------|--------|-----------|
| YOLO + ResNet50 classifier | 70% precision | Contaminated labels, class imbalance |
| YOLO + OpenCV hybrid | Equal or worse than OpenCV | Crops too distorted for decode |
| YOLO + cascade (multi-strategy) | +5-6pp on hard cameras | Modest; temporal fill may subsume it |
| DeepArUco-PT (all variants) | 2-7% F1 | Sim-to-real gap, fragile soft-argmax |
| Session whitelist matcher | 2.1% yield, 0 wrong | Near-zero recall; quads rectify to garbage |
| SLEAP crop rescue | 897 decodes, all noise | Spurious quads in background texture |
| Body-axis oriented crops | Worse than center crops | Body angle doesn't improve marker presentation |

### Remaining research directions (deprioritised)

- [ ] **Re-benchmark cascade post-temporal** — does it still help on hard
      cameras after gap-fill already recovers most interior misses?
- [ ] **Track-level metrics** — fraction of track labeled, ID switch rate,
      longest wrong-propagation run, time-to-first-lock
- [ ] **Train YOLO on diverse cameras** — if cascade is still needed,
      broader training data would fix viewpoint generalization
- [ ] **For future experiments: smaller ArUco dictionary** — DICT_4X4_1000
      has minimum Hamming distance 2; a custom dictionary with fewer IDs
      would be more robust

---

## Key Metrics

- **Detection rate**: % of frames where the correct marker ID is found
  (only meaningful with independent ground truth)
- **Track coverage**: % of SLEAP tracklet frames with a propagated ID
  (production metric after temporal fill)
- **Fill precision**: fraction of gap-filled frames with correct ID
- **False positive rate**: detections with wrong IDs
- **Speed (FPS)**: must be practical for 25 cameras × hours of video

## File Inventory

```
nn-aruco-detection-test/
  config.yaml              — video paths and settings
  protocol.md              — this file
  benchmark_real.py        — benchmark using OpenCV as GT (biased)
  benchmark_single_ant.py  — fair benchmark with known true IDs (corrected 2026-04-08)
  benchmark_temporal.py    — temporal gap-fill benchmark (synthetic + natural gaps)
  harvest_training_data.py — extract training crops from OpenCV detections
  hybrid_pipeline.py       — seed -> train -> recover pipeline
  prep_training.py         — resize images and create train/val split
  extract_corner_gt.py     — extract crops with OpenCV corners for GT review
  review_gui.py            — interactive GUI for filtering GT crops
  audit_confused_cameras.py — bit-level audit for wrong-ID cameras
  sweep_opencv_params.py   — OpenCV parameter profile sweep
  whitelist_experiment.py  — offline whitelist matcher evaluation
  sleap_rescue_ablation.py — SLEAP-crop rescue arm comparison
  rescue_crosscheck.py     — cross-check rescue decode precision
  build_disagreement_set.py — extract detector disagreement crops
  results/                 — benchmark CSVs, plots, frame details, logs
  training_data/           — harvested crops (classification + YOLO + hard negatives)
  corner_gt_review/        — human-verified GT crops with corners
  models/                  — trained weights (.pt, .pth, .npy)
  audit/                   — bit-level audit crops and summaries
  disagreement_set/        — detector disagreement crops + metadata

aruco_detection/nn_detection/
  base.py                  — abstract ArucoDetector interface
  opencv_baseline.py       — OpenCV detector wrapper
  yolo_detector.py         — YOLO + ResNet50 classifier
  yolo_opencv_hybrid.py    — YOLO detection + OpenCV ID decoding
  yolo_cascade_hybrid.py   — YOLO + multi-strategy OpenCV decode cascade
  yolo_warp_hybrid.py      — YOLO + contour corners + perspective warp + OpenCV
  deeparuco_pytorch.py     — YOLO + U-Net corners + CNN decoder (PyTorch)
  deeparuco_detector.py    — DeepArUco++ TF wrapper (not used — wrong dictionary)
  rtdetr_detector.py       — RT-DETR + classifier (not trained)
  dict_4x4_1000.py         — DICT_4X4_1000 bit patterns + Hamming matching + whitelist
  whitelist.py             — session whitelist load/discover utilities
  models/
    corner_refiner.py      — U-Net for corner heatmap prediction
    bit_decoder.py         — CNN for 4x4 bit pattern decoding
  training/
    datagen_corners.py     — synthetic corner refinement training data
    datagen_decoder.py     — synthetic decoder training data
    train_corner_refiner.py — corner refiner training script
    train_bit_decoder.py   — decoder training script
    finetune_real.py       — fine-tune on classification crops (contaminated labels)
    finetune_from_gt.py    — fine-tune on human-verified GT crops
  data/                    — synthetic data generation + augmentation
  utils/                   — tiling, NMS

tracking/
  temporal_id.py           — conservative temporal gap-fill on SLEAP tracklets
  tracking_utils.py        — ArUco + SLEAP track merging (temporal_propagation param)
```
