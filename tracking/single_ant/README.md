# Single-ant SLEAP pipeline

The current entry point is `pipeline.py`, with cluster orchestration in
`submit.py`. It uses **SLEAP only**, one ant identity per camera, camera-local
pixels, and the masks in `ROOT/arena_seg`. No tags, ArUco files, panorama
calibration, or cross-camera association are required. `build_tracks.py` is the
legacy CSV/ArUco workflow; new runs do not call it or the generic colony stitcher.

## Input and time contract

Each immediate `YYYYMMDD-HHMMSS` folder contains original `camXX_camN_*.avi`
recordings and existing consecutive video chunks under `data/`. The optional
`*_frame_counts.csv` sidecars are cross-checked against video headers. Missing
chunks, inconsistent counts/FPS, ambiguous or empty masks, and overlapping
recordings stop preparation. Masks must match full video dimensions. Global
overview cameras are excluded because they do not represent individual ants.

The timestamp in each **camera filename** defines its recording start; the
folder timestamp can differ by seconds. The default recording timezone is
`Asia/Tokyo`, explicitly recorded in the manifest. Start times have only the
one-second precision available in filenames. Timing within a recording uses
the actual video frame count and FPS, assuming a constant frame rate. There are
no recorder hardware timestamps for reconstructing dropped acquisition frames.

`Frame` is an `int64` index on a shared experiment timeline. `Timestamp` is UTC,
`ElapsedSeconds` is relative to the earliest camera start, `FrameInRecording`
indexes the original recording, and `FrameLocal` indexes the source chunk.
Sparse detections, empty chunks, recording pauses, and missing camera sessions
never shorten the timeline. Recording intervals are half open, with end time
one frame period after the last frame. `gap_before_seconds` measures the gap
from the previous recording's end for that camera. All cameras must share FPS.

The camera number is the stable numeric `Track` identity. This assumes the same
ant remains in that camera's arena across the input sessions. The pipeline does
not infer swaps or match identities between cameras.

## Inference and tracking

Pass the model pair and TensorRT export used by the current colony run
explicitly; do not infer "latest" from example paths in old documentation.
Preparation records checkpoint SHA-256 hashes, inference settings, mask hashes,
and a run contract hash. New predictions go into a new output directory, so old
CSV/H5 files are never accidentally reused. Successful GPU tasks run the same
`sleap-nn predict` invocation and SLP-to-H5 converter as the colony pipeline,
then stamp the contract and frame count into the H5. They publish a completion
receipt only after inference, conversion, and upload finish successfully.

For each frame, tracking builds candidates from complete SLEAP instances. The
default anchor is the occiput, bodypoint 1 in the current ten-point skeleton;
bodypoint 0 is the model's tag location, and is not required. If the anchor is
missing, the median of valid nodes provides a fallback. Candidates need at
least two nodes with score >= 0.2 and an anchor inside the arena mask. A
distance/confidence score selects one whole skeleton per frame. Confidence is
the sum of valid node scores divided by the full skeleton's node count, so a
few isolated confident points cannot outrank a complete well-supported pose. A displacement
gate of 100 pixels per elapsed frame applies while the previous detection is
less than five seconds old. After a longer gap, acquisition restarts from
confidence. These settings are configurable and recorded in the manifest.

For untagged ants, validate the colony model on representative clips before a
large run: its centroid detector can miss visible ants at the colony cutoff.
`tune_export.py adjust --source EXPORT --target NEW_EXPORT --centroid-threshold 0.01`
creates a separate ONNX variant with only the two centroid cutoff constants
changed. It verifies that learned weights and the bodypoint cutoff are
unchanged, checks the graph, and records hashes and changes. Test the variant
before selecting it; lower cutoffs admit more candidates. Then compile on an
A100 with `tune_export.py build-trt --target NEW_EXPORT --height 3036 --width 4024`.
This engine supports batches 1–8 at the specified fixed image dimensions. Keep
the new export path and effective cutoffs in the production run contract.
Pass its `export_metadata.json` to `prepare --export-metadata PATH`; add
`--build-trt-if-missing` to compile the validated graph lazily in a GPU worker.
`submit.py run --prerequisite-job JOBID,...` can wait for successful export and
GPU sample validation before submitting production arrays.
Failed prerequisites stop the coordinator even when dependent jobs remain queued;
missing accounting records are never treated as successful validation.
Scheduler connection outages and query timeouts are retried until service
returns, without resubmitting jobs or treating an unavailable queue as empty.
These utilities require the colony `sleap-nn` environment; ordinary tracking
does not depend on ONNX or TensorRT.

The selected instance retains all its bodypoint rows; low-confidence/nonfinite
nodes receive missing XY and `NodeValid=False`. Nodes are not independently
assigned to arenas. No interpolation fills missing detections or recording
gaps. Confidence, instance ID, anchor type, camera, recording, and chunk remain
available for QC. `TrackX/TrackY` give the selected anchor. Position state spans
chunks within a recording and resets at each new recording; identity remains
the camera number.

## Run on the clusters

Use a frozen copy of this repository in shared home storage and the same
absolute path on Deigo and Saion. First prepare on Deigo login:

```bash
PY=/apps/unit/ReiterU/ant_tracking/venv/bin/python
ROOT=/bucket/ReiterU/Ants/basler/single_ants/test_2
OUT=$ROOT/single_ant_260803_20260924
MODEL=/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20260803_models
EXPORT=/work/ReiterU/$USER/sleap_export/260803_topdown.centroid__260803_topdown.centered_instance__largegpu

$PY tracking/single_ant/pipeline.py prepare \
  --input-folder "$ROOT" --out-dir "$OUT" \
  --model-centroid "$MODEL/260803_topdown.centroid" \
  --model-instance "$MODEL/260803_topdown.centered_instance" \
  --export-dir "$EXPORT" --sleap-module sleap-nn/0.3.3
```

Preparation requires a fresh output directory. Resuming uses its existing
`manifest.json`; never edit that contract to change a model or timing. Validate
short clips before a large run (one representative recording per invocation):

```bash
COMMON=(--manifest "$OUT/manifest.json" --jobs "$HOME/ants_runs/single_ant_test2/jobs"
        --gpu-scratch "/work/ReiterU/$USER/single_ant_test2"
        --cpu-scratch "/flash/ReiterU/$USER/single_ant_test2")
$PY tracking/single_ant/submit.py smoke "${COMMON[@]}" --recording-index 0 --frames 240
```

After inspecting fresh smoke predictions, launch the durable coordinator:

```bash
nohup "$PY" -u tracking/single_ant/submit.py run "${COMMON[@]}" \
  > "$HOME/ants_runs/single_ant_test2/coordinator.log" 2>&1 < /dev/null &
```

The default is 32 concurrent `short-a100` GPU tasks, one video chunk per task,
8 CPUs/64 GB and a two-hour limit. Arrays contain at most 256 tasks to respect
submission limits. Incomplete tasks are retried up to three times. Use shorter
input chunks or adjust the generated resource policy when inference exceeds
two hours. CPU tracking starts after all fresh predictions are uploaded: one
worker per recording/camera, then one worker per camera for stitching. CPU
outputs go to flash and are published by the login-side coordinator. It checks
receipts and files, records job IDs, and writes `COMPLETE.json` only after final
publication/validation. A failed run has `jobs/FAILED.json`; inspect the named
worker logs, fix the cause, and rerun the same coordinator command to resume.
A filesystem lock prevents two coordinators managing the same jobs directory.

For local execution on completed fresh detections, use
`pipeline.py run --manifest PATH`; the `track`, `stitch`, and `finalize`
subcommands also support individual stages. `--work-output` redirects CPU
outputs to scratch without changing the detection input directory.

## Outputs and monitoring

```text
OUT/
  manifest.json                     frozen inputs, model and settings
  detections/<recording>/            fresh .slp, _sleap_data.h5, .done.json
  tracks/<recording>_<camera>.parquet
  coverage/<recording>_<camera>.parquet
  stitched/per_track/TrackID_<camera-number>.parquet
  stitched/track_manifest.parquet
  recordings.csv / recordings.parquet
  state/                            per-stage completion summaries and QC counts
  jobs.json                         published job ledger
  COMPLETE.json                     successful final validation
```

`coverage` stores compact half-open `[FrameStart, FrameStop)` intervals with
`Observed=True/False` for **every recorded frame**, including empty chunks.
Recording pauses have no recorded frames and are described by `recordings.csv`.
The original skeleton's `Bodypoint`, `X`, `Y`, `Score_node` columns and the
colony-compatible `Frame`, `Track`, `TrackX`, `TrackY` columns are retained.

Monitor `jobs/coordinator.json`, the coordinator log, and the per-task Slurm
logs. `squeue -u "$USER"` shows Deigo CPU work;
`ssh saion 'bash -lc "squeue -u $USER"'` shows GPU work.

Run the synthetic timing, mask, identity, missing-output, and stale-model tests:

```bash
python -m pytest tracking/single_ant/test_workflow.py tracking/single_ant/test_submit.py detection_pipeline/test_sleap2h5.py -q
```
