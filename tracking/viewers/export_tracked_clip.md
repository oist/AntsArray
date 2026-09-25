# Export an identified SLEAP clip

`export_tracked_clip.py` renders a camera's video with cyan body landmarks and
segments, orange antenna landmarks and segments, white numeric ArUco track IDs,
and a calibrated scale bar. It adds no title, timestamp, legend, or audio.

Run with the repository's Python dependencies installed and `ffmpeg` / `ffprobe`
on `PATH`. For the July 24 camera 04 example:

```bash
python tracking/viewers/export_tracked_clip.py \
  --block /bucket/ReiterU/Ants/basler/20260724/block01 \
  --video /bucket/ReiterU/Ants/basler/20260724/block01/cam04_cam3_2026-07-24-09-30-55.mkv \
  --homographies /bucket/ReiterU/Ants/basler/cameraArray_calib/20260623_calib_elevated_by_2mm_from_arenafloor/frame0/aruco_stitch/aruco_H_mats.npz \
  --start-frame 7200 --seconds 10 --width 2012 \
  --mm-per-pixel 0.016 --scale-bar-mm 1 \
  --output cam04_sleap_aruco_10s.mp4
```

Camera numbering uses the first one-based `camNN` filename prefix: `cam04_cam3`
uses homography index 3. The second camera number is a capture-stream identifier.
The exporter expects constant-rate video and zero-based frame indices shared
with the detections. It uses FFmpeg's accurate time seeking for long MKVs whose
missing duration index prevents OpenCV frame seeking.

Poses and identities come from `tracks/*_chunkNNN_{left,right}.parquet`, with
chunk-local frames translated by `--frames-per-chunk` (default 43200). Only
landmarks assigned to this camera's `SleapCam` are rendered; a same-camera ArUco
measurement can supply an ID when a pose is missing. Coordinates are projected
back from the panorama using the inverse homography. Antennal branches connect
to the occiput (node 1). Missing detections are not filled by the exporter.

The scale bar represents a horizontal camera-image segment whose projected
length equals the requested physical distance under the supplied homography
and panorama millimeters per pixel. It therefore also accounts for output
resizing. Supply the calibration and physical scale used to produce the tracks.

The MP4 retains the source frame rate and full field of view. A JSON sidecar
records sources, calibration, colors, interval, and detection counts; JPEGs
show the first, middle, and final frames. Add `--preview-only` to render only the
first JPEG and its one-frame sidecar before encoding the full clip.
