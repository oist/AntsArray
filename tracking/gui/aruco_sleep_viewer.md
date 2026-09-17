# ArUco/SLEAP Sleep Classifier Viewer

This is the focused interface for tuning sleep classification against video. It
shows only ArUco IDs, SLEAP skeletons, and the current cached-motion sleep label.

## Launch For 0723

```bash
/home/sam-reiter/miniforge3/envs/ants/bin/python tracking/gui/aruco_sleep_viewer.py \
  --block-dir /home/sam-reiter/bucket/ReiterU/Ants/basler/20260723/block02 \
  --video /home/sam-reiter/bucket/ReiterU/Ants/basler/20260723/block02/cam04_cam3_2026-07-23-19-31-09.mkv \
  --side auto
```

The viewer reads `data/*_aruco_tracks.h5`, `data/*_sleap_data.h5`, and
`stitched/sleep_motion/per_track/*`. It never recomputes bodypoint motion. For
this dataset, `auto` side selection uses the nearest preceding camera-array
homography and `stitched/grid_bounds_from_tracks.json`.

## Classification Rule

Body points 0-3 are the ArUco anchor, occiput, petiole, and gaster tip. Antenna
points 4-9 are treated separately. A valid frame is low-motion only when:

```text
body group speed <= body threshold
and
antenna group speed <= antenna threshold
```

The body threshold is strict. The antenna threshold is higher, so an otherwise
stationary ant can have slow antenna movement. The ant is labeled `SLEEP` when
the required fraction of valid frames in the complete trailing history window
are low-motion. Reducing the required low-motion fraction allows occasional
movement without ending sleep.

The current defaults are a `0.5 mm/s` body threshold, `0.7 mm/s` antenna
threshold, and a 10-second trailing history.

Group speed is a configurable percentile across the points in that anatomical
group. Point speeds spanning gaps larger than `Maximum point gap` and speeds
above `Jump cutoff` are excluded as unreliable tracking evidence.

Use `Apply` after changing parameters. The overlay reports instantaneous body
speed (`B`), instantaneous antenna speed (`A`), and the trailing low-motion
fraction (`q`). The labels are `SLEEP`, `WAKE`, `WARMUP`, `NO DATA`, and `SIDE?`.
