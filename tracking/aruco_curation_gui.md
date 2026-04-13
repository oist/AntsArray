# ArUco Curation GUI

Manual correction tool for `*_aruco_detections.csv` before downstream mapping and tracking.

## Launch

```bash
python tracking/aruco_curation_gui.py \
  --video /home/sam-reiter/bucket/ReiterU/Ants/basler/20251117_2_stim/data/cam01_cam0_2025-11-17-04-57-25_000.avi \
  --detections /home/sam-reiter/bucket/ReiterU/Ants/basler/20251117_2_stim/data/cam01_cam0_2025-11-17-04-57-25_000_aruco_detections.csv
```

If `--detections` is omitted, the tool tries to infer `<video_stem>_aruco_detections.csv`.
If you launch with no arguments, file-pickers are shown.
Shortcut help is printed to the terminal when the GUI starts.
Default playback is 20 FPS. Override it with `--fps` if you want a slower or faster review speed.

## Main Actions

- Click a detection to select it.
- Drag a selected detection to move its location.
- Toggle `Add/Update Mode`, set a tag ID, then click to place or replace that tag in the current frame.
- Use `Relabel Selected` to change the selected detection to the tag ID in the tag box.
- Use `Delete Selected` to remove the selected detection.
- Use the tag navigation buttons to jump to the next or previous frame where a tag is present or missing.
- The GUI auto-loads the corresponding `*_sleap_data.csv` sidecar in the background when it exists.
- Use `Bridge Tag (SLEAP NN)` to bridge the current frame for the tag in the tag box, show that filled result immediately, then keep bridging that tag during playback until `Play/Pause` is pressed again.
- Use `Bridge All Tags (SLEAP NN)` to do the same thing for every bridgeable missing tag on the current frame and on subsequent playback frames.
- Use `Preview All Frames` to bridge every bridgeable gap in the whole chunk without playing video.
- Use the `Range` boxes plus `Preview Range` to preview bridging over a selectable frame subset without playback.
- Use `Back Bridge` to undo the most recent grouped bridge command.
- The SLEAP bridge flow is non-blocking and does not open a popup window.
- Use the bottom transport bar for `<<`, `<`, `Play/Pause`, `>`, and `>>`.
- The right-side panel has a `Current Frame Detections` tab and a `Tag Trajectory` tab. The trajectory tab shows the selected tag ID from the tag box as an XY path with color encoding time from early to late, supports a zoom control, and clicking the path seeks the main video to the nearest plotted frame.

## Save Output

`Save CSV + Log` writes files into `<detections_dir>/curation` by default:

- `<base>_aruco_detections_curated.csv`
- `<base>_aruco_edits.json`

`Export Dense H5` separately writes:

- `<base>_aruco_tracks_curated.h5`

The dense H5 export contains:

- `aruco_tracks`
- `aruco_confidences`

It is intended to be compatible with the existing ArUco pipeline inputs, but it is heavier and slower than saving the curated CSV and edit log.

## Shortcuts

- `Left` / `Right`: previous or next frame
- `Shift+Left` / `Shift+Right`: jump by 10 frames
- `Space`: play or pause
- `A`: toggle add/update mode
- `Delete`: delete selected detection
- `Ctrl+Z` / `Ctrl+Y`: undo or redo
- `Ctrl+S`: save curated CSV + edit log
