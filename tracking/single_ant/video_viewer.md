# Single-ant cup video viewer

Open the original videos with the original full-posture S0/S1 assignments:

```bash
python -m tracking.single_ant.video_viewer DATASET --cup 30 --start-hours 0
```

The installed launcher for the October 27 block02 + October 28 block01 pair is
`/home/sam-reiter/ants_runs/single_ant_video_viewer_20261103/open_cup_video.sh`.
It uses an isolated environment containing PyAV 19.0.2 and the existing ant-analysis
environment's OpenCV, Pillow, NumPy and Tk packages.

The detection overlays are enabled by default:

- **ArUco IDs**: green crosses with the decoded tag ID and source camera. Repeated
  IDs remain separate detections. This reads lossless `aruco_detections`, including
  false IDs; it does not use the dense export that drops duplicate IDs.
- **Raw SLEAP**: magenta instances from the exact `_sleap_data.h5` tracking inputs.
  Partial poses are included even when their tag node is missing. Hollow raw nodes
  have score below 0.2; **Raw node min score** controls this display filter only.
  A score-zero tag anchor may be a centroid-filled estimate from the detector.
- **Tracked pose**: cyan coordinates from the final stitched tracking parquet.
  The yellow diamond is the tracker position. ArUco-only frames show no invented
  SLEAP skeleton. The summary shows the assigned ID, position source, source
  camera, node count and missing node names. Exported tracking poses have no node
  confidence field; the raw-score control applies to raw SLEAP only.
- **5 s track trail**: real tracked positions, with breaks at missing frames.
- **Node names**: names from the SLEAP model metadata. These models have no declared
  edges, so the display connects the anatomical body and antennal chains.
- **All camera detections**: display overlapping camera predictions together.
  By default, raw overlays follow the panorama's camera ownership, preventing
  duplicate views from obscuring the animal. Counts describe all loaded instances
  near the selected cup, including portions hidden by camera ownership.

**Show tracked camera** switches to the camera that supplied the selected pose
(or the ArUco source camera for tag-only frames). This helps inspect wall parallax
and compare the selected track with the raw prediction in its original view.
The existing **Camera view** menu also selects individual views. Hiding a layer
is labeled as off, separately from missing detections or unavailable input files.

Raw detections use the source manifests' chunk-local frame offsets and calibrated
homographies. Final tracks use their recorded output coordinate scale (0.016 mm
per tracking pixel in this dataset); the camera calibration has a separate scale.
Video and overlays update together after the same requested frame has loaded.
Detection reads run in the existing background worker. HDF5 frame indexes and
nearby parquet row groups are cached for playback; instances are not capped.
Changing overlays preserves the video position and playback state, and changes no
tracking or classifier outputs.

Select a cup using the numbered overview or the cup menu. The default cup panorama
uses the calibrated camera views needed to cover that cup. Individual camera views
are also available. Use the mouse wheel to zoom and drag to pan.

Controls:

- Play, Pause, Rewind and Fast forward; repeated rewind/fast-forward increases speed.
- Space toggles playback; J/K/L rewind/pause/play.
- Left/right arrows step one source frame; shift+left/right jumps five seconds.
- Click either timeline to seek, or enter a time in hours.
- Previous/next label change and previous/next sleep jump between labeled events.
- Timelines preserve brief events as per-pixel category proportions.

Labels are purple S0/S1, amber not-S0/S1, and gray ambiguous. The displayed S-number
comes from the original full-posture classifier. Earlier tag/antenna velocity exclusions and the abandoned partial-posture candidate
are not used. By default, missing full-posture labels stay ambiguous.

Enable **Cup motion exclusions** (or launch with `--cup-motion`) to include the
validated video-motion overlay from `analysis/sleep_s01_video_energy`. Newly
excluded intervals show a teal **NOT S0/S1 · MOTION** badge. Original full-posture
labels always take precedence; video motion never adds sleep. Low-motion periods,
failed quality checks and unrecorded gaps remain ambiguous. **Reload motion** loads
newly published results without moving the current cup or video position. The
status line reports how many time groups have been measured. Cup-level motion
can be assessed independently of resident tag identity; existing identity warnings
remain visible. This optional overlay requires precomputed cup-motion labels;
the viewer does not compute them.

Synchronization uses the stitched manifest's frame offsets and the same verified
common hardware-trigger/block IDs as the tracking pipeline. Filename creation
timestamps are not alignment offsets. An unrecorded inter-block gap has no video
and no label. Playback skips this gap; manually seeking into it shows an explicit
not-recorded view. Labels update only after the matching video frame has decoded.

These source MKVs have invalid duration/seek metadata in OpenCV. The reader locates
a real Matroska keyframe near the requested trigger frame, gives its original
timestamped clusters to PyAV, and checks each decoded presentation timestamp.
Only the selected cup is warped into a small view. Nearby frames are cached for
rewind; a distant seek or uncached reverse interval may take a few seconds.
Playback remains synchronized if decoding is slower than the requested speed.
Source videos, tracking files and classifier outputs are read-only.
