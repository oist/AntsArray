# Live distance review

This viewer now isolates one interaction parameter: **Skeleton distance (mm)**,
default **0.1 mm**. Moving the slider, using the stepper, or typing a value updates
the current frame automatically, while paused or playing. There is no Apply
button for interactions. Sleep parameters retain their separate Apply button.
Sleep/wake overlays are **off by default**; enable **Sleep/wake** in the toolbar
to show them. The view stays full-camera; the focal-ant follow control is removed.
A **1 mm scale bar** in the lower-left corner uses the panorama calibration and
camera homography at that location. It scales with the video when resizing.

For each visible unordered pair, the rule is:

`HIT = shortest distance between the two finished skeletons <= threshold`.

The geometry includes all observed body and antenna nodes and all drawn skeleton
segments whose endpoints are observed. Segment crossings have distance zero even
when their endpoint nodes are far apart. This is not filled-body silhouette
overlap, and the thickness of the displayed lines does not change the distance.
Missing nodes never bridge missing segments; a pair without usable pose is
marked **no pose**, not assigned zero distance. Shapely/GEOS computes closest
points. Distances are measured in panorama tracking coordinates at the cache's
scale, **0.016 mm per pixel** for this recording, before camera projection.

There are no antenna-only restrictions, center-radius cutoff, duplicate-pose
rejection, onset rules, history windows, minimum-duration requirements, refractory
periods, separation checks, or one-second highlights in this diagnostic view.
A stationary overlapping pair is a HIT on every frame. The hit disappears
immediately when its distance exceeds the threshold. This is deliberately not
yet a definition of a *new* interaction.

- Cyan **HIT** marks each current hit at its closest skeleton points.
- The scrollable table lists all visible pairs, nearest first, including misses.
  Select a row to highlight its closest points even when it is not a hit.
- **Export frame** writes current pair distances, hit flags, closest-point
  coordinates, frame number, threshold, scale and source metadata under the
  pose cache's `reviews/distance_frame_...` directory.
- Invalid or empty thresholds clear HIT classifications and show an error.
- All displayed identities and skeletons come from finished
  `block02/tracks/*.parquet` data, cached in `geometry.npz` for this interval.
  There is no raw-camera overlay or nearest-neighbor identity reassignment.
  Sleep labels use those same finished IDs and colony.
- Only geometry for the current frame is evaluated and retained. The viewer
  does not load saved interaction detections or precalculate bouts for the clip.

The production detector now shares `tracking/colony/skeleton_contacts.py` with
this viewer. The replacement block02 fanout uses this same 0.1 mm rule and writes
undirected pair/frame distances into `block02/interactions_skeleton_0p1mm`.
Older antenna-to-node caches remain separate. Moving the GUI slider only changes
the live display; rebuilding production caches requires another run. The return
analysis separately merges hits into bouts using its existing 2-second gap rule,
so a sustained HIT is not repeatedly counted as a new interaction.
Skeleton proximity or overlap alone does not prove physical contact.

Fast relaunch for the cached cam04 episode:

```bash
/home/sam-reiter/miniforge3/envs/ants/bin/python tracking/gui/interaction_debug_viewer.py \
  --video /home/sam-reiter/bucket/ReiterU/Ants/basler/20260723/block02/cam04_cam3_2026-07-23-19-31-09.mkv \
  --review-cache /home/sam-reiter/bucket/ReiterU/Ants/basler/20260723/block02/stitched/analysis_cache/contact_review/985ad54a180cb0f7 \
  --video-clip /home/sam-reiter/.cache/AntsArray/contact_review/cam04_T56_676122_684282.mkv \
  --clip-start-frame 676122 --start-frame 680747 --distance-mm 0.1
```

The clip covers global frames 676122 through 684281. T56's colony crossing is
677082; entry into this camera is around 677286. These episodes were selected
for debugging, not as an unbiased estimate of wake effects. A local video clip
must be frame-exact; a keyframe-only trim with an assumed offset can misalign
labels. For another episode, omit the cache/local-clip arguments and set the
return/start frame, target and colony side. The legacy preparation utility
builds the cache; only its finished poses are used by this live viewer.
