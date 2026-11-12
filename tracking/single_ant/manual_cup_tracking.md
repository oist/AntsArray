# Manual cup tracking with the native colony engine

Run the reusable workflow from a Deigo login node:

```bash
/home/s/samuel-reiter/.venvs/ants-eigenposture/bin/python -m tracking.single_ant.run \
  /bucket/ReiterU/Ants/basler/single_ants/20261027 --blocks block01 block02 --wait
```

Omit `--blocks` to select every numbered block. The latest earlier manual cup
labels are the default (October 26 for this example); block-local labels take
precedence. Use `--regions /path/panorama_regions.csv` to override. Calibration
comes from the label metadata. This workflow requires lossless ArUco HDF5,
SLEAP HDF5, and valid camera diagnostics covering the complete recording.
For an interrupted recording, explicitly list cameras
with synthesized crash sidecars using `--interrupted-cameras cam09 cam10 ...`.
Their shorter recording spans remain missing after they stop. The manifest
records that these cameras lack recorder telemetry and their common-trigger
alignment is assumed; dropped frames cannot be ruled out. ArUco-only frames
past the terminal SLEAP export are excluded. All other cameras retain the
strict recorder checks. Use `--track-task-batch-size 8` to group eight independent
chunk/cup tasks per Slurm slot when the full array exceeds cluster limits.

Cup names and TrackIDs follow image reading order: left to right within each
row, then top to bottom. Slightly misaligned centers are grouped within half
the median cup radius. `cup_order.csv` records row, column and previous names;
cup geometry and raw ArUco IDs are preserved.

Each block publishes to `blockNN/analysis_outputs/single_ant`. Consecutive
blocks are stitched into `continous_stitched`, matching the colony dataset's
directory convention. Use `--analysis-dir` and `--combined-dir` to choose
different destinations without overwriting existing results. Scratch work,
frozen source code, source fingerprints and Slurm IDs are recorded in the
printed `plan.json`. Omit `--wait` to run the login-node publisher detached.
Resume or retry failed jobs with `--resume /path/to/plan.json --wait`.
Finished tasks are retained; failed array slots are retried once the array
has settled. `--exclude-nodes NODELIST` can exclude an unavailable compute
node when starting or resuming a run. All submission attempts are recorded.
Publishing runs on the login node because compute-node bucket mounts are
read-only. Slurm arrays have no additional concurrency throttle.
Published metadata points to the published tracks and caches, retaining the
scratch run location separately in the publication record.

To track a new date and extend the stitch using completed earlier blocks:

```bash
python -m tracking.single_ant.run /bucket/ReiterU/Ants/basler/single_ants/20261028 \
  --blocks block01 --stitch-with \
  /bucket/ReiterU/Ants/basler/single_ants/20261027/block01/analysis_outputs/single_ant/manifest.json \
  /bucket/ReiterU/Ants/basler/single_ants/20261027/block02/analysis_outputs/single_ant/manifest.json
```

Earlier blocks are reused without tracking them again. Individual-block
manifests must be complete. Combined sources are sorted by trigger timestamp;
block labels include the date when spanning dates. The new combined output
lives under the selected new date, preserving earlier stitched outputs.

To make a different stitch from completed blocks, with no tracking jobs:

```bash
python -m tracking.single_ant.run --stitch-only /path/blockA/manifest.json /path/blockB/manifest.json \
  --output /path/new_stitch
```

For interactive speed inspection on the workstation:

```bash
/home/sam-reiter/miniforge3/envs/ants/bin/python -m tracking.single_ant.speed_viewer /path/new_stitch
```

Select a cup, use the toolbar to pan/zoom, or enter start/end hours and click
Go. Previous/Next cup retains the time window. Full recording resets the
view; Fit Y adjusts its vertical range. Unclipped speed and Log scale expose
fast movements and outliers. The lower panel shows valid-speed coverage;
grey spans mark unrecorded time. Overview bins include every finite sample
and display their mean and min–max range. Zooming resolves individual frames.
Missing samples stay missing, and lines break at recording boundaries.
The viewer reads cached vectors only and does not change tracking results.
Run without a folder argument to choose results interactively, or use
`--cup N` to start at a particular cup.

The tracking engine entry point is `tracking.single_ant.full_cup_tracking`.
The restrictive pilot remains in `manual_cup_tracking.py` for reproduction
and shared input utilities; it is not the full-run tracker.

The full run calls the shared
`tracking.core.tracking_utils.get_complete_tracks` function. Each manually
labeled cup is processed independently. Its assigned raw ArUco ID remains
the resident identity. Other IDs compete for skeletons under the colony
rules and never cause a permanent stop. Public TrackID is CupID, with the
raw tag preserved in ArucoID, so cups sharing a tag remain distinct.

Known cup identities from the initial >60% qualification remain fixed even
when later tag visibility decreases. Previously unassigned cups can qualify
over the entire recording: one instance of an ID in strictly over 60% of all
frames, with skeleton support to establish the identity. Extra IDs do not
invalidate an already assigned cup.

The native tracker receives all finite SLEAP nodes, including partial and
low-confidence poses, as in the colony mapper. No extra minimum node count,
pose merging, sole-ant requirement, or intrusion censor is applied.
Bodypoint 0 assigns a whole skeleton to its manual cup. Anchors outside or
ambiguously inside overlapping cups are excluded. No source colony, arena
inference, or left/right split is used.

Competitor ArUco IDs use the colony mapper's 25% all-camera frame-frequency
filter per chunk. The dedicated ID is always retained, including chunks in
which its visibility falls below that filter. The colony engine then applies
its own tag-to-SLEAP support filter, nearest matching, one-to-one assignments,
continuity gates, and SLEAP bridging. Parameters match the working pipeline:
90 px at 0.016 mm/px (1.44 mm), 120-frame lost-track memory, and the normal
age-expanded ArUco reacquisition radius. Ambiguous frames may remain missing;
the tracker keeps trying and can reacquire the resident later.

The shared tracker now treats geometrically matching camera views as one
SLEAP bridge hypothesis. Matching requires at least six common finite landmarks,
agreement of body direction and length, and tight distances relative to the
smaller body axis. Complete-link grouping prevents chains of nearby ants from
being merged, and conflicting decoded identities cannot be grouped. Within one
camera, a complete duplicate is preferred over its matching partial prediction.
The selected skeleton remains one original observation; coordinates are never
averaged. All bridge aliases and same-camera duplicates are claimed together
to prevent reuse by another identity. These two fixes also apply to colony tracking. Tags without SLEAP
support are still rejected.

The verified camera contract uses common hardware-trigger block IDs and
actual HDF5 chunk lengths. Filename arming timestamps are not offsets.
Raw calibration coordinates at 11 px/mm are normalized to 0.016 mm/px before
native tracking. Independent 30-minute chunk tracking and subsequent
stitching follow the colony workflow. Actual source chunk lengths establish
frame offsets even when camera export chunks do not line up.

The run maps all chunks, assigns identities, tracks every chunk/cup pair,
stitches residents, computes diagnostics, builds reports and publishes the
finished outputs. Arrays have no additional concurrency throttle; Slurm
schedules their requested resources. Unknown cups are explicitly reported.

Speed calls `analysis.compute_track_speed_vector.speed_vector` unchanged:
the recording's frame rate, 0.016 mm/px, interpolation across at most five missing frames,
Gaussian sigma two frames, and the standard 20 mm/s exclusion. A second
vector preserves unclipped speeds so jumps remain visible. This interpolation
does not alter saved observed positions or skeletons.

Every assigned cup receives full-duration trajectory, x/y, standard speed,
unclipped speed, source coverage, posture completeness, gap duration and
matching-distance plots. Full-resolution speed vectors and per-frame tracks
accompany the plots. Summary plots show the cup map, coverage and motion
across the entire recording. The complete PDF includes each cup's individual
diagnostics. This run does not fit a new eigenposture model.

Cross-block stitching uses the production colony functions
`concatenate_shifted_parquets` and `block_caches.merge_dense`. Trigger-start
timestamps set real frame offsets; unrecorded intervals stay NaN. Tracks
retain `source_block` and `block_frame`, and speed estimates are copied
unchanged from each block without smoothing or interpolation across boundaries.
The same cup and raw tag are one ant record; a newly qualifying different tag
in that cup creates a separate record. A previously established ID is retained
when a later block cannot qualify an ID, so tracking keeps trying. Equal raw
tags in different cups remain separate. Changed cup labels or geometry require
review before stitching rather than silently assuming a correspondence.

Eigenposture analysis is a separate command for a completed block or stitch,
run from a Deigo login shell with the analysis environment:

```bash
python -m tracking.single_ant.eigenposture submit /bucket/path/to/completed_dataset
python -m tracking.single_ant.eigenposture publish --plan /flash/path/printed_by_submit/plan.json
```

Submission launches dependency-linked preparation, extraction, and model/report
jobs. Run the publisher on the login node; compute nodes cannot write to the
bucket. Results go to `analysis/eigenposture`, with reusable posture and motion
caches beside that folder. The model packet contains its code snapshot and a
`run_report.sh` launcher. Existing tracking and speed vectors are preserved.
For long combined recordings, use `submit --prepare-mem 128G` to increase
memory per preparation worker from the default 64 GB.

The state discovery cohort uses the colony pipeline's strictly greater than 60%
position-coverage rule. Single-ant prediction also assigns the fixed model to
locally valid full-body windows from ants outside that cohort. Missing or partial
posture does not become a sleep label, and discovery eligibility remains unchanged.
This extra prediction step is confined to the single-ant workflow. Body geometry QC is calibrated to the ants' measured
lengths, with the original alignment, normalization, PCA, windowing, state and
class methods. Contact features are unmeasured and excluded. The report reuses
the colony posture, state, class, speed, sleep and clock-profile cells, adding
eigenposture skeleton reconstructions and within-cup occupancy for every cohort
ant. Nests, trips and social contacts do not apply to individually housed ants.

For a paired rerun using exactly the existing mapped detections, identities and
frozen state dictionary, use the original flash manifest and a fresh destination:

```bash
python -m tracking.single_ant.retrack prepare --source /flash/path/block02/manifest.json \
  --reference /bucket/path/analysis/eigenposture --destination /bucket/path/new_result
python -m tracking.single_ant.retrack submit --plan /flash/path/printed_run/plan.json
python -m tracking.single_ant.retrack publish --plan /flash/path/printed_run/plan.json
```

The rerun writes per-ant coverage comparisons and separates the effect of local
prediction on original tracks from changes caused by pose association. The
reference dictionary, original tracking and raw detections are preserved.

Use `retrack prepare --tracking-only` to rerun an individual block before
restitching and fitting a fresh model. The recording date and block name come
from its source manifest. This mode reuses mapped detections and the resident
registry, and does not assign states with the previous dictionary.

After a fresh eigenposture run, `analysis/sleep_s01_exclusion` contains direct
S0/S1-versus-other labels and `s01_every_ant.pdf`. Every cup is represented.
Locally valid windows from ants outside the discovery cohort can be predicted;
unrecorded gaps, unusable full-body measurements and excluded resident IDs
remain ambiguous. This export adds no motion-inferred labels.

For a sleep/speed report with a chosen putative sleep-state set, use the direct
state export as input and a separate output directory:

```bash
python -m analysis.single_ant_sleep_speed_report all \
  --dataset /path/to/completed_dataset \
  --metadata /path/to/completed_dataset/analysis/sleep_s01_exclusion \
  --state-root /path/to/completed_dataset/analysis/sleep_s01_exclusion \
  --registry /path/to/cup_tag_ids.tsv \
  --sleep-states 0 1 2 \
  --output /path/to/new_sleep_speed_report
```

Registry metadata are joined by cup and resident tag, allowing repeated tags in
different cups. Omit `--registry` when the metadata input already contains colony
and role. S0/S1/S2 are positive in this example; every other observed state is
negative, and missing posture and unrecorded gaps remain unknown. The original
observed state numbers and fitted dictionary are preserved. The output contains
PDF and offline HTML reports, per-ant label vectors, summary tables and
provenance. `extract`, `analyse` and `render` can also be run separately to reuse
the cached summaries. The default timeline bin is ten minutes.

The speed comparison chooses thresholds using other ants, evaluates each held-out
ant, and includes sustained-stationarity and temporal sensitivity checks. These
measure agreement with the chosen posture-state labels; the state model includes
motion features and remains subject to its reported stability limits.
