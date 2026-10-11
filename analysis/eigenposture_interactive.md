# Full-body states, ant classes, and S0-calibrated motion sleep

Open `analysis/eigenposture_interactive.py` and run its eleven `# %%` cells in
VS Code/Jupyter from the repository root. It uses the new full-recording model
packet and continues through the established grid-occupancy analyses. Plots are
ordinary Matplotlib figures, saved as PNGs; the batch runner writes a PDF with a
text description beneath every figure and an opening workflow schematic. Figure
numbers in the PDF are sequential; the original speed plot is identified by its
title rather than its former section number 10.
Companion modules handle the costly extraction, model comparisons, and summaries.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
python -m analysis.eigenposture_task_report --output /path/to/results
```

The default block is `20260724/block01` under the basler bucket, with inputs in
`analysis_outputs/full_body_behavior_20261011`. Set `ANTS_DATASET_ROOT` to change
the block, `ANTS_BEHAVIOR_SOURCE` (or the runner's `--source`) to change the model
packet, and `ANTS_ROLE_OUTPUT` to change the interactive output. The default
output is the packet's `report` directory. Install
`analysis/eigenposture_leiden_requirements.txt` in the pipeline environment;
downstream readers also require PyArrow and Shapely.


## Camera handling and one-second sleep (October 11 update)

The standard colony pipeline now waits for ≤1 s posture interpolation and runs
both colony reports. Position and posture may come from different overlapping
cameras in the common panorama. Each coordinate stream uses its own camera
provenance; a mismatch alone does not invalidate posture.

Posture smoothing uses five complete observations from the same camera and
recording: the observed center and two observations on either side, each within
five native frames. Quadratic fits use the actual frame times. The usual
consecutive-frame case equals the previous five-frame Savitzky–Golay filter.
Missing centers remain unknown. Position, antenna, turning and PC derivatives
never subtract coordinates from different cameras. One-second behavioral
windows can span camera changes after alignment, but never missing samples or
recording boundaries. View-dependent shape distortion may still affect posture
means or variability; this is not a camera calibration correction.

The separate sleep readout uses **observed raw landmarks** (including partial
poses), with displacements paired within the same SLEAP camera and recording.
Each speed retains its actual frame gap; the sleep classifier accepts gaps up
to five native frames. Interpolation supports the full-body state model but does
not itself supply evidence of immobility to this sleep readout. Refit the S0
thresholds after rebuilding measurements. The trailing sleep window is now
**1 second**, with ≥50% valid and ≥90% quiet observations. Labels are putative
sleep, not an exact replacement for S0 or a physiological sleep validation.

Versioned motion success markers prevent old cross-camera caches from being
silently skipped. Feature/model fingerprints reject stale results; use a fresh
analysis output directory when changing measurements. `ANTS_MOTION_CACHE_ROOT`
can point to an isolated replacement cache; calibration and return-context
readers use the same override. Old published outputs remain readable.

The numerical examples below describe historical July runs and are not results
of the updated method. Run-specific definitions, metrics and figures are saved
in each report packet.

## Fresh motion sleep calibrated from S0

`analysis.s0_motion_sleep` runs four dependent Slurm stages: per-track motion
aggregation, S0 threshold calibration, per-track labeling, and validation. Write
worker outputs to **flash/scratch**; compute nodes may mount bucket read-only.
Publish the completed `labels` directory as `sleep_from_s0_motion_20261011` in the
model packet. `ANTS_MOTION_SLEEP` overrides that path.

```bash
python -m analysis.s0_motion_sleep submit \
  --source "$RUN" --block "$BLOCK" --work /flash/path/s0_motion \
  --output /flash/path/s0_motion/labels --concurrency 24
```

This reuses calibrated bodypoint **motion measurements**, never old sleep labels.
Body motion is the 75th percentile across four body points with at least three
valid points; antenna motion is the 75th percentile across six antennal points
with at least three valid points. Displacements may span at most five frames;
individual point speeds above 20 mm/s are excluded. Both metrics are lab-frame
motion, unlike the head-relative antennal features of the state model.

Calibration samples up to 10,000 valid 12-Hz observations per ant/state throughout
the recording and gives each ant equal total weight within a state. Separate
95th percentiles in S0 set body and antenna limits, **0.160509 and 0.793659 mm/s**.
The joint envelope contains 90.77% of S0 calibration observations and 90.71%
in held-out time blocks. It also contains 85.88% of S1 observations; S0 and S1
are not cleanly separated by these motion variables.
Alternating 30-minute blocks assess temporal transfer using thresholds fitted
only on the other blocks; the final thresholds then use all calibration blocks.
Neither contact yield, spatial groups nor behavioral ant classes select limits.
These choices are explicit modeling assumptions, not physiological validation.

Fresh frame labels require a complete trailing one-second window, at least 50%
jointly valid body/antenna observations, and at least 90% of the valid observations
below both limits. Windows without sufficient evidence stay unknown. The same
rule applies to all 114 tracks, including observations without complete posture.
S1 and other states may satisfy this motion rule: it is a broader readout calibrated
from S0, not a claim of exact S0 recovery. The overlap/threshold plot makes this
visible. S0 remains the sole putative sleep **state** in the learned dictionary.

The completed fanout labels all 114 tracks. For the 44 right-colony ants, mean
full-recording label coverage is 82.43%; 87.65% of focal contact onsets have a
fully classified preceding ten seconds. There are 704 eligible recent-return
contacts and **484 matched contacts in 43 recipients**, versus 11 matches in
eight recipients with direct S0 labels. These pooled event counts are independent
of the subsequent ant-class reduction. Precise returns remain 1,827. The stricter
S0-derived body threshold retains fewer contacts than the historical 726-match
motion result, while restoring support in every behavioral class. Matching and
return-event rules are unchanged.

The standalone sleep time course matches the speed plot's clock, light shading,
ten-minute bins and class colors. It uses the fresh motion labels. Existing
state occupancy, bout, class-summary and clock-profile plots retain direct S0
membership and say so in their captions. Every PDF figure now has an explanatory
paragraph; `figure_descriptions.json` contains the same text.

Return-context caching can reuse fingerprint-matched physical position, speed,
and crossing arrays. Both sleep-dependent arrays (state and sleep age) are always
recomputed from the fresh labels; `physical_reuse.json` records this optimization.
If the physical cache is absent or incompatible, the ordinary raw-input reader
reconstructs it. Eligibility still requires ten continuously sleeping seconds
before contact, with the same matching criteria as before.

## What each cell does

1. **Cohort:** require detection in strictly more than 60% of all recording
   frames, including missing time in the denominator. Both colonies contribute
   states (26 left, 44 right); only those 44 right ants enter ant-class analysis.
2. **Posture PCA:** all ten tracked landmarks, after tag centering and alignment
   to the petiole-to-tag axis. Divide by each ant's median axis length. Four body
   and six antennal points are available; there are no leg landmarks. Fit the
   covariance from all 81,159,438 usable 12-Hz postures across the full recording.
   Eight PCs explain 96.87% of variance. Plot variance and coordinate loadings.
3. **Behavioral states:** compare mean posture PCs (4 or 8), unsigned forward and
   lateral peak velocity, both head-relative antennal speeds, turning, and exact
   concurrent partner counts; then add within-window PC SD and RMS PC rate.
   Continuous observations are partitioned into at most one-second windows,
   retaining tails of at least 250 ms. Missing samples and recording boundaries break segments.
   Camera handoffs can remain inside a window; all motion and PC rates are computed within each camera.
   Three UMAP settings show the same labels; clustering uses the balanced feature
   space, never the 2-D coordinates. Inspect feature profiles and quiet candidates.
   The mean-posture figure adds each state's mean full-body skeleton and a heatmap of its
   unsigned body/antennal velocity, turning, and exact concurrent contacts.
4. **Ant representation:** count each ant's assigned state observations directly,
   then normalize. Counts weight variable-length windows by their observed
   duration. No equal-hour averaging, additional eligibility filter, or final PCA.
5. **Ant classes:** compare K-means, Ward, tied/diagonal GMM, Leiden, and HDBSCAN
   in square-root state proportions (Hellinger geometry). Refit the selected
   method visibly and assert agreement with the saved classes. Prefer the most
   reproducible partition, with fewer classes breaking ties. The current result
   is two classes, containing 26 and 18 ants.
6. **Inspect:** show how ant state proportions form the behavioral ant classes:
   one UMAP point per ant, colored by its fitted class, alongside the class mean
   state proportions. Compare UMAP neighbors 4/8/20 and min_dist 0.02/0.15/0.5,
   with three seeds per setting. Select the view by median five-neighbor
   trustworthiness, then neighborhood overlap, without class or spatial labels.
   The nine panels show the fixed primary seed; all 27 fits and scores are saved.
   UMAP is a view only; class fitting retains all eight square-root proportion
   coordinates. Apparent island spacing is not evidence for discrete classes. Also plot
   individual state proportions and a stage-by-stage coverage audit.
7. **Space:** hand the fixed classes to existing grid utilities; require an exact
   side/ID/filename match for every classified ant. Spatial labels are read for
   the first time here, solely for post hoc comparison. Plot mean/individual
    occupancy and nest use. Never overwrite the original spatial classes.
    The published packet carries the original two-group spatial reference.
    The current grid-label file has three right-colony groups; this is a different
    spatial partition. A new packet without a saved reference snapshots that
    current file instead. Neither reference affects any model fit.
8. **Speed and putative sleep:** use **S0 only** as the putative-sleep state,
   then derive frame labels directly from those assignments. S1 is non-sleep
   under this explicit definition, even though it also has low median motion.
   Plot quiet-run durations, class curves, and individual heatmaps. A combined
   sleep/speed figure shows individual ant values, equal-ant class means, and
   ten-minute time courses. Whole-recording sleep is quiet samples / classified
   samples for each ant; it uses every classified observation. Whole-recording
   speed averages all finite samples in the existing trajectory cache for each
   ant. Classes then give each ant equal weight. Unknown observations never count
   as awake or zero speed. Time-course sleep bins retain the established 50%
   classified-coverage threshold; their contributing ant counts are shown in
   the detailed sleep figure. The whole-recording points have no extra bin filter.
9. **Clock profiles:** activity and state-derived quietness by time of day,
   using complete light-dark cycles when available.
10. **Trip investment:** restore the grid-occupancy trip frequency versus median
    duration plot, colored by behavioral class, plus class comparisons of return
    rate and total time on completed trips. Include all classified ants; zero-trip
    ants retain zero rate/time and have no median duration. Reuse the grid utility's
    1-s position bins, gap bridging up to 30 s, 5-s border/colony support, 30-s
    minimum trips, and 20% observed trip coverage. Rates use the full recording
    duration. Bridged position gaps can contribute to trip duration; these rules
    do not fill gaps in sleep labels. Trip summaries and uncertainty are saved.
11. **Returns and contacts:** retain the same ant classes and use fresh
    **S0-calibrated motion labels** for recipient selection, matched controls,
    and response curves. All 57 right ants have motion labels and remain social
    context; only the 44 behavior-classified ants enter recipient class comparisons.
    Show pooled returns and recipients before individual class panels, with event/ant counts
    in titles. The precise return-response rules are stricter than trip investment
    (no long position-gap bridging; observed crossing within five frames).

## Choices to assess

The graph dictionary contains 300 windows per ant, sampled throughout the whole
recording, with duration weighting within ant. All 7,763,243 usable windows are
then assigned by five-neighbor distance-weighted voting; dictionary observations
retain their graph memberships. Thus feature measurement and assignment cover
all usable segments, while graph fitting uses landmarks for tractability.
Exact-neighbor spot checks audit approximate assignment.

Posture coordinates are not replaced with hand-defined antennal straightness.
Nonnegative variation, rate, motion and contact measurements receive log1p
transforms. Each feature family has equal total variance, preserving every
coordinate. Exact contacts are unique simultaneous partners per raw frame,
summarized by mean and maximum; persistent contact counts on every observed frame.

The mean-posture figure averages **actual aligned coordinates**, without PCA
reconstruction, across all assigned observations from both colonies. The same
body-normalized scale is used in every panel; body points are teal and antennae
orange. These are average skeletons, not representative frames; averaging can
hide posture variation. Every observed 12-Hz sample receives equal weight, so
longer observed windows and ants with more classified time contribute more.
The accompanying feature means use the same duration weights. Forward/lateral
velocity and peak contacts remain **within-window maxima**, averaged across
windows; antennal/turning speeds and mean contacts are within-window means.
Printed heatmap values retain mm/s, rad/s, or partner-count units. Colors span
the lowest to highest state mean separately in each feature, allowing comparison
without mixing units. These pooled means differ from the ant-balanced dictionary
medians used for quiet-state interpretation; plotting these means does not alter
labels. Numerical values are in `state_mean_features.csv` and
`state_mean_postures.csv` beside the plots.

The original feature screen compared Leiden resolutions 0.4, 0.8, 1.2, and 1.6,
selecting the finest partition passing seed ARI >=0.8, ant-removal ARI >=0.65,
at least five ants per state and minimum dictionary fraction 0.002. This used
only one alternative partition seed and one ant-removal fit. The expanded audit
below supersedes that limited evidence: eight is now a **provisional reference**.
Stars use median matched-state Jaccard below 0.5 across five ant removals.
These checks hold full-cohort PCA/scaling fixed and measure perturbation
stability, not independent held-out biological validation.

Ant comparisons allow K=2–8, plus variable-K graph/density methods. Candidates
must have at least three ants per class, no unassigned ants, median
ant-removal ARI >=0.8 and weakest-class Jaccard >=0.6, and median temporal ARI
>=0.65 and weakest-class Jaccard >=0.6. Use 30 ant-removal repetitions and
alternating 1-, 3-, 6-, and 24-hour blocks. Among passing candidates, maximize
the minimum of the 10th-percentile ant-removal ARI/Jaccard and the worst temporal
ARI/Jaccard across the four split widths. Ties prefer fewer classes, then higher
silhouette. This replaces the previous finest-passing rule and prevents a median
from hiding failure at a particular timescale. Hours are validation blocks only;
class fitting always pools observations directly. Fall back to one class if
none passes. A reproducible partition is not proof of discrete biological castes.

Putative sleep defaults to **`SLEEP_STATES=[0]`**, an explicit post hoc choice.
The command-line sleep builder also defaults to S0 (`--quiet-states 0`). The
motion-profile candidate rule (median unsigned body peak <=0.05 mm/s and median
mean-antennal speed <=0.15 mm/s) remains descriptive; both S0/S1 meet it, but only
S0 is used for sleep. `SLEEP_STATES=None` explicitly opts into all candidates.
These choices do not train states. `MIN_SLEEP_SECONDS=0` shows membership directly; the
bout plot separately assesses persistence. Unknown intervals remain unknown and
censor bouts. They never count as awakenings. This is putative sleep without an
arousal assay. **No historical motion-sleep label arrays enter these analyses.**
Direct S0 membership is retained for state occupancy and quiet-run plots. The new
motion readout below is used for return figures and the standalone sleep time course.
Ant classes were fitted to these same state proportions: differences in total
quiet time describe the classes and do not independently validate them.
Contacts enter state discovery but do not enter the new motion sleep classifier.
Its calibration observations come from S0, which was learned using contacts as
one feature family. Labels use only past and current motion; the trailing window
can delay detected transitions. The recipient comparison remains descriptive: its
awake fraction is defined through motion, so also inspect the physical traces.
The recipient analysis still requires ten continuously classified sleeping
seconds before contact. `recipient_sleep_support.csv` separates unknown history
from known non-sleep history. It also reports a diagnostic interval ending one
second before contact, to inspect short windows that straddle onset; that guard
does not change primary event selection. `recipient_sample_counts.csv` reports
matched events and ants by class/condition. Bootstrap bands require at least five
contributing ants and can disappear when splitting a small sample into classes.

Median classified coverage is 52.8% of recording time for left ants and 49.8%
for right ants, despite their higher detection coverage. Complete skeletons,
valid motion support, and the 250-ms minimum are needed for these features.
Short-tail exclusion removes 3.60% of otherwise feature-valid samples. Missing
posture can be behavior dependent; pooled proportions describe observed behavior.

The coverage audit explains the gap between detection and state assignment.
For right-colony ants, **mean** coverage of the full recording is 82.7% for
detected positions, 53.3% after full-body posture checks, 50.4% after motion
checks, 50.4% after requiring contact-output availability, and 48.5% after
discarding segments shorter than 250 ms. The corresponding left-colony values
are 82.8%, 56.8%, 54.7%, 54.7%, and 52.9%. These are equal-ant means; the median
classified coverages above differ slightly. Detection needs a tracked position;
posture needs all ten landmarks, valid body geometry, and five consecutive
24-Hz frames from one camera. Motion additionally needs finite derivatives,
matching pose/position cameras, and body speed at most 20 mm/s. Cached counters
identify the total loss at each stage, not the separate contributions of each
check inside the posture stage. Contact availability causes no further loss in
this run. Sampling at 12 Hz does **not** halve coverage: posture-stage counts
use the 12-Hz denominator. `state_coverage_stages.csv` and
`state_coverage_summary.csv` save the audit.

The speed summary reuses the grid-occupancy trajectory cache: short gaps up to
five frames are interpolated, positions have Gaussian smoothing with sigma two
frames, and speeds above 5 mm/s are rejected in this dataset's cache. These
mean trajectory speeds therefore differ from the unsigned within-window peak
features used to learn states. `ant_sleep_speed_summary.csv` and
`class_sleep_speed_summary.csv` save the individual values and equal-ant means.

## Rebuild from pipeline outputs

Choose a new output directory when changing measurement or modeling choices.
Use local disk for roughly 20 GB of intermediate arrays, not a quota-limited
remote home. Existing stitched tracks, speed metadata, published skeleton
interaction outputs, and downstream grid/motion caches are required. The motion fanout must finish
before the report runs; if its output is elsewhere, publish it under the model
packet or set `ANTS_MOTION_SLEEP` to the completed labels directory.

```bash
BLOCK=/path/to/20260724/block01
RUN=/local/scratch/full_body_behavior
python -m analysis.full_behavior_extract --block "$BLOCK" --output "$RUN" --workers 6
python -m analysis.full_behavior_states features --root "$RUN" --workers 4
python -m analysis.full_behavior_states learn --root "$RUN"
python -m analysis.full_behavior_states assign --root "$RUN"
python -m analysis.full_behavior_profiles --root "$RUN"
python -m analysis.full_behavior_roles --root "$RUN"
# Run on the cluster, with paths visible to workers and writable scratch output.
python -m analysis.s0_motion_sleep submit --source "$RUN" --block "$BLOCK" \
  --work /flash/path/s0_motion --output "$RUN/sleep_from_s0_motion_20261011"
# Continue after the final Slurm job succeeds and the output complete.json exists.
ANTS_DATASET_ROOT="$BLOCK" python -m analysis.eigenposture_task_report \
  --source "$RUN" --output "$RUN/report"
```

The published packet retains features, model comparisons, assignments, PCA,
coverage/provenance, classes, derived quiet labels, and the report. Large dense
pose/motion arrays are reproducible local intermediates. `models/state_means.*`
retains the small all-observation summary; plots reuse it without the dense
arrays. Rebuild that summary after changing assignments or features. Earlier
sampled-clip scripts have been removed; their committed versions remain in Git
history. Their state numbers are not interchangeable with this dictionary.

## Current state and ant-class reduction audit

The reference dictionary has eight states at Leiden resolution 0.4 in 11
dimensions: four full-body posture PCs (89.79% of posture variance), five motion
measurements, and mean/peak concurrent contacts. Expanded representations and
finer resolutions in the original screen produced 9–24 states but failed its
checks. The original single-repeat seed/ant-removal ARIs were 0.813/0.753.

The stronger audit uses **three new seeds, excluding the reference seed**, and
five separate removals of 20% of the ants. Median reference ARIs fall to
**0.726 / 0.706**. State S3 has median held-out matched Jaccard zero; S6 has
0.426. These uncertain fine states are marked with asterisks in the report.
Retaining the dictionary does not validate eight distinct biological actions.

We allowed every state, including S0, to merge. Ward hierarchies either weight
each original state equally or weight it by its observation count in the
ant-balanced dictionary. Each stability repeat refits Leiden before merging,
then tests withheld ants by fine-state nearest-neighbor voting followed by the
merge. Fixed-dictionary relabeling alone would overstate stability.

The conservative selection rule requires >=90% of the original between-state
feature variation, seed ARI >=0.8, ant-removal ARI >=0.65, and no decrease from
the reference median seed/ant-removal ARI or weakest-state Jaccard. Sensitivity
at 85% and 95% retained variation gives the same decision: **no supported
reduction**. For example, equal-state six-way merging combines S0/S1 and S4/S5,
retains 91.6% of the variation, but has seed ARI 0.653 versus 0.726 for the
reference. Observation-weighted six-way merging retains 94.5%, with seed/ant
ARIs 0.724/0.696, also without improvement. Direct lower-resolution Leiden
(0.01–0.4) gives stable coarse alternatives, but these mainly distinguish
contact intensity and discard substantial quiet/movement differences. This
screen uses two alternative seeds and has no separate ant-removal validation.

S0 and S1 remain separate low-motion candidates. Their median body motion is
about 0.012 mm/s and median antennal motion about 0.084/0.090 mm/s. **S0 alone
remains the putative sleep state**, so its existing motion calibration remains
valid. This is an outcome of the merge audit, not a constraint that protected S0.
Missing observations censor quiet runs; physiological sleep requires more than
low motion alone.

For ant classes, the new rule selects a **tied-covariance two-component GMM**
on all eight square-root state proportions: **C0 = 26 ants, C1 = 18 ants**.
Median ant-removal ARI is 1.00; its 10th percentile is 0.886. Median temporal ARI
is 0.955 and its worst value across the four split widths is 0.909. The primary
weakest-perturbation scores for the best two-, three-, and four-class candidates
are **0.886, 0.800, and 0.575**, respectively. Comparing state counts 2–8 under
both merge hierarchies also favors two ant classes across detailed vocabularies.

Every former C0 ant (10) and C1 ant (14) enters the new C0, together with two
former C2 ants. The other six former C2 ants and all twelve former C3 ants enter
new C1. This is a new behavioral fit, not a manual forced merge. The prior
four-class labels are retained in the audit for this comparison only. Spatial
labels are examined after fixing this result and never select a candidate.
The two behavioral classes match the frozen two-group spatial reference for all
44 ants (ARI 1.00). This agreement describes this recording; it is not an
independent experiment or held-out prediction test.

`state_class_reduction_summary.pdf` collects the state-count audit, weak-state
support, ant-class selection, old/new class mapping, sensitivity to state count,
and post hoc spatial comparison. The full report carries the two classes
through sleep, speed, trip and contact analyses.

To reproduce the audit on an existing packet (save old classes before refitting):

```bash
AUDIT="$RUN/reduction_20261011"
mkdir -p "$AUDIT"
cp "$RUN/roles/ant_classes.csv" "$AUDIT/original_ant_classes.csv"
python -m analysis.full_behavior_states resolutions --root "$RUN" --output "$AUDIT"
python -m analysis.full_behavior_states reduce --root "$RUN" --output "$AUDIT" --merge-weighting state
python -m analysis.full_behavior_states reduce --root "$RUN" --output "$AUDIT/weighted" --merge-weighting observation
python -m analysis.full_behavior_roles --root "$RUN" --compare-reductions "$AUDIT" --merge-weighting state
python -m analysis.full_behavior_roles --root "$RUN" --compare-reductions "$AUDIT/weighted" --merge-weighting observation
python -m analysis.full_behavior_roles --root "$RUN"
ANTS_REDUCTION_AUDIT="$AUDIT" ANTS_DATASET_ROOT="$BLOCK" \
  python -m analysis.eigenposture_task_report --source "$RUN" --output "$RUN/report"
```

`ANTS_ROLE_MODELS` can point to a separate rebuilt `roles` directory while
reviewing results before replacing a packet. The state audit saves exact mappings,
repeat-level agreement, per-state Jaccards and retention-threshold sensitivity.
The ant audit saves every candidate and perturbation result. No state dictionary,
assignment or S0 motion label was altered by this class-only update.

## Display and trip results

The UMAP view is selected by neighborhood preservation, independently of class
labels. All nine settings show the same two fitted classes; apparent separation
does not determine their membership. The recording contains 5,923 completed
trips across the 44 ants and 1,827 precise response returns. These differ because
the trip-investment rules allow longer position gaps and less observed support.
Current per-class values are saved in `class_trip_investment.csv`. C0/C1 average
18.6/133.7 completed returns per recording day and 36.8/612.9 minutes on completed
trips per day. Their mean trajectory speeds are 0.193/0.639 mm/s; direct S0
occupancy averages 29.8%/16.2% of classified observations. These summaries give
each ant equal weight. S0 occupancy is distinct from the motion-defined sleep
time course used alongside return analysis.

## Historical diagnosis: why state-only return-contact plots became sparse

The comparison below holds the 44 right-ant identities fixed. The earlier
motion-label analysis and the state-label analysis both retain 1,827 return
events. Splitting return curves across four ant classes reduces support per
panel; the pooled view is restored for comparison.

Sleeping-recipient selection is affected much more strongly. The unchanged
eligibility rule requires ten **continuously classified sleeping seconds**
before a new contact. The earlier motion classifier used ten-second windows
with >=50% valid frames, >=90% quiet valid frames, partial landmark support,
and short-gap handling. State labels require the full-body feature support
described above. Among 1,103,126 focal ant-contact onsets, a fully classified
preceding ten seconds is available for 87.6% with the historical motion labels,
but only 3.2% with either state-label definition. The main loss is continuity
of label support, not a decrease in detected returns.

| Sleep definition | Recent-return contacts entering matching | Matched contacts |
| --- | ---: | ---: |
| Historical motion labels, same 44 ants | 1,036 | 726 |
| Previous S0 + S1 | 36 | 29 |
| Previous direct S0 only | 17 | 11 |

The preceding direct-S0 report had eleven matched contacts in eight recipients: C0 has one event
in one ant, C1 three in three, C2 seven in four, and C3 none. No class has five
contributing recipients, the minimum for the bootstrap intervals. Sparse curves
here do not establish class-specific contact responses.

Contact-containing one-second windows can also change the state assigned to
frames before contact. As a diagnostic, ending the ten-second history one second
earlier raises all-onset sleep-history support from 1,071 to 8,522 for S0+S1,
and from 541 to 5,065 for S0. These are counts **before** other eligibility and
matching filters, not extra matched events. The change is consistent with
onset-straddling feature windows; ordinary state changes can also contribute.
The guard was diagnostic only. Current return figures now use fresh motion labels
calibrated from S0, while keeping the original event eligibility and matching rules.
S0 membership remains unchanged for state and ant classification.

`contact_label_history_totals.csv`, `contact_label_history_audit.csv`, and
`contact_label_coverage_comparison.csv` retain the historical comparison in the
published dataset. Source metadata is recorded in `contact_label_audit_inputs.json` in
`reduction_20261011/historical_return_labels/`.
The standalone historical-label diagnostic has been removed; its committed
version remains in Git history. The supported report uses the current S0 and
S0-calibrated motion definitions described above.
