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

Fresh frame labels require a complete trailing ten-second window, at least 50%
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
eight recipients with direct S0 labels. C0/C1/C2/C3 contribute 134/164/123/63
matched events in 10/14/8/11 ants. Precise returns remain 1,827. The stricter
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
   retaining tails of at least 250 ms. Gaps and camera changes break segments.
   Three UMAP settings show the same labels; clustering uses the balanced feature
   space, never the 2-D coordinates. Inspect feature profiles and quiet candidates.
   The mean-posture figure adds each state's mean full-body skeleton and a heatmap of its
   unsigned body/antennal velocity, turning, and exact concurrent contacts.
4. **Ant representation:** count each ant's assigned state observations directly,
   then normalize. Counts weight variable-length windows by their observed
   duration. No equal-hour averaging, additional eligibility filter, or final PCA.
5. **Ant classes:** compare K-means, Ward, tied/diagonal GMM, Leiden, and HDBSCAN
   in square-root state proportions (Hellinger geometry). Refit the selected
   method visibly and assert agreement with the saved classes.
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

State comparisons use Leiden resolutions 0.4, 0.8, 1.2, and 1.6. Prefer the finest
partition with seed ARI >=0.8, ant-removal ARI >=0.65, at least five ants per state,
and minimum dictionary fraction 0.002. Stars flag individual states with matched
Jaccard below 0.5. If none passes, report the most reproducible option as
unsupported. These checks condition on the full-cohort PCA/scaling and constitute
perturbation stability, not independent held-out biological validation.

Ant comparisons allow K=2–8, plus variable-K graph/density methods. Prefer the
finest partition with at least three ants per class, no unassigned ants, median
ant-removal ARI >=0.8 and weakest-class Jaccard >=0.6, and median temporal ARI
>=0.65 and weakest-class Jaccard >=0.6. Use 30 ant-removal repetitions and
alternating 1-, 3-, 6-, and 24-hour blocks. Hours are validation blocks only;
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

## Full-recording state result

The new fit selects eight states at Leiden resolution 0.4 in 11 dimensions:
four full-body posture PCs (89.79% of posture variance), five motion measurements,
and mean/peak concurrent contacts. Repeat-fit ARI is 0.813; ant-removal ARI is
0.753, with minimum matched-state Jaccard 0.559. Each state occurs in at least
64 dictionary ants. Expanded representations and finer resolutions produced
9–24 states, but none passed all stability checks. This supports a compact
vocabulary at the tested scale, not a claim that ants perform only eight actions.

S0 and S1 are low-motion candidates. Their median body motion is about
0.012 mm/s and median antennal motion about 0.084 and 0.090 mm/s, respectively.
Only S0 enters the current putative-sleep plots and contact-response analysis;
both remain separate states in the ant-proportion analysis. S1 is not merged
into sleep. Ant classes and state assignments are unchanged by this interpretation.
They are quiet-enriched rather than pure immobility states: 62.6% and 57.5% of
their dictionary windows jointly satisfy the displayed motion limits. State
membership is the downstream label, without adding another motion threshold.
Missing observations censor quiet runs, so observed run lengths can substantially
underestimate uninterrupted quietness. Do not interpret
those missing-data boundaries as awakenings or claim physiological sleep validation.

The ant comparison selects four classes (10, 14, 8, and 12 ants). K-means and
Leiden at resolution 1 recover identical full-data assignments. K-means median
ant-removal ARI is 1.00; temporal ARI is 0.692 and weakest-class temporal Jaccard
0.727. The three-class solution is more consistent across time (ARI 0.935).
Four is the finest partition passing the stated rule, not an unequivocal count
of biological roles. Against the frozen two-group spatial reference, C0/C1 are
entirely in group 0, C3 entirely in group 1, and C2 contains two group-0 and six
group-1 ants. Thus 42/44 match their behavioral class's predominant spatial group;
this is a post hoc description, not held-out prediction accuracy.
The most temporally stable passing two-class fit (diagonal-covariance GMM)
contains 24 versus 20 ants and matches 42/44 frozen spatial labels after mapping
group identities. Its median temporal and ant-removal ARIs are both 1.00. The
report shows this broad grouping beside the selected four-class solution. The
tied-covariance two-class GMM matches all 44 spatial labels, but has slightly
lower temporal stability (ARI 0.955). The broad fit is chosen by behavioral
stability; spatial agreement does not select it or change the four downstream
classes. All these spatial comparisons remain descriptive.

## Current display and trip results

The UMAP comparison selects 8 neighbors and min_dist 0.5 by neighborhood
preservation (median trustworthiness 0.966; five-neighbor overlap 72.7% across
three seeds). All nine parameter settings show the same four fitted classes.
This display choice does not select the class partition.

The grid-occupancy trip rules identify 5,923 completed trips across the 44 ants.
Equal-ant means for C0/C1/C2/C3 are 16.2/17.2/68.2/162.1 completed returns per
recording day, and 34.5/31.0/382.5/679.2 minutes on completed trips per day.
The precise response-event rules retain 1,827 returns. These counts differ
because the trip-investment rules allow longer position gaps and less observed
support. They are separately named and should not be interchanged.

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
published dataset. Source metadata is recorded in `contact_label_audit_inputs.json`.
The standalone historical-label diagnostic has been removed; its committed
version remains in Git history. The supported report uses the current S0 and
S0-calibrated motion definitions described above.
