# Full-body states, right-colony ant classes, and state-derived sleep

Open `analysis/eigenposture_interactive.py` and run its ten `# %%` cells in
VS Code/Jupyter from the repository root. It uses the new full-recording model
packet and continues through the established grid-occupancy analyses. Plots are
ordinary Matplotlib figures, saved as PNGs; the batch runner also writes a PDF.
The costly extraction and model comparisons are in four small companion modules.

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
4. **Ant representation:** count each ant's assigned state observations directly,
   then normalize. Counts weight variable-length windows by their observed
   duration. No equal-hour averaging, additional eligibility filter, or final PCA.
5. **Ant classes:** compare K-means, Ward, tied/diagonal GMM, Leiden, and HDBSCAN
   in square-root state proportions (Hellinger geometry). Refit the selected
   method visibly and assert agreement with the saved classes.
6. **Inspect:** per-ant state proportions and detection versus classified coverage.
7. **Space:** hand the fixed classes to existing grid utilities; require an exact
   side/ID/filename match for every classified ant. Spatial labels are read for
   the first time here, solely for post hoc comparison. Plot mean/individual
    occupancy and nest use. Never overwrite the original spatial classes.
    The published packet carries the original two-group spatial reference.
    The current grid-label file has three right-colony groups; this is a different
    spatial partition. A new packet without a saved reference snapshots that
    current file instead. Neither reference affects any model fit.
8. **Speed and putative sleep:** interpret quiet clusters from their measured
   body and antennal motion, then derive frame labels from those state assignments.
   Plot quiet-run durations, class curves, and individual heatmaps.
9. **Clock profiles:** activity and state-derived quietness by time of day,
   using complete light-dark cycles when available.
10. **Returns and contacts:** retain the same ant classes and use state-derived
    quietness for recipient selection, matched controls, and response curves.
    Other right ants remain social context, with unknown sleep where ineligible.

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

Quiet-state interpretation defaults to median unsigned body peak <=0.05 mm/s
and median mean-antennal speed <=0.15 mm/s, applied **after** clustering. Inspect
the scatter/table and edit `SLEEP_STATES` to test an explicit set. These cutoffs do
not train states. `MIN_SLEEP_SECONDS=0` shows cluster membership directly; the
bout plot separately assesses persistence. Unknown intervals remain unknown and
censor bouts. They never count as awakenings. This is putative sleep without an
arousal assay. **No pre-existing cached sleep labels enter these analyses.**
The old motion cache metadata supplies only frame alignment for downstream readers.
Ant classes were fitted to these same state proportions: differences in total
quiet time describe the classes and do not independently validate them.
Contacts also enter the state features. A quiet-state label changing after a
contact is therefore partly definition dependent, and a one-second feature
window can include measurements on both sides of the contact time. The recipient
comparison is descriptive; inspect the body/antennal motion traces and do not
treat it as an independent demonstration of contact-induced awakening.

Median classified coverage is 52.8% of recording time for left ants and 49.8%
for right ants, despite their higher detection coverage. Complete skeletons,
valid motion support, and the 250-ms minimum are needed for these features.
Short-tail exclusion removes 3.60% of otherwise feature-valid samples. Missing
posture can be behavior dependent; pooled proportions describe observed behavior.

## Rebuild from pipeline outputs

Choose a new output directory when changing measurement or modeling choices.
Use local disk for roughly 20 GB of intermediate arrays, not a quota-limited
remote home. Existing stitched tracks, speed metadata, published skeleton
interaction outputs, and downstream grid/motion caches are required.

```bash
BLOCK=/path/to/20260724/block01
RUN=/local/scratch/full_body_behavior
python -m analysis.full_behavior_extract --block "$BLOCK" --output "$RUN" --workers 6
python -m analysis.full_behavior_states features --root "$RUN" --workers 4
python -m analysis.full_behavior_states learn --root "$RUN"
python -m analysis.full_behavior_states assign --root "$RUN"
python -m analysis.full_behavior_roles --root "$RUN"
ANTS_DATASET_ROOT="$BLOCK" python -m analysis.eigenposture_task_report \
  --source "$RUN" --output "$RUN/report"
```

The published packet retains features, model comparisons, assignments, PCA,
coverage/provenance, classes, derived quiet labels, and the report. Large dense
pose/motion arrays are reproducible local intermediates. The preceding eight-state
sampled-clip analysis remains documented in `fine_behavior.md` as a historical
result; its state numbers are not interchangeable with this rebuilt dictionary.

## Full-recording state result

The new fit selects eight states at Leiden resolution 0.4 in 11 dimensions:
four full-body posture PCs (89.79% of posture variance), five motion measurements,
and mean/peak concurrent contacts. Repeat-fit ARI is 0.813; ant-removal ARI is
0.753, with minimum matched-state Jaccard 0.559. Each state occurs in at least
64 dictionary ants. Expanded representations and finer resolutions produced
9–24 states, but none passed all stability checks. This supports a compact
vocabulary at the tested scale, not a claim that ants perform only eight actions.

New S0 and S1 are the quiet candidates. Their median body motion is about
0.012 mm/s and median antennal motion about 0.084 and 0.090 mm/s, respectively.
They are combined for the state-derived putative-sleep plots; both remain
separate states in the ant-proportion analysis.
They are quiet-enriched rather than pure immobility states: 62.6% and 57.5% of
their dictionary windows jointly satisfy the displayed motion limits. State
membership is the downstream label, without adding another motion threshold.
About 67–77% of right-colony quiet runs end at missing observations, so observed
run lengths substantially underestimate uninterrupted quietness. Do not interpret
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
The most temporally stable passing two-class fit (tied-covariance GMM) matches
the frozen spatial groups exactly: 26 versus 18 ants, ARI=1.00. The report shows
this broad grouping beside the selected four-class solution. The broad fit is
chosen by behavioral stability, not by spatial agreement; the spatial comparison
remains descriptive and does not change the four classes used downstream.
