# Fine-scale behavior discovery, 20260724/block01

This analysis compares **0.5, 1 and 2 second windows**, with assignments every
0.25 seconds. It learns states across ants first, then describes ant repertoires
using state proportions. Spatial occupancy and existing sleep labels do not
choose features, clusters, Leiden resolution, or ant roles.

## What the literature changes

Recent methods support analyzing movement sequences, accounting for observation
noise, and checking whether a behavioral vocabulary recurs across animals. They
do not imply that an attractive embedding or a large cluster count is sufficient.

| Primary study | Relevant idea | Application here |
| --- | --- | --- |
| [Keypoint-MoSeq, Weinreb et al., 2024](https://www.nature.com/articles/s41592-024-02318-2) | A generative model separates keypoint noise from pose dynamics; movement boundaries and the chosen timescale matter. | Explicit gap/camera checks, short smoothing, noise diagnostics and temporal examples. This is **not** a fit of Keypoint-MoSeq. The stitched files do not retain SLEAP confidence scores. |
| [Mlost et al., Patterns, 2025](https://pubmed.ncbi.nlm.nih.gov/40486967/) | Compares B-SOiD, BFA, VAME and Keypoint-MoSeq; method and granularity affect the behavioral descriptions. | Compare representations on the same observations; use different ants for validation and testing, not UMAP appearance. |
| [Mapping the landscape of social behavior, Klibaite et al., Cell, 2025](https://pubmed.ncbi.nlm.nih.gov/40043703/) | Fine social descriptions combine resolved body kinematics, contact and joint behavior. | Retain contact context separately from motion. An interaction count alone cannot distinguish the motor actions occurring during contact. |
| [CLOSER, published December 2025, Nature Communications 2026](https://www.nature.com/articles/s41467-025-67559-x) | Learns representations of motion through self-supervision and kinematic augmentation. | A learned temporal representation is an alternative to adding more scalar averages. CLOSER itself is not fitted or benchmarked here. |
| [DISK, published December 2025, Nature Methods 2026](https://www.nature.com/articles/s41592-025-02893-y) | Learns to infer missing skeleton coordinates using temporal and other-keypoint information, with artificial missingness during training. | Missing data can selectively erase active behavior. Longer imputation would need held-out-coordinate validation; this run only interpolates short, bounded gaps. |
| [VAME, Luxem et al., 2022](https://www.nature.com/articles/s42003-022-04080-7) | Represents egocentric pose trajectories with a recurrent variational model and segments its latent time series. | Compare ordered short trajectories with simple window summaries. No VAE or HMM is fitted in this run. |
| [MotionMapper, Berman et al., 2014](https://pubmed.ncbi.nlm.nih.gov/25142523/) | Multiscale posture dynamics can expose stereotyped movement patterns. | Compare normalized wavelet-power patterns, preserving frequency/PC structure rather than summing it into activity. This is an adaptation, not a reproduction of MotionMapper. |

## Inputs and time base

- All 114 tracked identities in the existing coverage inventory are attempted.
  Empty/insufficient tracks are listed, not silently omitted.
- One uniformly random, continuous 60-second segment per clock hour per ant,
  covering the existing 48-hour window, July 24 10:00 to July 26 10:00.
  This schedules 91.2 ant-hours. Sampling does not consult behavior labels.
- Raw 24 Hz stitched keypoints; five-frame quadratic Savitzky–Golay smoothing,
  sampled at 12 Hz. Up to three missing raw frames (125 ms) may be interpolated
  only between observations with the same known camera throughout the gap.
  Longer gaps, camera changes, duplicate frames and implausible geometry are
  excluded. This can still leave tracking noise and missingness bias.
- The four antennal posture PCs use the previously fitted **full 48-hour** basis.
  Consequently this fixed basis has seen the evaluation ants; subsequent state
  fitting and feature normalization have not. The experiment evaluates state
  discovery on a shared posture coordinate system, not completely unseen pose
  representation learning.
- Antenna coordinates are head-relative and body-aligned. Absolute arena
  coordinates are never features. Forward/lateral body speed and each antenna's
  speed are unsigned. Antennal speed averages its three tracked landmarks and
  is relative to the body, unlike the global antennal speed in the sleep rule.
- Contact features come from the fingerprint-validated published pair-bout
  cache. Its bouts bridge gaps up to two seconds: contact occupancy is **bout
  context**, not a claim of frame-precise physical contact at subsecond scale.

## Representations

The baseline has 16 coordinates: mean PC1–4, head/gaster direction cosines,
peak unsigned forward/lateral velocity, mean unsigned velocity of each antenna,
mean unsigned turning speed, mean concurrent contact partners, contact-bout
fraction and new contact-bout count.

1. **Kinematics:** those 16 coordinates.
2. **Ordered trajectory:** adds five ordered, mean-subtracted samples of the
   four PCs and four articulation coordinates (40 more coordinates). This
   preserves temporal variation and direction through posture space.
3. **Normalized spectrum:** adds posture-PC power at 0.5, 0.8, 1.2, 1.8, 2.7 and
   4 Hz. A regularizing floor (training fifth percentile of total power) becomes
   an additional channel before probability normalization and square rooting.
   This represents spectral shape without equating larger amplitude with every
   kind of movement. The floor limits, but cannot eliminate, noise amplification.

Each feature family has one training-fitted total-variance scale. All motion
amplitude features share one family, rather than each correlated summary
receiving its own full weight. There is **no final PCA** and no clustering in a
two-dimensional embedding.

Wavelet coefficients require their entire finite kernel to lie within observed
data from one camera. The 0.5 Hz kernel spans approximately 6.5 seconds; averaging
it over a two-second window requires roughly 8.5 seconds of context. A 0.5-second
summary does not make this a purely subsecond measurement.

The initial comparison uses identical centers for all representations and widths.
The follow-up removes the slow-wavelet gate and compares kinematics/trajectories
on common two-second support. The audit also records how many individually supported
0.5/1/2-second windows are available. It is essential to inspect the resulting
speed and coverage differences before interpreting state proportions.

## Clustering and interpretation

Eligible ants (at least 100 common windows) are split within colony into 60%
training, 20% validation and the remainder test. Dictionary centers lie on a
four-second grid, with a cap of 240 randomly selected windows per training ant.
This cap reduces imbalance but does **not** give every ant equal weight. Wavelet
contexts may overlap; evaluation is ant-disjoint to avoid treating adjacent
windows as independent validation examples.

Leiden operates on a 30-neighbor fuzzy graph in the complete balanced feature
space. Resolutions 0.5, 1 and 2 are compared, with three random starts and a
refit omitting 20% of training ants. The initial screen uses three iterations;
the short-support follow-up and wavelet convergence audit run to convergence.
Stored protocols distinguish these experiments.

Before fitting, the exploratory acceptance rule is fixed: minimum seed ARI
0.75, validation-ant agreement ARI 0.65, minimum dictionary fraction 0.5%, and
at least five training ants per state. Choose the most detailed passing
partition, then validation/seed agreement. If none passes, the highest validation
agreement is plotted explicitly as **provisional**. These thresholds are
analysis choices, not a statistical estimate of the number of natural behaviors.
Test ants are evaluated only after freezing that choice within each experiment.
Comparisons between experiments remain exploratory.

After selection, five additional random omissions of training ants check the
same frozen dictionary on the test cohort. They do not reselect the model.
The role sensitivity check removes the weakly matched state's proportion,
renormalizes the remaining proportions, and refits the chosen role count on
the unchanged eligible cohort. This is a sensitivity check, not another search
for a better spatial match.

Five-neighbor weighted votes assign supported windows to the dictionary. Vote
agreement is not a calibrated probability of a biological behavior. Original
dictionary memberships are retained. State IDs are reordered by body speed for
readability only. Three UMAP parameterizations show identical assignments;
neighborhood trustworthiness is reported, but layouts never affect the states.

Summary plots include measured feature profiles, real antennal sequences from
different ants, gap-preserving ethograms, camera/noise diagnostics, and a post
hoc sleep comparison. A low-motion candidate is chosen by body/antenna speed
before opening sleep labels. Movement-defined sleep labels cannot establish
arousal threshold or validate sleep independently.

Ant role analysis uses only mean observed-hour state proportions (at least 20
windows/hour in at least 24 hours). It retains the existing tied-covariance GMM
K=1..4, BIC, group-size and bootstrap/temporal stability rules, with 50 exploratory
bootstraps. If fewer than 10 ants qualify in a colony, no role fit is attempted.
These are **conditional sampled repertoires**, not whole-day time budgets.

The present skeleton has body and antenna landmarks but no legs. Neither a
cluster count nor an antennal trajectory alone justifies naming a state grooming,
feeding, brood care, etc. State videos and expanded tracking would be needed to
validate those action names. A stable continuous gradient may also be a better
description than sharply bounded actions.

## Reproduce

Run from the repository root using the pinned environment in
`analysis/eigenposture_leiden_requirements.txt` (Python 3.12 was used locally).
Extraction also ran with Python 3.11 on Deigo; source timestamps and sequence
hashes are recorded. Raw tracking and the referenced analysis inputs reside on
the lab bucket and are not distributed through GitHub.

```bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
block=/home/sam-reiter/bucket/ReiterU/Ants/basler/20260724/block01
source=$block/analysis_outputs/eigenposture_20261009
basis=$block/analysis_outputs/task_states_interactions_20261010/task_states.npz
run=/path/to/new/fine_behavior_run

python -m analysis.fine_behavior_extract --block "$block" --source "$source" \
  --basis "$basis" --output "$run/sequences" --workers 8
python -m analysis.fine_behavior_features --block "$block" --source "$source" \
  --sequences "$run/sequences" --output "$run/features"
python -m analysis.fine_behavior_states --features "$run/features" \
  --source "$source" --output "$run/models"
python -m analysis.fine_behavior_short_support prepare --block "$block" \
  --source "$source" --sequences "$run/sequences" --output "$run/short_features"
python -m analysis.fine_behavior_short_support run --features "$run/short_features" \
  --source "$source" --output "$run/short_models"
python -m analysis.fine_behavior_short_support run --wavelet --features "$run/features" \
  --source "$source" --output "$run/wavelet_converged"
python -m analysis.fine_behavior_report --block "$block" --source "$source" \
  --sequences "$run/sequences" --features "$run/short_features" \
  --models "$run/short_models" --output "$run/report_short"
python -m analysis.fine_behavior_audit --features "$run/short_features" \
  --models "$run/short_models" --report "$run/report_short" \
  --wavelet-features "$run/features"
python -m unittest analysis.test_fine_behavior -v
```

To plot another experiment, pass its feature and model directories to the report
command. All exports are PNG/PDF/CSV/NPZ/JSON; no interactive HTML is generated.
The functions can also be imported directly for step-by-step inspection.
