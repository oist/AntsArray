# Five-minute task states and ant division of labor

Open [`eigenposture_interactive.py`](eigenposture_interactive.py) and run its
numbered `# %%` cells in order. It uses ordinary NumPy, pandas, SciPy,
scikit-learn and Matplotlib, with no analysis-pipeline imports. It leaves arrays
and figures available for inspection and does not automatically write outputs.

## What each step does

1. Load all 114 tracked identities, aligned antennal coordinates, and unsigned
   forward/lateral motion measurements. Refit the four posture PCs over the
   full 48 hours, using the same equal-colony/equal-ant moment calculation.
2. Form one six-dimensional vector for each ant's five-minute bin: four mean
   posture-PC scores and maximum unsigned forward/lateral speed. Absolute value
   is taken on the original short-window velocity components **before** any
   averaging. These are maxima of sampled, smoothed velocities, not continuous
   five-minute observations. Each minute contributes one 2.5-second clip.
3. Require at least three clips jointly observed in all six channels per bin.
   Missing bins remain missing. Pool valid bins from all ants, both colonies,
   and both days. Identity, time and spatial position are not input features.
4. Apply `log1p(v/0.1)` to unsigned speeds and balance the velocity/posture
   families to equal total variance. Fit K-means in all six dimensions. Select
   among K=2–10 using silhouette on a fixed sample of up to 3,000 pooled bins.
   This chooses a descriptive behavioral resolution; it does not infer the
   number of biological tasks or test whether behavior is intrinsically discrete.
5. Assign every valid bin to a state. Display all ants across 48 hours, with
   five-minute columns and gray for missing bins. State labels T0, T1, etc. are
   ordered by unsigned forward speed for display, not named as biological tasks.
6. Count each ant's fraction of observed bins in each state. Fit shared-covariance
   Gaussian mixtures to the square-root proportions, separately by colony,
   retaining every task coordinate without another PCA. Require 12 observed
   bin-hours across the 48-hour recording for this ant-level fit. Lower-coverage
   ants remain visible in the task timelines and proportion plots.
7. Choose ant-group K=1–4 using BIC, minimum group size five, median ant-bootstrap
   ARI at least 0.8, and median temporal-split ARI at least 0.6. Select the smaller
   qualifying K within two BIC units of the best, or K=1 if none qualifies.
   Bootstrap refits resample ants; time splits alternate 30-, 60-, and 120-minute
   blocks. These checks condition on the pooled task dictionary. Both days enter
   that dictionary and the posture PCA, so this is not an independent holdout.
8. Reorder the timelines by ant group and plot each ant's task proportions.
   Only then load spatial classes and compare occupancy on the same eligible
   ants. Neither stage is chosen for spatial agreement; two ant groups are not
   forced. State categories and ant groups describe this recording, not proven
   biological task identities or lifelong castes.

## Current result

The current run has **33,736 valid bins and two behavioral states**. Mean
forward/lateral peak speeds are 0.55/0.45 mm/s in T0 and 4.53/2.02 mm/s in T1;
their posture-PC profiles also differ. K=2 has silhouette 0.378, versus 0.261
for K=3. This is the lowest candidate resolution, not evidence for exactly two
biological tasks.

Ant task proportions select **three groups in each colony**: 24/8/9 ants on the
left and 25/7/13 on the right. Their mean fractions of T1 bins are 32/65/79% and
24/47/77%, respectively. The BIC advantage of three versus two groups is small
(2.35 left, 2.12 right). K=2 is also stable; do not treat the third group as an
established biological caste. For the selected K=3, median ant-bootstrap ARI is
1.000 left and 0.824 right; tenth percentiles are 1.000 and 0.648. Temporal-split
medians are 0.960 and 0.668. The right division is less stable.

The spatial contingency is consistent with the lowest-activity group occupying
one spatial class and both higher-activity groups mainly occupying the other:

| Colony / activity group | Spatial 0 | Spatial 1 |
|---|---:|---:|
| Left G0 | 23 | 1 |
| Left G1 | 0 | 8 |
| Left G2 | 0 | 9 |
| Right G0 | 25 | 0 |
| Right G1 | 2 | 5 |
| Right G2 | 0 | 13 |

Spatial ARI is 0.726 left and 0.748 right. These are three-to-two comparisons;
the script does not force a one-to-one class match or merge groups using space.

All 114 identities appear in the timelines. Forty-one left and 45 right ants
have at least 12 observed bin-hours and enter the ant-level fit; the remaining
16/12 identities have under one observed bin-hour. An initial 24-hour cutoff
excluded substantially observed ants (only 25 left and 43 right remained), so
the final cutoff follows this coverage gap. That earlier run is archived in
`strict_24h/`; its outcomes were not used to select the better spatial match.

## Inspectable arrays

| Variable | Meaning |
|---|---|
| `coordinate_modes` | Full-recording posture basis |
| `binned` | Ant × 576 five-minute bins × six features |
| `clip_counts` | Jointly observed clips per bin |
| `task_input` | Pooled valid bins, balanced in six dimensions |
| `task_k_table`, `task_summary` | State-resolution scores and physical feature means |
| `tasks` | Ant × time state labels; -1 is missing |
| `bin_table` | Every ant/bin, timestamp, task, coverage, and six measurements |
| `proportion_table` | Each ant's distribution of observed task states |
| `ant_k_tables`, `ant_bootstrap` | Ant-group selection and stability diagnostics |
| `assignments` | Coverage, ant groups, and subsequent spatial comparison |

`MIN_CLIPS`, `MIN_ANT_HOURS`, `TASK_KS`, and `BOOTSTRAPS` are near the top.
The defaults use 200 ant-bootstrap refits per candidate K; 20 is a quick preview.
Rerun cells in order after changing a setting. Keep spatial labels out of choices.

## Inputs and reproduction

`DATA` points to the published `eigenposture_20261009` folder. The new small
`unsigned_velocity.npz` sits in the neighboring `task_states_20261010` folder.
It was rebuilt from the original motion samples because taking the absolute
value of a signed clip mean would lose reversals within a clip.

To regenerate that cache from the existing unfitted pose cache:

```bash
python -m analysis.eigenposture_unsigned_velocity \
  --pose-cache /path/to/pose_cache \
  --source /path/to/eigenposture_20261009 \
  --output /path/to/task_states_20261010/unsigned_velocity.npz
python -m unittest analysis.test_eigenposture_task_states -v
python -i analysis/eigenposture_interactive.py
```

To export the timelines, feature tables, plots, PDF and a standalone hoverable
HTML report, run:

```bash
python -m analysis.eigenposture_task_report --output /path/to/new/results
```

The current report is in
`20260724/block01/analysis_outputs/task_states_20261010/index.html` under the
basler bucket. It includes all 114 identities, colony/coverage/ant filters, and
bin-level hover details. `five_minute_tasks.csv.gz` includes every bin, with
`task=-1` for insufficient observations. Use `--from-results` to rebuild just
the HTML from an already exported run.

The extractor reuses the original body-axis, camera-continuity, and speed QC;
requires at least eight valid velocity samples per clip; validates the current
tracking source sizes/timestamps; and checks recomputed signed clip means against
the original published cache before saving unsigned peaks. Its companion JSON
records source hashes and the statistic. Peaks are sensitive to extreme values
and sampling; unobserved motion cannot be recovered from this sampled recording.
