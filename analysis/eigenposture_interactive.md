# Five-minute behavior states and ant groups

Run the numbered cells in `eigenposture_interactive.py` to inspect each step.
For static PNGs, a PDF, and CSV/NPZ tables:

```bash
python -m analysis.eigenposture_task_report --output /path/to/results
```

No HTML is generated. The script uses NumPy, pandas, SciPy, scikit-learn,
Matplotlib, threadpoolctl and umap-learn. Extraction of the interaction cache
also needs the existing pipeline dependencies, including PyArrow and Shapely.

## State clustering

1. Fit landmark PCA over the full 48 hours, with equal colony and ant weights.
   Retain four posture PCs. No additional PCA follows.
2. For each ant and five-minute interval, take mean posture scores and maximum
   **unsigned** forward/lateral velocity. Absolute values are taken before
   averaging, from the original short-window motion samples. Require three
   jointly valid clips out of five. Each minute has one sampled 2.5-second clip;
   these are not continuous five-minute posture measurements.
3. Add a seventh feature: the number of **new undirected pair-contact bouts**
   starting in that five-minute interval. Each bout counts for both ants;
   repeated contact with the same partner can count again. Contacts use all
   skeleton segments/nodes within 0.1 mm, joining detections separated by at
   most two seconds. Left-censored onsets are excluded. These counts use the
   continuous interaction output, including times outside sampled posture
   clips. Missing interaction coverage stays missing. They are detection
   counts, not rates corrected for each ant's tracking exposure.
4. Pool bins across all ants, without identity, time or position as features.
   Apply `log1p(v/0.1)` to velocities and `log1p(count)` to interactions.
   Scale the velocity, posture and interaction families to equal total variance.
   Fit K-means in **all seven dimensions**. Select K=2–10 by silhouette on a
   fixed random sample of at most 3,000 bins. This selects a descriptive
   resolution, not proof of a particular number of biological tasks.
5. Display the states using a UMAP of 10,000 randomly sampled bins:
   `n_neighbors=20`, `min_dist=0.05`, Euclidean distance, 300 epochs, fixed seed.
   UMAP does not determine any cluster assignment or K. Its apparent gaps and
   densities are not additional evidence for discrete classes. See the
   [UMAP parameter documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html).
6. Assign every valid bin, display all 114 identities through time, and leave
   missing bins gray. State IDs are ordered by mean forward peak speed.

## Does temporal organization reveal additional ant groups?

Compare two explicitly defined ant representations, separately by colony:

- **Proportions:** square roots of each ant's state fractions.
- **Proportions + transitions:** the same fractions plus square roots of the
  normalized joint counts of ordered adjacent-bin state pairs. Diagonal pairs
  describe persistence; off-diagonal pairs describe switching. Proportions and
  transitions receive equal total between-ant variance, using one scale per
  family. Every coordinate is retained without PCA.

Only consecutive, observed five-minute bins contribute a pair. Missing bins,
including time blocks removed during validation, break the sequence. Two bins
on opposite sides of a gap are never joined. The transition representation
measures organization at five-minute resolution; it is not an explicit model
of long dwell-time distributions or transitions inside individual bins.

Both representations use the same shared-covariance Gaussian mixture and K=1–4
selection: BIC below K=1, minimum group size five, median ant-bootstrap ARI at
least 0.8, and median temporal-split ARI at least 0.6. Choose the smaller eligible
K within two BIC units of the best, or K=1. There are 200 ant bootstraps; temporal
splits alternate 30-, 60- and 120-minute blocks. BIC is compared **within each
representation**, not between representations of different dimensions.

The ant fit requires at least 12 observed bin-hours over the full 48 hours;
this retains 41 left and 45 right ants in the previous run. All identities
remain visible in timelines. Stability checks condition on the pooled state
labels and the full-recording posture basis; these are not independent holdouts.

The summary plots compare selected K and stability, group overlap, state
fractions, and conditional stay probabilities. A stay probability is the chance
of the same state in the immediately following bin, using only adjacent observed
pairs. Spatial classes and occupancy are loaded only after both fits and are
never used to select them. Groups describe this recording; extra groups alone
would not establish biological roles.

## Inputs and outputs

`DATA` is the published `eigenposture_20261009` folder. `VELOCITY_DATA` is the
previous `task_states_20261010/unsigned_velocity.npz`. `INTERACTION_DATA` is
`task_states_interactions_20261010/interaction_counts.npz`. Regenerate caches:

```bash
python -m analysis.eigenposture_unsigned_velocity \
  --pose-cache /path/to/pose_cache \
  --source /path/to/eigenposture_20261009 \
  --output /path/to/task_states_20261010/unsigned_velocity.npz
python -m analysis.eigenposture_interactions \
  --block /path/to/20260724/block01 \
  --source /path/to/eigenposture_20261009 \
  --output /path/to/task_states_interactions_20261010/interaction_counts.npz
python -m unittest analysis.test_eigenposture_task_states -v
```

The contact loader checks the published run, current tracking sources, contact
metadata, and cache keys before reuse. If staging/published path keys both
exist, their entire bout tables must agree exactly. The extractor records its
source fingerprints, cache hash, geometry, time origin and counting definition.

Useful arrays are `coordinate_modes`, `binned` (ant × 576 bins × 7), `task_input`,
`task_summary`, `umap_table`, `tasks`, `proportion_table`, `transition_table`,
`stay_probabilities`, `baseline_k_tables`, `ant_k_tables`, `comparison_table`,
and `assignments`. The exporter saves the same quantities and the clean script,
with CSV group-selection diagnostics for both representations. The original
six-feature results remain in `task_states_20261010`.

## July 24 result (interaction-expanded run, 2026-10-10)

The 33,736 valid bins select **three states**, with silhouette 0.293 versus
0.287 for two states: a small preference, not strong evidence for a unique K.
Their mean unsigned forward/lateral peaks and interaction counts are:

| State | Forward (mm/s) | Lateral (mm/s) | New contacts / 5 min | Bins |
|---|---:|---:|---:|---:|
| T0 | 0.646 | 0.515 | 50.40 | 14,538 |
| T1 | 1.340 | 0.699 | 10.66 | 5,749 |
| T2 | 4.919 | 2.169 | 46.20 | 13,449 |

The posture profiles also differ. T1 is relatively low in contacts; interaction
counts do not increase monotonically with speed. The labels remain descriptive.

| Ant representation | Left (41 ants) | Right (45 ants) |
|---|---|---|
| State proportions | K=2; sizes 24/17 | K=3; sizes 25/7/13 |
| Proportions + transitions | K=1 | K=1 |

Proportion groups have median bootstrap ARI 1.000 in both colonies, with tenth
percentiles 1.000/0.814; temporal-split medians are 1.000/0.773. Spatial ARI is
0.902 left and 0.748 right. The left two-group matching is 40/41; the right
comparison has three activity groups versus two spatial classes.

Adding transitions **does not support additional groups under this rule**.
For the transition representation, K=2 is worse than K=1 by 6.55 BIC units on
the left and 15.67 on the right. K=3 is worse by 35.35/17.41. Its 12 coordinates
(three fractions plus nine transitions) include related information and increase
the mixture's complexity penalty with only 41/45 ants. Thus K=1 here is not
evidence that biological roles are absent, and it does not establish that dwell
history could never help with a different temporal model.

The main timelines and spatial maps retain **proportion groups**. The comparison
figure shows the transition result separately, and the persistence profiles show
how the proportion groups differ in remaining in each state. In exports,
`baseline_groups` / `proportion_group` denote proportions, while `ant_groups` /
`ant_group` denote the transition comparison. All identities remain in timelines;
only the 86 adequately observed ants enter either ant-group fit.

Static figures are numbered: 1 PCA, 2 state K/centroids, 3 UMAP, 4 identity
raster, 5 ant K comparison, 6 group overlap/fractions/persistence, 7 grouped
raster, 8 ant state proportions, 9–10 spatial comparison. Results are in
`20260724/block01/analysis_outputs/task_states_interactions_20261010`.
The Python 3.12 environment is pinned in `eigenposture_task_requirements.txt`.
