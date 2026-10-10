# Posture dynamics, behavioral states and ant roles

This workflow replaces the earlier K-means / transition comparison for the
July 24 recording. It pools time bins across ants, discovers states with Leiden,
reduces the feature set, and clusters ants using **state proportions only**.
Spatial occupancy and the existing sleep classifier are downstream comparisons.
There is no final PCA and no interactive HTML.

## Run and inspect

Install `eigenposture_leiden_requirements.txt` in a Python 3.12 environment.
Run from the repository root. The initial extraction needs the unfitted pose
cache; subsequent steps use small NPZ files and can run locally.

```bash
python -m analysis.eigenposture_wavelets \
  --pose-cache /path/to/pose_cache \
  --source /path/to/eigenposture_20261009 \
  --basis /path/to/task_states_interactions_20261010/task_states.npz \
  --output /path/to/run/clips
python -m analysis.eigenposture_leiden_features \
  --block /path/to/20260724/block01 \
  --source /path/to/eigenposture_20261009 \
  --clips /path/to/run/clips --output /path/to/run/binned
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python -m analysis.eigenposture_leiden \
  --source /path/to/eigenposture_20261009 \
  --inputs /path/to/run/binned --output /path/to/run/results
python -m analysis.eigenposture_leiden_sleep \
  --block /path/to/20260724/block01 --output /path/to/run/results
python -m analysis.eigenposture_leiden_plots \
  --source /path/to/eigenposture_20261009 \
  --inputs /path/to/run/binned --output /path/to/run/results \
  --clips /path/to/run/clips
python -m unittest analysis.test_eigenposture_leiden -v
```

`eigenposture_leiden_inspect.py` contains short numbered cells for inspecting the
saved arrays, measurements, state search, feature reduction, roles and sleep
comparison. The main fitting functions are in `eigenposture_leiden.py`; changing
the inspection plots does not repeat the analysis.
If interrupted after saving `minimal_dictionary.npz`, add `--resume-dictionary`
to the main command to finish assignment, role fitting and exports without
repeating the feature search.

The extraction reuses the posture basis fitted across the **full 48 hours** in
the previous script, with equal colony and ant weights. It validates identities,
tracking fingerprints, clip clocks and agreement with published posture means.
It does not reuse the old behavioral clusters. Interaction preparation validates
the published contact run and its tracking inputs. Five-minute onset counts
must exactly match the previous saved counts.

## Measured features

The broad set has 49 features in seven equally weighted families:

| Family | Features |
|---|---|
| Body velocity (8) | Forward/lateral unsigned peaks, RMS and signed standard deviations; forward/lateral RMS acceleration |
| Posture (8) | Mean scores of posture PCs 1–8 |
| Variation (8) | Within-clip and between-clip standard deviations of PCs 1–4 |
| Posture rate (4) | RMS time derivatives of PCs 1–4 |
| Wavelet (12) | Mean Morlet power at 2, 3 and 4 Hz for PCs 1–4 |
| Interaction (3) | New pair-contact bouts, distinct partners, fraction covered by any contact bout |
| Antenna motion (6) | Unsigned mean and peak speed of six landmarks, mean absolute forward/lateral components, separate mean speeds of the two tips |

Antennal speed is measured **relative to the head in body-axis coordinates**, in
mm/s. Taking the Euclidean norm before averaging prevents movements in opposite
directions from cancelling. This distinguishes body stillness from antennal
stillness. It removes translation and rotation of the animal; body velocity
remains a separate feature. The posture coordinates use a fixed scale per ant,
and multiplying their derivative by that scale restores physical units.

The existing cache samples **one 2.5-second clip per minute**, with 28 smoothed
samples at 12 Hz. Derivatives and wavelets stay within each clip, and reject
missing samples and camera changes. The finite-support complex Morlet kernel is
`exp(-t²/(2σ²)) * (exp(2πift) - weighted_mean_carrier)`, with
`σ = 3/(2πf)`, truncated at ±2σ and normalized to unit squared energy. There is
no padding or interpolation; each frequency requires three fully valid kernel
centers. These broad, overlapping bands describe rapid within-clip variation,
not precise spectral peaks or movements with periods of minutes. Smoothing
attenuates fast movement. The implementation tests DC rejection, frequency
response and invalid-support handling.

The posture-wavelet representation follows the general approach of
[Berman et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4233753/), but
these sparsely sampled ant clips support a much narrower temporal range.

Bins are 2, 5 or 15 minutes. Core features require at least 60% valid clips and
at least two clips. Each wavelet band needs at least 20% valid clips and at
least one, drawn from the same core-valid clips; its longer support otherwise
discards substantially more data. Per-feature support counts are saved. Body
and antennal peaks take the maximum over the sampled clips; other clip features
are averaged. Missing values are never filled with zero. All feature comparisons
within a bin width use exactly the same bins complete in the broad set.

Interactions come from continuous contact output, with a 0.1-mm skeleton
threshold and gaps up to two seconds merged. Bout occupancy includes these
merged gaps. Counts are observed detections, without correction for each ant's
tracking exposure. Missing published interaction coverage invalidates the bin.

Nonnegative features are transformed with `log1p(value / scale)`; PC means and
contact fraction remain linear. See `balanced_matrix` for the physical scales.
Each retained family receives total variance one, using one common divisor for
its coordinates. Features are not individually whitened, so tiny-variance PCs
do not acquire the same weight as dominant PCs. Family weights are recalculated
after feature removal. No additional dimension reduction precedes clustering.

## State selection and reduction

The dictionary is a fixed random 80% of valid bins, capped at 20,000. A weighted
30-neighbor graph is constructed in the balanced measured-feature space using
UMAP's fuzzy graph builder. **Leiden operates on this graph, not on a 2D UMAP.**
It uses the RB configuration objective at resolutions 0.1, 0.25, 0.5 and 1.0,
three iterations and three random seeds. The objective and resolution parameter
are documented in the [Leiden reference](https://leidenalg.readthedocs.io/en/latest/reference.html).

A partition qualifies if it has at least two states, median between-seed ARI
≥0.9, ARI ≥0.8 after removing a random 20% of ant identities and rebuilding the
graph, every state ≥2% of bins, every state observed in ≥10 ants and ≥3 ants
from each colony. The finest qualifying partition is selected per width; the
primary width has the most qualifying states, with stability breaking ties.
If none qualify, the most stable result is explicitly provisional. This is a
predefined descriptive resolution rule, not evidence of a unique biological K.
The optional `--check-convergence` mode on the main command reruns the selected
broad graph until convergence at three seeds, without changing its selection.

Feature reduction explains the selected broad dictionary with Extra Trees,
training on 70% of ant identities and evaluating permutation importance on
held-out ants. That importance predicts the broad labels; it is not causal
importance or independent validation. Correlated features can substitute for
one another. Each family is also removed in a full graph / Leiden refit.

Ranked prefixes of 4, 6, 8, 12, 16, 24, 32 and 49 features are tested, always
retaining unsigned forward and lateral peak speed and mean antennal speed for
physical interpretation. The smallest tested prefix preserving the broad K,
ARI ≥0.9, every matched-state Jaccard ≥0.75, and seed ARI ≥0.9 undergoes one
greedy deletion pass, with a full refit at each step. “Minimal” means this tested
reduction under the three-feature constraint, not a global optimum. All rejected
fits are exported. Remaining bins receive a five-neighbor inverse-distance vote
in the same feature space. Assignment confidence is saved; dictionary entries
are kept at their fitted label. States are numbered by mean forward peak speed.

UMAP explores neighbors 10/30/60 and minimum distance 0/0.2, with repulsion 2,
300 epochs and a fixed sample of up to 10,000 bins. The displayed view maximizes
2D silhouette among layouts with trustworthiness ≥0.95, falling back to the
most trustworthy view. This choice changes only the picture; all six layouts
show identical state labels. Apparent gaps or densities do not establish new
states. See the [UMAP parameter documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html).

## Ant roles and the sleep comparison

Each ant is represented by the square roots of its state proportions, retaining
every coordinate. Separately per colony, tied-covariance Gaussian mixtures test
K=1–4. Eligibility requires at least 12 observed bin-hours across 48 hours;
these are covered bin durations, not continuous posture observation hours.
K>1 needs BIC below K=1, every group ≥5 ants, median ARI ≥0.8 across 200 ant
bootstraps, and median ARI ≥0.6 between alternate 30/60/120-minute time blocks.
Choose the smaller eligible K within two BIC units of the best; otherwise K=1.
All 114 identities remain visible, including those below coverage thresholds.
These checks condition on the shared full-recording posture basis and dictionary.

Only after the state definition and roles are frozen are spatial labels opened.
Occupancy comparisons use the same eligible ant cohort. Spatial agreement cannot
choose the states, feature subset or number of roles.

The candidate sleep state is fixed before sleep-label loading: the state with
the lowest median summed log-transformed forward peak, lateral peak and mean
antennal speed. This merely identifies a candidate; an actual low-motion or
sleep-enriched state is not guaranteed. The comparison uses the existing
continuous ten-second trailing-window classifier: at least 90% quiet, body
75th-percentile speed ≤0.5 mm/s and antenna 75th-percentile speed ≤0.7 mm/s,
subject to the classifier's coverage rules. Its antenna measurement uses global
landmark motion, so that threshold is **not applied** to the new head-relative
speed. The exact classifier parameters and source fingerprints are exported.

Sleep fractions exclude unknown frames and require at least 50% classified
frames per bin. State means weight ants equally (at least three bins per ant
and state), with 1,000 ant-bootstrap confidence intervals. Adjacent-bin state
persistence excludes gaps. A paired comparison also contrasts candidate versus
other bins within the same ants, requiring three bins of each kind.
Movement-based sleep labels share information with
the clustering features: agreement supports the descriptive label “putative
sleep,” but is not an independent test of sleep or arousal threshold. A bin can
mix wake and sleep; sparse posture clips cannot establish continuous quiescence.

Outputs include a static multipage PDF, numbered PNGs, state and role arrays,
feature importance and ablation tables, every selection score, post hoc sleep
comparisons, frozen decisions and provenance. Prior analyses remain intact.

## July 24 result (2026-10-10)

**A sleep-enriched low-motion partition appears, but a robust state dictionary
and a preserving minimal feature set were not obtained in this search.**
None of the 12 broad width/resolution fits passes all support criteria. The
fallback is the two-minute, resolution-0.1, two-state partition: seed ARI 0.405
and ant-removal ARI 0.571. Running it to convergence still gives seed ARI 0.405.
These are exploratory labels, not established biological task classes.

There are 50,518 valid bins. Seven family ablations, eight ranked-prefix fits
and 46 single-feature deletion fits find no reduction meeting the preservation
and seed-stability criteria. All 49 features are therefore retained for the
exploratory plots. This does **not** show that all 49 features are necessary:
the broad reference itself is unstable, making its exact boundaries an
unreliable reduction target. The search does not support a claim that wavelets
or posture rates reveal additional reproducible states under these settings.

The low-motion candidate T0 was chosen without sleep labels. Comparison with
the existing continuous classifier gives:

| State | Median forward peak (mm/s) | Median unsigned antennal speed (mm/s) | Sleep-rule fraction, equal ant weights | Ants |
|---|---:|---:|---:|---:|
| T0, low motion | 0.034 | 0.112 | 57.0% (95% bootstrap CI 55.5–58.5%) | 88 |
| T1, more active | 0.755 | 0.512 | 16.0% (15.2–16.8%) | 87 |

Within 87 ants contributing both states, T0 has **40.8 percentage points more
sleep-rule occupancy** (95% ant-bootstrap CI 39.5–42.1). This supports a
putative sleep-enriched interpretation, but T0 mixes sleep and wake. Sparse
clips cannot establish continuous ten-second quiescence, and the comparison
shares movement information with the clustering.

Proportion-based role selection returns K=1 in both colonies, with 38 left and
43 right ants meeting coverage. It therefore does not recapitulate the two
spatial groups in this run. All 114 identities remain in the plots. The earlier
seven-feature results are preserved separately; the new result is not evidence
that spatial roles are absent.
