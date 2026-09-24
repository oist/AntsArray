# Posture-first behavioral landscape: 20260724/block01

This analysis implements the order of inference suggested by the eigenworm
work: measure intrinsic shape, learn shape modes, describe short histories,
compare individual usage, then inspect external behavior. It does not fit
posture to the previous spatial groups. The ten tracked nodes cover the tag
anchor, head, petiole, gaster tip and antennae; they do not cover legs.

The implementation is scoped to this recording, its 24 Hz clock, its 0.016
mm/pixel calibration, and two matched 24-hour windows beginning July 24 at
10:00. It rejects a different recording instead of silently reusing those
assumptions. A general analysis of other dates should make those parameters
explicit and repeat validation.

## Reproduce

Use the repository environment with NumPy, pandas, SciPy, scikit-learn,
matplotlib, pyarrow and pytest. Exact versions are recorded in each completed
run's `run_manifest.json`.

```bash
python -m analysis.postural_dynamics_extract \
  --block /bucket/ReiterU/Ants/basler/20260724/block01 \
  --prepare /path/to/run

# Run once for every index in tasks.json (the 0724 inventory selects 86).
# Independent array tasks are safe. Write caches to a compute-writable disk.
python -m analysis.postural_dynamics_extract \
  --tasks /path/to/run/tasks.json --task-index 0 \
  --output /path/to/pose_cache

python -m pytest analysis/test_postural_dynamics.py -q

python -m analysis.postural_dynamics \
  --tasks /path/to/run/tasks.json --pose-cache /path/to/pose_cache \
  --block /bucket/ReiterU/Ants/basler/20260724/block01 \
  --spatial-atoms /bucket/ReiterU/Ants/basler/20260724/block01/analysis_outputs/long_timescale_0723_0724_0729_20260915/temporal_clusters_4h/atoms/2 \
  --output /path/to/results

# Rebuild figures from an existing completed run without refitting models.
python -m analysis.postural_dynamics_plots \
  --block /bucket/ReiterU/Ants/basler/20260724/block01 \
  --output /path/to/results
```

The extraction streams each long-format per-ant parquet once. Its compact
cache stores eight body-relative unit directions, segment lengths for QC,
camera IDs and global clip-start frames. It stores no arena positions,
absolute headings, translation speeds, sleep labels or spatial groups.
Sampling uses the full recording length in parquet metadata, not the
per-ant observed span in speed metadata. A deterministic random 2.5-second
clip is selected within each minute of the two days.

The original >40% detection screen uses each ant's observed span, matching
the existing analysis cohort. Fitting additionally requires at least 240
fully measured day-1 clips distributed over at least 24 half-hours. This
threshold is applied before spatial outcomes are loaded. Day-2 availability
does not determine inclusion in training. Missing points, duplicates,
camera switches and degenerate segment lengths invalidate the entire clip;
there is no interpolation. Segment QC accepts a 0.1–1.25 mm body axis and
0.05–2 mm analyzed segments. These broad bounds remove obvious failures,
not all tracking noise. There are no finished per-node confidence scores.

Four-frame causal filtering precedes 12 Hz sampling. Unit directions are
renormalized after filtering. History ends at raw frame 52; its future target
is raw frame 58. The smoothing windows therefore do not overlap. Tests
explicitly perturb future frames to guard against leakage, and verify
translation/rotation/scale invariance while preserving head articulation.

PCA retains 95% of direction-cosine variance and balances ants. Every fourth
hour of day 1 is withheld for short-history prediction and dictionary
resolution selection. A fixed one-second primary history is compared with
instantaneous posture, shuffled past times and mean-removed dynamics. The
motif dictionary size is chosen from 12/24/48 using the smallest within one
standard error of the best future-posture prediction. The final basis and
dictionary are refitted on day 1 and applied unchanged to day 2.

Ants are represented by square-root motif frequencies and clustered
separately within the two colonies. K=2 is a prespecified comparison;
K=2–6, whole-ant bootstrap stability and a continuous profile alternative
are also reported. The selection rule requires a minimum group size of four
and median bootstrap ARI at least 0.8, then chooses the smallest K within
0.02 of the best eligible silhouette. A result of K=1 means no candidate
passed, rather than a formal test of unimodality. P0/P1 labels follow median
angular motion, without consulting spatial outcomes.

`SPACE_BLIND_FROZEN.json` records model/group hashes before any spatial
files are opened. Only then are the exact 0724 spatial atoms, previous
fine-grid groups, arena annotations and measured behavior loaded. The
existing spatial groups are a descriptive comparison, not training labels.
Spatial forecasting predicts an ant's second-day map from the first-day
maps of *other* members of its posture group. The baseline uses all other
ants. Both days need at least 40% position coverage. Uncertainty resamples
whole ants and the null permutes whole-ant labels. The observed association
does not imply that posture causes spatial occupancy.

## Deliverables

- Seven figure pairs (PNG/PDF), a combined PDF and `REPORT.md`.
- `index.html`, a linked gallery, and `explorer.html`, a self-contained
  offline mode/motif/individual explorer with no external dependencies.
- Quality, model selection, prediction, group assignment and post hoc
  comparison tables; compact models, measured motif examples and maps.
- Frozen model hashes, source fingerprints, exact software versions and
  a completion marker written only after all deliverables succeed.

The extraction cache is reproducible from the original tracking files and
is not a Git artifact. Publish completed results alongside the recording;
commit the analysis source and tests. `colony_behavioral_landscape.py` is a
tracked dependency supplying whole-ant bootstrap and Hellinger helpers.

Methodological inspiration: [Stephens et al., 2008](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000028)
and [Costa, Ahamed, Jordan and Stephens](https://arxiv.org/abs/2105.12811).
This implementation does not reproduce the latter's full transfer-operator
construction or claim to identify maximally predictive states.
