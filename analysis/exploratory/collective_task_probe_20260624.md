# Collective behavior probe: 20260624/block02

Exploratory analysis run on 2026-09-08. The strongest lead is how persistent
individual movement propensities combine with changing collective activation.
These are within-recording observations and hypotheses, not established task
identities or across-day findings.

## What this dataset supports

The stitched recording spans **12.63 hours**, approximately June 24 20:37 to
June 25 09:15, at 24 frames/s. It contains 146 track files; requiring detection
in at least 40% of the entire recording retains 46 left-colony and 49
right-colony ants. Further comparisons use matched, sufficiently observed ants.
Crossing midnight does not provide two days of replicated behavior.

This is also a perturbation experiment: the session log contains **30 five-second
PWM/vibration pulses**, ten minutes apart, approximately 20:42–01:32. I aligned
events using logged camera frame numbers. The embedded device timestamps differ
from the recording clock, so using their time strings directly would misalign
responses.

Two preprocessing problems prevent transferring the previous foraging analysis
unchanged:

- Existing colony-presence vectors report zero inside-colony time for every
  track. Their inherited colony rectangles do not match the new panorama
  coordinates. There is no block-specific `panorama_regions.csv` here.
- The inherited grid split and physical scale also need checking. I used raw
  panorama positions, and converted stored speeds back to pixels/s. Absolute
  physical speeds are not validated. The existing speed filter still imposes
  a cutoff equivalent to 312.5 panorama pixels/s.

Consequently, movement is a **task-allocation proxy**, not proof of foraging,
brood care, inactivity, or sleep. A stationary ant may be working. Missing
detections are not treated as inactivity. Correct annotations and calibration
are prerequisites for genuine trip/resource comparisons on this block.

## Findings worth following

### 1. Persistent individual movement propensities

Comparing equal-duration early and late halves, individual mean-speed rankings
are strongly preserved: Spearman rho **0.86 left / 0.82 right**, using 40/39
matched ants. Ants also tend to revisit their own spatial distributions:
same-ant early/late spatial distances average 0.576/0.571, versus 0.730/0.701
for different-ant comparisons. These are Jensen–Shannon distances between
normalized occupancy histograms, not distances traveled.

This establishes a useful baseline: collective changes occur on top of
substantial individual persistence. It does not establish discrete jobs, and
the halves differ in stimulation and clock time.

[Figure: individual movement persistence and two-hour rank histories](</home/sam-reiter/bucket/ReiterU/Ants/basler/20260624/block02/analysis_outputs/collective_task_probe_20260908/03_movement_allocation.png>)

### 2. The morning increase amplifies existing movement allocation

Both colonies show an abrupt activity rise around **05:36**. For matched ants,
comparing 05:09–05:29 with 05:45–06:05:

| Measurement | Left | Right |
| --- | ---: | ---: |
| Matched ants | 40 | 32 |
| Colony mean-speed increase | 2.68-fold | 2.75-fold |
| Ants whose mean speed increases | 80% | 72% |
| Earlier more-active half: speed increase | 2.95-fold | 3.37-fold |
| Earlier less-active half: speed increase | 1.85-fold | 1.68-fold |
| More-active half's share of movement, before → after | 75% → 83% | 63% → 77% |

The halves are defined independently using activity from 01:45–04:45. The
earlier more-active half supplies **84%/82% of summed positive individual speed
increases**; negative changes are excluded from that particular denominator.
Thus the result is not just a larger absolute increase among initially faster
ants: their proportional increase and their share of movement are also larger.

An independent calculation of one-second displacements confirms the rise.
Tracking coverage actually falls over it, rather than improving. Nevertheless,
position/state-dependent visibility and residual tracking artifacts remain
possible, and the comparison windows were chosen after seeing the event.

Interpretation: increased collective movement is carried mainly by previously
more-mobile ants. This is a promising allocation pattern, but not evidence that
the colony faced increased work demand. The near-simultaneous onset in separate
colonies strongly motivates checking lighting, disturbance, and acquisition
logs before invoking social coordination or an endogenous daily rhythm.

[Figure: morning increase and each ant's contribution](</home/sam-reiter/bucket/ReiterU/Ants/basler/20260624/block02/analysis_outputs/collective_task_probe_20260908/07_morning_allocation.png>)

### 3. Repeated stimulation changes responses within the same ants

Define response as speed 10–60 seconds after pulse onset minus speed 60–10
seconds before it, excluding the pulse itself. Among ants with adequate
observations in both early and late trials, mean response to the first five
versus last five pulses declines:

- Left: **13.34 → 1.72 px/s**, 31 matched ants.
- Right: **20.57 → 1.41 px/s**, 22 matched ants.

Decline remains after adjusting for initial speed and its square, commanded
duty, tracking coverage, and ant identity. Estimated decline per ten trials is
5.20 px/s on the left and 8.74 on the right; trial-clustered 95% intervals are
3.01–7.38 and 6.60–10.87, respectively.

Individual adjusted response rankings have some persistence between independent
15-trial halves: rho **0.63 left / 0.45 right**, 38/30 ants. Nuisance models are
fitted separately in each half without an ant-identity term. This suggests
persistent differences in responsiveness despite a large overall decline.

Call this **habituation-like attenuation**, not demonstrated habituation:
trial order is confounded with elapsed time, there is no unstimulated treatment
colony, and commanded duty is not necessarily delivered vibration dose. Logged
gyro RMS varies little despite varying duty. The midpoint timing controls are
not experimental sham stimuli. Trial-level intervals are descriptive within
these recordings, not independent-colony replication.

[Figure: repeated-pulse attenuation](</home/sam-reiter/bucket/ReiterU/Ants/basler/20260624/block02/analysis_outputs/collective_task_probe_20260908/06_response_adaptation.png>)

### A result I would not promote: coordinated work shifts

Pooling the post-stimulation period initially suggests excess synchrony.
Excluding the shared morning rise makes this weak or inconsistent across
colonies and detrending windows. Late-morning collective variance is close to
independently time-shifted controls. There is some slower overnight covariation
on the left, but no robust general evidence here for turn-taking, endogenous
collective bursts, or ants activating one another.

[Figure: synchrony sensitivity checks](</home/sam-reiter/bucket/ReiterU/Ants/basler/20260624/block02/analysis_outputs/collective_task_probe_20260908/05_residual_synchrony.png>)

## Where the scientific opportunity lies

Whole-colony tracking, persistent spatial organization, experience-dependent
specialization, and daily activity patterns already have substantial precedent.
For example, [Mersch et al. (2013)](https://doi.org/10.1126/science.1234316)
tracked six colonies over 41 days and linked social organization to spatial
fidelity. High temporal resolution is an advantage, but not by itself a novelty
claim. I would prioritize the following mechanistic questions.

### A. Does a stable daily workload hide changing workers?

Extend the previous continuous trip analysis to **ant × clock time × day**.
Keep trip rate, duration, outside-time fraction, and resource occupancy separate.
Estimate both each ant's share of colony investment and the colony total.
Decompose daily changes into previously contributing ants doing more, new
contributors entering, and previous contributors stopping. These are observable
participation changes, not automatically causal social recruitment.

The key distinction is stable specialization versus rotating participation
despite stable colony output. The present morning event suggests amplification
of existing contributors; repeated days could reveal whether that is a durable
rule or whether the amplified ants change. Reserve-worker recruitment has
already been demonstrated after worker removal in
[Charbonneau et al. (2017)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0184074).
The opportunity here is to quantify the continuous, unmanipulated balance
between replacement and increased investment, at the timescale of individual
trips and over successive days.

Clean figure: one ant-by-day investment heatmap, paired with colony total and
the contributions of continuing, entering, and leaving participants. Define
participation thresholds explicitly and test sensitivity; do not force clusters.

### B. Does an ant's recent history predict tomorrow's allocation?

Test whether resource-reaching trips, unrewarded-looking roaming, return
intervals, or contacts predict subsequent exit probability and next-day trip
investment beyond identity, clock time, spatial opportunity, and observation
coverage. A resource visit is not proof of consumption or reward.

Experience influencing specialization is established experimentally by
[Ravary et al. (2007)](https://doi.org/10.1016/j.cub.2007.06.047).
A potentially distinctive contribution would be identifying the history and
timescale that predict natural reallocations, prospectively on held-out days,
while separating experience from simply being near resources or the nest exit.
Observational prediction would motivate, not replace, causal manipulation.

Clean figure: within-ant next-trip/next-day investment change against recent
history, alongside held-out prediction improvement over identity-and-clock
baselines. Use full-day holdouts, not randomly split frames.

### C. Are task propensity and responsiveness different dimensions?

Connect stimulation response to validated trips: are strong responders habitual
foragers, or do normally nest-associated ants become mobile without visiting
resources? Does attenuation change who participates, or mostly reduce the
response of the same individuals? Does recent activity or time since return
predict responsiveness better than a fixed behavioral type?

The combination of repeated frame-aligned perturbations and continuous
individual trip histories makes this block especially useful for this question.
The initial response-persistence result is encouraging, but responsiveness has
not yet been connected to actual tasks. Proper physical-dose calibration,
unstimulated controls, recovery/retest trials, and additional colonies would
make a stronger experiment.

For day/night analyses, separate clock time from measured light exposure and
repeat the same phase over multiple days. Light shifts can substantially alter
colony foraging dynamics; see
[Pamplona-Barbosa et al. (2025)](https://www.nature.com/articles/s42003-025-08117-5).
One morning transition cannot establish a circadian rhythm.

## Reproducibility and next requirements

The removable script is [collective_task_probe.py](collective_task_probe.py).
It leaves the canonical grid-occupancy workflow untouched and uses block-specific
follow-up windows. Run from the repository root:

```bash
python analysis/exploratory/collective_task_probe.py --dataset /home/sam-reiter/bucket/ReiterU/Ants/basler/20260624/block02
```

Seven PNG figures, per-ant/per-trial CSVs, JSON statistics, and intermediate
sampling caches are saved in the dataset's
`analysis_outputs/collective_task_probe_20260908/` directory. Speeds are reduced
to five-second bins, requiring at least 25% valid frames; key comparisons require
70% usable bins per epoch or 80% per pulse window. Position summaries sample one
frame per second. Morning confidence intervals resample ants; stimulation
intervals resample or cluster trials. Neither supplies additional biological
replicates, and exploratory comparisons have not been multiplicity-corrected.

Before the across-day study: confirm species, lighting and disturbance logs;
validate calibration and this block's colony/resource polygons; identify longer
linked recordings and verify identities across blocks. Retain interrupted and
boundary-crossing trips as censored observations rather than silently dropping
them, and normalize rates by observable exposure. The existing May 15 multi-day
recording is a possible pilot, but it is a different experiment, not extra days
from this June 24 block.
