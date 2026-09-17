# May 15 block02: testing allocation across days

Analysis run 2026-09-08. Strongest leads: persistent differences in investment
coexist with flexible timing among right-colony foragers; the morning movement
peak accompanies a return-to-nest wave, not an increase in roaming.

## Data and scope

This recording contains 66.60 hours, from May 15 approximately 14:20 to May 18
08:56 on the frame-derived clock. I compared the two complete calendar days,
May 16 and May 17, and examined all three morning transitions.

I resampled raw bodypoint-0 tracking positions at 1 Hz and applied the existing
`panorama_regions.csv`, assigning unsuffixed water labels by tracking geometry.
The analysis includes 126 distinct ants with at least 40% detection over the
whole recording: 56 left and 70 right. A shorter duplicate file for right tag 8
is excluded by this criterion. Daily comparisons further require at least 70%
observation on each day, retaining 32 left and 66 right ants. Phase comparisons
require 70% coverage separately in each light/dark phase and therefore have
smaller cohorts, especially on the left.

New extraction found 12,866 retained departures, including 11,958 completed
trips. These counts differ from the previous cache because that cache covered
only 64 preselected roaming ants and used different observation criteria.
The primary outside-time measure includes observed outside time even when a
departure or return is unobserved. It is not restricted to completed trips.

All outputs are separate from the existing grid analysis:

`/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/analysis_outputs/multiday_task_allocation_20260908/`

## 1. Persistent investment, but flexible timing on the right

Across all matched ants, outside-fraction rankings correlate 0.95 on the left
and 0.91 on the right between days. However, this partly reflects the large
difference between colony-associated and roaming ants. Within the previously
identified roaming cohort, outside-fraction correlations are 0.64 left
(14 matched ants) and 0.75 right (37 ants). Trip-rate correlations in those
same cohorts are 0.88 and 0.75.

The particularly interesting comparison uses the **same 34 right-colony
foragers** with adequate observation in both phases on both days:

| Individual measure | Day-to-day Spearman correlation |
| --- | ---: |
| Fraction of observed time outside | 0.77 |
| Preference for dark versus light roaming | 0.28 |

Dark/light preference is the ratio of outside fractions within the two phases:
`(outside_dark / observed_dark) / (outside_light / observed_light)`.
Thus the unequal ten-hour dark and fourteen-hour light durations are accounted
for. A paired ant bootstrap gives a 95% interval of approximately 0.20–0.77
for the difference between these correlations. This is uncertainty across ants
within this recording, not replication across colonies or many days.

The contrast also survives changing the day boundary: comparing complete
19:30-to-19:30 dark–light cycles, right-forager timing correlation is 0.03
(33 ants at 70% phase coverage), or 0.11 with a 50% threshold (39 ants).
The configured phase boundaries are 19:30 and 05:30; the physical lighting log
has not been independently verified.

Concrete examples on calendar days: right ant 84's dark/light ratio changes
from 1.35 to 5.61, whereas ant 15 changes from 2.54 to 1.03. These are changes in
relative phase allocation, not necessarily equal-and-opposite worker exchanges.

The left does not show the same pattern: timing correlation is 0.87 among its
nine phase-eligible original foragers. The small, strongly selected left cohort
does not justify a general biological explanation for the colony difference.

Interpretation: on the right, who tends to invest more in roaming is more
stable than when that investment occurs. This is compatible with persistent
individual propensities and flexible daily timing. It is **not evidence of
socially coordinated shift swapping**.

[Figure: matched foragers' investment, timing, and resource allocation](</home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/analysis_outputs/multiday_task_allocation_20260908/07_forager_investment_timing_and_resources.png>)

## 2. A morning movement increase accompanies returns to the nest

All three mornings show a transient speed peak around 05:35–05:45, accompanied
by a decrease in the fraction of observed time outside. The broader
position-matched cohorts show the following changes, comparing 05:00–05:20
with 05:40–06:00:

| Morning | Left: outside fraction | Right: outside fraction |
| --- | ---: | ---: |
| May 16 | 33.0% → 22.5% | 39.3% → 33.4% |
| May 17 | 39.4% → 26.9% | 40.7% → 29.2% |
| May 18 | 20.0% → 17.0% | 51.6% → 31.4% |

Counting anchored boundary transitions confirms net movement into the nest:
returns exceed departures during 05:30–05:45 in all six colony-mornings.
For example, the right colony on May 17 has 73 return events versus 59 departure
events within the matched cohort. These are events, not 73 distinct ants.

Additional speed-quality filtering produces smaller cohorts; their post/pre
mean-speed ratios are 1.56, 1.39, 1.19 on the left and 1.76, 1.08, 1.10 on the
right. The speed and position panels therefore do not use identical cohorts.
Restricting outside-time calculations to the speed-qualified ants still gives
a decrease in five of six cases; the exception is left May 18, whose
speed-qualified cohort is almost entirely inside before and after the window.

Raw trajectory spot-checks of left tags 1/19 and right tags 0/12 on May 16 show
outside positions before the event and sustained inside positions afterwards.
This checks the coordinate/state transition, not the complete accuracy of tag
identity or a manually reviewed behavioral classification.

Interpretation: increased locomotion cannot be equated with increased foraging.
Morning activation and nest returns occur together. The data do not establish
that returning ants account for all of the speed increase, or that contacts
cause the collective transition. A shared lighting/environmental trigger is
plausible and needs the experimental log. This also means that the June 24
speed increase alone should not be interpreted as increased foraging.

[Figure: speed, outside occupancy, and net returns](</home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/analysis_outputs/multiday_task_allocation_20260908/08_morning_speed_and_return_wave.png>)

## 3. Resource differences are not entirely explained by roaming amount

Among the original roaming ants with adequate daily coverage, the fraction of
outside time spent in resource regions has day-to-day correlations of 0.78
left (14 ants) and 0.54 right (37 ants). Thus some ants consistently devote a
larger fraction of their outside time to these regions.

Food versus water allocation is also repeatable on the right: food's share of
resource time correlates 0.69 across days in 36 ants with at least 30 observed
resource seconds on each day. Left correlation is 0.46 in 13 ants and is much
less convincing. These patterns suggest retaining continuous resource-use
differences rather than assuming they are entirely interchangeable.

Important limits: region dwell time is not food/water intake; different region
areas affect encounter opportunity; and persistent spatial habits could explain
the apparent resource preferences. This has not demonstrated discrete food
specialists and water specialists.

## 4. The simple recent-experience prediction is weak

I tested whether reaching a resource on one trip predicts the next nest
interval or resource-reaching trip. Eligible consecutive trips require observed
returns, at least 70% observation during the intervening interval, at least 90%
inside among observed interval samples, and no missing stretch longer than
30 seconds. This yields 935 sequence pairs from 31 left ants and 3,435 from
69 right ants during the two full days.

A descriptive model controlling identity, trip duration, clock harmonics, and
day associates resource visits with a 1.45-fold longer subsequent interval on
the left; the corresponding right estimate is only 1.04-fold. Ant-clustered
intervals for the log effects are reported in `results.json`.

More importantly, training on May 16 and testing on May 17, adding the resource
visit indicator does **not reliably improve next-departure timing prediction**
beyond identity, clock, and previous-trip duration. Ant-weighted prediction gains
are approximately zero in both colonies, with bootstrap intervals spanning zero.

Prediction of the next resource-reaching trip improves only slightly on the
right: held-out log loss changes from 0.473 to 0.469 and AUC from 0.640 to 0.653.
It does not improve on the left. This is not a strong general feedback result.
The test addresses a simple last-trip feature, not longer learning histories,
contact-mediated effects, or causality.

[Figure: resource history and subsequent departure](</home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/analysis_outputs/multiday_task_allocation_20260908/05_trip_history_and_next_departure.png>)

## What to prioritize

The best candidate story is **persistent investment and resource-use tendencies
with more flexible timing**, particularly on the right, together with a
repeatable morning transition from roaming toward nest occupancy. The broad
workforce is fairly stable: the first day's top 20% retain about 56% of summed
outside fractions on the left and 42% on the right on the second day, close to
their first-day shares of 58% and 44%.

Next tests should determine whether individual timing changes persist over
additional cycles and whether contact history predicts who returns or leaves
around these transitions after controlling for shared environmental timing and
position. This recording has only two complete daily comparisons and two
colonies; it cannot establish general scheduling rules or long-term careers.
No repeated-pulse responsiveness analysis is claimed for May 15.

## Reproduction and safeguards

Run [multiday_task_allocation.py](multiday_task_allocation.py) from the repository
root with `--dataset /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02`.
It is block-specific and leaves the main grid-occupancy workflow unchanged.

Eight figures, daily/hourly tables, trip sequences, held-out predictions,
statistics, resolved regions, and the one-Hz state cache are saved in the output
directory. Trips use 30-second minimum outside runs, five-second nest anchors,
up-to-30-second missing-state bridging, and at least 50% observed trip coverage.
Short border flicker is removed. Incomplete trips remain flagged; their truncated
durations are not used as completed-trip durations.

Synthetic checks covered complete returns, censored trips, short-gap bridging,
and clock-matched observation comparisons. Coverage and day-boundary sensitivity
checks are saved in `results.json`. Resource sampling totals agree to within
about 1.2% with the earlier full-frame resource totals despite cohort differences.
Plots were visually inspected. All findings are exploratory; no multiplicity
correction or independent multi-colony replication is claimed.
