# ArUco-referenced evaluation harness for the SLEAP centroid models

**Design: W1.1 — PDA-C / TagProbe** (Paired Disagreement Adjudication, cluster-level,
with a bit-margin ArUco probe). Converged 2026-08-05 via a 21-agent structured debate
(5 candidate designs → FOR/AGAINST advocacy per design → scoring → refinement →
3 adversarial rebuttals → adjudication). Rejected alternatives and the decisive reason
for each are recorded in §9 — that record is part of the deliverable, not an appendix.

---

## 1. What is being evaluated

| | A0 (incumbent) | centroid_all6 (candidate) |
|---|---|---|
| path | `/bucket/.../Simple_skeleton_nn/250408_141245.centroid/` | `/bucket/.../20260803_topdown/models/centroid_all6/` |
| P | 0.9770 | 0.9973 |
| R | 0.9825 | 0.9672 |
| F1 | 0.9797 | 0.9820 |
| mean loc err | 2.67 px | 2.90 px |
| FP / FN | 53 / 40 | 6 / 75 |

Measured on the 100 held-out human-reviewed cam10 frames
(`block01_20260515_round1_chunk06_..._REVIEWED_test.pkg.slp`, 100 frames /
2287 user instances, images embedded).

F1 is a tie. The models differ **only** in where they sit on the precision/recall
trade-off, so the harness exists to separate those two axes at scale.

**Provenance, stated plainly.** `centroid_all6`'s `training_config.yaml` sets both
`pretrained_backbone_weights` and `pretrained_head_weights` to A0's `best.ckpt`. It
early-stopped at epoch 40 with `best.ckpt` from roughly epoch 1, under settings later
found broken (`gaussian_noise_p 1.0`, `lr 3e-5`, rotation ±15°). It is a warm-start
perturbation of A0. It must not be described as "improved by six datasets", and a
negative result here says nothing about a correctly trained six-dataset model.

**Both models anchor on the tag.** `head_configs.centroid.confmaps.anchor_part: 'aruco '`
in *both* configs, `sigma 2.5`, `output_stride 2`, `preprocessing.scale 0.5`. The
centroid a model predicts *is* the ArUco tag position on the ant's back — not a body
centre. This makes centroid↔tag comparison a direct point-to-point measurement and is
why the localisation arm (M9) is nearly free.

---

## 2. The core idea

For a frame `f` and operating point `π`, with `A(f)` the unknown set of real ants:

```
ΔFN ≡ FN_all6 − FN_A0 = TP_A0 − TP_all6      (N_f cancels — the ant count is not needed)
ΔFP ≡ FP_A0 − FP_all6
Cobs ≡ |D_A0| − |D_all6| = ΔFN + ΔFP          (exactly observed, no reference at all)
```

Take the union `D_A0 ∪ D_all6` and its single-linkage connected components at
`R_link = 25 px`. Call each a **cluster**. Pair the two models' detections inside a
cluster by optimal assignment at `R_pair = 8 px`. If every detection pairs, the two
models present the same point set and the cluster contributes **exactly zero** to both
ΔFN and ΔFP — *whatever* is actually in it. The queen, untagged workers, unread tags
and shared hallucinations all drop out with coefficient zero.

Only **disagreement clusters** (≥1 model-exclusive detection) carry signal. A blinded
human answers one question per sampled disagreement cluster — *how many distinct ants
are here, and which marked point sits on which ant* — which splits the exactly-observed
`Cobs` into ΔFP and ΔFN with **no reference of any kind in the estimator's bias**.

That is the whole design. Everything else is variance reduction, scale and honesty.

ArUco enters in exactly three non-decision roles:
1. a **stratification variable** that sets sampling *rate* (affects variance, never bias);
2. an absolute **tag-recall probe** (M5/M6), labelled CORROBORATIVE, read by no gate;
3. the **localisation** arm (M9), where the radial tag-to-anchor offset is identical for
   both models on the same tag in the same frame and cancels exactly.

ArUco never adjudicates anything.

---

## 3. Empirical corrections applied to the debated spec

The debate ran against a briefing containing three load-bearing premises. All three were
checked against the data on 2026-08-05; two were wrong. The corrections are folded into
the implementation and are why it departs from the spec text in places.

### 3.1 `errorCorrectionRate=0` is *not* a free purity win — it deletes real tags

The brief asserted that ECR=1.0 manufactures ghosts and that ECR=0 cuts them −85% "with
no real markers lost". Measured on 20260515 with the actual dictionary:

* The shipped `_aruco_tracks.h5` **exactly reproduces ECR=1.0** detections (7 of 8 probe
  frames identical ID sets), confirming the files were written at the default — and
  confirming frame-index alignment between `.avi` and `.h5`.
* ECR ∈ {0.0, 0.2, 0.4, 0.6} are **identical**; the step is between 0.6 and 1.0. The
  dictionary has `max_correction_bits = 1`, so ECR ≤ 0.6 permits zero corrections.
* Over 120 strided cam10 frames, ECR=1.0 → 0 removes **142 roster detections**
  (IDs with ≥5% occupancy) and **1** rare detection. IDs killed by ECR=0 include ID 82
  (present in 77% of frames, 0% teleports, 0.14 px median step) — unambiguously a real,
  well-behaved tag.
* Cross-block total drop: 3.8% (block01 cam03), 7.0% (block01 cam22), 4.5% (block02
  cam10), 0.0% (block02 cam17), 6.7% (block03 cam05).
* Genuine phantoms exist but are numerically tiny and have a different signature —
  ID 21: 1.03% occupancy, 9.23% teleport rate, 0.90 px median step.

**Why:** `min_distance = 4 > 2 × max_correction_bits = 2`, so a 1-bit correction decodes
*uniquely and provably correctly*. ECR=0's entire cost is deleting real tags with one
misread bit — i.e. exactly the blurred, tilted, occluded ants where the two models
differ. An ECR=0 pre-filter would have removed the hard population from the reference.

**Consequence:** we run **both** detectors on every burst frame and tier by measured
**bit margin** (`EXACT` = decoded at ECR 0; `CORR` = decoded only at ECR 1.0). Purity
becomes a reported axis (the M6 ladder) instead of a hidden filter. Phantoms are removed
by the `ROSTER` + `MOBILE` + `GEOM` tier bits, which target the phantom signature
directly.

### 3.2 The training footage no longer exists — the strata mean something different

`chunk_manifest.csv` sources every training chunk from videos timestamped
`2026-05-15 13-09-56 … 13-10-00`. Those are block00's `*_000.avi` entries, and **all 25
are dangling symlinks** pointing at
`/bucket/.../block01/cam01_cam0_2026-05-15-13-09-56.avi` etc., which do not exist
(block01's actual videos are timestamped `13-52-54`). The labelled frames survive only as
embedded images inside the `.pkg.slp` packages.

**Therefore no frame of block01/block02/block03 was in training.** The briefing's
"block01, 19 cameras it trained on → in-domain ceiling, near-memorisation" is not
correct. The revised strata:

| id | contents | banner | role |
|---|---|---|---|
| **S1** | block01, the 19 non-holdout cameras | `SAME-CAMERA / DIFFERENT UNSEEN FOOTAGE — camera-identity overlap only, NOT memorisation` | optimism gap `Δ(S1) − Δ(S3)` |
| **S2** | block01, the 6 holdout cameras `cam10, cam15, cam21, cam22, cam23, cam25` | `UNDERPOWERED, n=6` | bridge to the 100-reviewed-frame cam10 result; six individual camera values, never a smoothed CI |
| **S3** | block02 (25) + block03 (25) = 50 camera-runs | `n_blocks = 2, n_colonies = 1, n_recording_days = 1` | **the decision stratum** |
| **S4** | calibration: 100 reviewed cam10 frames, 200 new labelled block02/03 frames, queen frames, audits | — | never merged |

`eval_report.py` raises if asked to aggregate across strata. There is no "overall" column.

This correction *strengthens* the study — the feared in-domain contamination is absent —
but S1 is still optimistic (same camera, same colony, 43 minutes later) and keeps its
banner.

### 3.3 Feasibility items the spec flagged as risks, now measured

| spec concern | measured result |
|---|---|
| 0c seek exactness (ABORT gate) | **PASSES.** Seek-then-read is byte-identical to sequential decode on 4/4 probe frames (`np.array_equal`, maxdiff 0), including inside the 105 GB block02 file. |
| 0d `--centroid_only` "does not exist" | **Wrong — it exists** in sleap-nn 0.3.1's CLI (`sleap-nn predict --centroid_only`, plus `--centroid-output {instance,centroid,both}`). The largest stated schedule risk is removed. `--frames` also exists (comma list or hyphen range), so prediction can target strided frames directly. |
| block02/03 scale | block02 cam10 is **5,754,108 frames** (66 h, 24 fps, 105 GB); block03 cam10 is 3,255,395 — not 37 k. Uniform stride therefore spans several day/night cycles, which is what we want; whole-video decode is impossible and seek-sampling is mandatory. Seek costs ~0.6 s/frame. |
| ArUco reference density | Highly variable. block02 cam10 gives ~19.9 tags/frame; **block02 cam17 gives 0.1** (4 detections in 40 frames). A minimum-reference gate per camera is required, and low-reference cameras must be listed in `coverage.json` rather than silently contributing near-empty denominators. |

---

## 4. Sampling

Per camera, **6 interpenetrating systematic replicates** of 100 anchors:

```
S   = span // 100                                # span = F − 2·burst_half
o_r = SHA1(f"{block}/{cam}/{r}") mod S           # r = 0..5
anchors_r = [lo + o_r + k·S for k in 0..99]
```

600 anchors per camera × 75 camera-runs = **45,000 anchor frames**; ×2 models =
90,000 model-frames. Each anchor drags a 7-frame burst `{f−3 … f+3}` for ArUco tiering
= 4,200 ArUco-decoded frames per camera.

Why interpenetrating: the spread across the 6 replicate estimates is a **design-based
variance that includes sampling phase** — the only estimator that answers "would a
one-frame stride shift change the verdict?", and it costs nothing extra.

Why uniform stride over the whole recording: ants cluster asleep at night and disperse by
day, so any contiguous window samples one behavioural regime.

Day/night covariate comes from the diagnostic sidecar's wall clock, **never**
`frame_idx / F` — this project has documented frame-drop events (19–26 frames lost on
cam08–17; a 13-minute dead tail costing ~24 k frames/camera) precisely because index does
not track time. Cameras without a sidecar are flagged `NO-CLOCK` and excluded from the
day/night slice only.

---

## 5. Reference construction (model-independent)

Two detectors per burst frame — `DET_EXACT` (ECR 0.0) and `DET_CORR` (ECR 1.0), identical
otherwise, both with the two newly exposed conservative parameters
(`max_erroneous_bits_in_border_rate = 0.35`, `min_otsu_std_dev = 5.0`). Bit margin by
6 px set difference. Storage is **ragged**, never `(F,100,2)` — the existing dense writers
are last-write-wins and silently drop two ants carrying the same ID in one frame.

Tiers are a **bitfield**, not a ladder, so the purity/coverage trade is an explicit
reported axis: `EXACT`, `CORR`, `ROSTER`, `INTERIOR`, `PERSIST3`, `PERSIST5`, `GEOM`,
`MOBILE`. Named reporting tiers `T_ALL / T_EXACT / T_STRICT / T_ULTRA`, plus
`T_TRANSIENT` for interpolated single-frame dropouts.

`MOBILE` exists because a detached tag, a tag on a dead ant, and a static decode artefact
all pass `PERSIST5` with a perfect quad and zero displacement — the tighter the
persistence tier, the *more* enriched in them it becomes. A sleeping ant is still
`MOBILE` (she moved earlier in the recording); a detached tag never is.

Two-colony split is detected by **same-frame co-occurrence** of one ID at >200 px in ≥1%
of burst frames — direct evidence a single ant cannot produce. (A 2-means split on x
fires on ordinary nest↔forage shuttling and would halve a real ant into two sub-roster
entities, erasing it from the reference.)

**No model output enters reference construction at any point.**

---

## 6. Operating points

Never a shared nominal threshold — A0 is a converted legacy model, all6 is native
sleap-nn trained under `gaussian_noise_p 1.0`; their score scales are not comparable.

1. **π_native** — reproduces each model's published cam10 detection count
   (23.04 / 22.22 per frame). A deployment descriptor. **Not gated.**
2. **π_match** — both models emit the same detections/camera-run, at the geometric mean
   of the native counts. **The gated capability point.** Here `Cobs = 0` by construction,
   so `ΔFN = −ΔFP` and a single statistic `Δ ≡ ΔFP(π_match)` says which model spends an
   equal detection budget better.
3. **Frontier** — {0.7, 0.85, 1.15, 1.3} × the π_match count, estimated by re-weighting
   the *same* verdicts (a detection's ant/no-ant status does not depend on the threshold),
   so the frontier costs zero extra human time.

**Why π_match is gated and π_native is not.** Under the warm-start hypothesis all6 *is* A0
at a stricter threshold; then `ΔFN(π_native) > 0` is a mathematical consequence of the
threshold, not a finding, and any gate on it is predetermined. `Δ(π_match) = 0` under the
same hypothesis, so an equivalence test on Δ gets *easier* to pass as the hypothesis
becomes truer — the correct direction for a test of it.

---

## 7. Metrics

**PRIMARY (the only gated pair).**
`M1 — Δ` at π_match, Horvitz–Thompson over adjudicated clusters with recorded inclusion
probabilities; cancelling clusters contribute zero and are not in the sum.
Design-unbiased regardless of every ArUco property, because every term comes from a human
verdict.
`M2 — (ΔFP, ΔFN)` at π_native, reported with `Cobs` and the identity check
`ΔFN + ΔFP = Cobs`. Deployment descriptor, not gated.

**SECONDARY — descriptive, no gate reads them.** `M3` Cobs · `M4` queen rate
(episode-clustered) · `M5` TagRecall by tier/radius/density/radial/day-night ·
`M6` bit-margin stability of ΔTagRecall (`T_ULTRA → … → CORR-only`; a delta growing
toward CORR-only localises all6's deficit to the blurred/occluded population) ·
`M7` provenance (per-layer weight distance, score-quantile map) · `M8` frontier ·
`M9` paired localisation (recovers the 2.67 vs 2.90 px axis at 45,000-frame scale for
zero extra compute) · `M10` calibration on S4 · `M11` adjudication quality (κ, can't-tell
rate, blinding-guess rate, exhaustive-vs-HT check) · `M12` coverage diagnostics.

---

## 8. Uncertainty and the decision rule

Every primary figure carries the **largest** of three variances: camera-clustered
bootstrap (10 k reps); interpenetrating-replicate variance (the only one including
phase); and two-way crossed camera × ant cluster-robust variance (detectability is
overwhelmingly an ant-level property and tagged ants migrate across the array, so cameras
are not independent clusters). Design effects are reported; `DEFF > 4` on the ant
dimension goes in the headline, because it means a handful of individuals drive the
result.

`can't tell` verdicts produce a **bracket**, and gates read the least favourable endpoint.
Sensitivity sweeps over `R_pair`, `R`, tier and roster cut are reported as envelopes whose
least favourable endpoint the gate reads — not as abstention filters, because
abstain-on-sign-flip retains exactly the runs whose estimates sat far from zero and drives
conditional coverage below nominal.

**Bias budget, stated before the gates** (primary, S3, ants/frame): human error on
ambiguous clusters ≤0.015 · can't-tell bracket ≤0.010 · blinding leakage ≤0.005 · HT
weight mis-specification ≤0.005 · **total ≈0.035**. The corroborative ArUco terms sum to
the same order as the effect they would measure — which is exactly why no gate reads them.

**Decision rule**, on S3 only, at the least favourable endpoint, with
`δ_cap = 0.05` detections/frame:

- **G0 prerequisites** — Stage 0 gates passed, κ ≥ 0.80, can't-tell ≤ 10%/stratum,
  blinding-guess ≤ 55%, exhaustive-vs-HT agreement on one audited camera. Any failure ⇒
  `INCONCLUSIVE`, naming the blocker.
- **G-Q queen** (terminal) — `queen_rate(all6) < queen_rate(A0) − 0.05` with an
  episode-clustered CI excluding 0 ⇒ **KEEP A0**, overriding everything.
- **G-C capability** at π_match — CI entirely above `+δ_cap` ⇒ **SHIP centroid_all6**;
  entirely below `−δ_cap` ⇒ **KEEP A0**; entirely inside `±δ_cap` ⇒ **EQUIVALENT**, go to
  G-O; otherwise **INDETERMINATE**, ship nothing.
- **G-O operating point** (on EQUIVALENT) — **KEEP A0 AND RETUNE**, naming the detection
  count at which A0 reproduces all6's native point. A config change on the incumbent
  strictly dominates shipping a checkpoint with known-broken training settings.
- **G-R retrain floor** — absolute recall < 0.95 for *both* models on the 200 labelled
  block02/03 frames ⇒ **NEITHER — RETRAIN**.
- **G-N** — if no monotone knob exists (Stage 0g), terminal `NATIVE-ONLY`: report
  (ΔFP, ΔFN) at π_native, state that capability and operating point cannot be separated,
  make no ship recommendation.
- **Default** — **KEEP A0** with the frontier published. Incumbency wins ties by
  pre-registration.

No FP/FN exchange rate is asserted anywhere; the gated statistic at matched budget makes
one unnecessary, and the stored frontier lets a later tracker-level A/B re-decide with no
new GPU work.

---

## 9. Rejected alternatives and why

Recorded permanently. Each was generated as a sharply committed design, given a
strongest-case FOR advocate and a mechanical AGAINST adversary, then scored against
C1 separation · C2 ArUco-FN robustness · C3 ArUco-FP robustness · C4 decision value ·
C5 honesty/generalisation · C6 feasibility · C7 statistical soundness · C8 ground-truth
anchoring.

### D1 — ArUco-as-Ground-Truth, maximally corrected (44/80)
*Accept the naive rule (unmatched tag = FN, unmatched centroid = FP) but rebuild the
reference: ECR=0, burst tag-carry, persistence-promoted untagged ants, threshold sweep.*

**Decisive reason:** its layers L2 and L4 make the reference **a function of the two
models being compared** — L2 deletes tags using model consensus, L4 *creates* untagged-ant
reference points from the pooled output of A0 and all6. That turns the paired difference
from bias-cancelling into bias-amplifying along the exact axis where the models differ. A
static confuser (debris, a stain, a shed tag, a dead ant) most easily satisfies L4's
≥4-of-5-frame persistence, and each one promoted becomes a TP for A0 and an FN for all6 —
roughly +0.012 of spurious ΔFN, enough alone to flip the verdict. The 3 px motion gate is
no defence (sub-pixel jitter accumulates past it) and it rejects sleeping ants, i.e. the
night-cluster stratum the uniform stride deliberately samples. The queen — the case L4
exists to solve — is provably not rescued: L4 drops every pooled detection within 30 px of
a tag, and she is surrounded by a tagged retinue at contact distance.

**Salvaged:** optimal LSAP assignment replacing the greedy matcher; deterministic
uniform-stride sampling with a written anchor manifest; the S1–S4 discipline; the cam10
wiring check before anything is believed.

### D2 — NREM, noisy-reference error model with algebraic deconvolution (41/80)
*Estimate each camera's ArUco FP rate φ and recall ρ, then invert observed match counts
into model precision and recall.*

**Decisive reason:** the absolute-precision arm needs ρ pinned to ≈±0.01 and the estimator
cannot deliver it. κ is an **ant-level** property but is drawn from a frame-level
posterior over ~2,291 labels that are ~23 distinct ants observed 100 times each —
pseudo-replication giving SE ≈ 0.063, not the ≈0.006 assumed. With
`∂ΔFP/∂ρ ≈ −31…−57` that propagates to ±1.9–2.9 FP/frame against signals of 0.53 and
0.06: the reference-model error bar is 3.6× the larger quantity being measured. Its own
fallback (the paired arm) multiplies the explain-away term by a model-dependent quantity,
so a per-model differential in P(detect | untagged ant) does not cancel and is sign-locked
in favour of the conservative candidate.

**Salvaged:** the single strongest idea in the round — measure recall **strictly
conditional on the reference point being a real ant**, so the queen and untagged workers
are in neither numerator nor denominator. That became the corroborative TagRecall arm.

### D3 — GOLDTAG, persistence-filtered reference with symmetric temporal triage (46/80)
*Keep only "gold" tags surviving ECR=0 + persistence + motion + quad geometry + non-edge,
measure recall on those, measure precision via split-peak and ephemeral-unmatched channels.*

**Decisive reason:** the two axes are measured on **different populations** and the
mismatch is sign-locked toward the conservative candidate. The gold filters select
precisely the slow, unoccluded, flat-lying, interior ants — the near-complement of the
marginal ants a conservative model sheds first — attenuating ΔR_gold by roughly an order
of magnitude so it passes the pre-registered gate. Meanwhile both precision channels leak
the other way: its `r_link = 12 px` is *smaller* than its own `v_max` clamp, so real ants
moving 13–20 px/frame are triple-qualified as ephemeral false positives in proportion to
each model's recall. Its out-of-arena channel is unbuildable — `get_arena_seg.py` is an
interactive Tk tool hardcoded to a single-ant path, so no batch masks exist.

**Salvaged:** model-independent temporal persistence tiering with per-camera `v_max`
estimated from data (clamp widened to [8, 60] px — D3's [8, 25] was calibrated circularly
on the already-filtered slow population); SHA1-derived per-camera phase, generalised into
the six interpenetrating replicates; symmetric edge exclusion.

### D4 — RP3, recall probe / precision by proxy (57/80 — round-1 top scorer)
*ArUco strictly one-sided as a recall instrument; precision reconstructed from iso-recall
over-detection, duplicate splits, arena geometry and the reviewed anchor set.*

**Decisive reason (closest to the winner, still rejected):** its central claim — that
ArUco is a purely one-sided recall probe — is **false in both directions**. An untagged
ant's detection (queen, untagged worker, blur-failed read) has no competing tag, so the
assignment spends it satisfying a *neighbouring* tag whose real ant was missed; the filler
rate rises with detection density, so A0 literally buys TagRecall with its own
over-detection and the recall axis is contaminated by the precision axis. Symmetrically,
its duplicate-rate metric charges an untagged ant inside `r_dup` as a split on that tag,
in proportion to the model's recall — ranking the higher-recall model as the worse
over-splitter. Its iso-recall estimator also has a hard bug: `argmin` over the τ grid
silently returns `s = 0` when a recall target exceeds a model's reachable recall, and at
`peak_threshold 0.0` the `s = 0` set is every confmap local maximum, so it inverts sign at
exactly the high targets it cares about.

**Salvaged — the largest contributor to the winner:** the code-enforced rule that no
unmatched-centroid count may appear in any numerator or denominator (`eval_report.py`
*raises* rather than footnotes); the code-enforced no-pooling report template; camera-run
as cluster unit with ratio-of-sums rather than mean-of-ratios; refusal to fabricate CIs
where n is too small; and the iso-recall insight — **enforce** the cancellation
precondition instead of asserting it — reformulated as π_match.

### D5 — Certified two-sided bounds, partial identification (39/80)
*Never emit a point estimate; derive certified intervals under audited assumptions about
ghosts φ and tagless ants T, then a tighter certified interval on the paired difference.*

**Decisive reason:** the cancellation is **inverted**. The terms that cancel (jointly
matched tags, jointly detected tagless slots) are the ones whose status is essentially
certain; the terms that survive (model-exclusive matched tags and model-exclusive
unmatched detections) are 100% ambiguous by construction — a model-exclusive unmatched
detection is *by definition* an object ArUco did not reference. Reconstructing from the
cam10 rates gives a certified ΔTP width of ~0.6–1.25 TP/frame against a true effect of
0.35, and it does not shrink with frames, cameras or blocks because every term is
per-frame. Independently, the ghost allowance `ceil(φ·|T_f|)` with φ ≤ 0.0099 and
|T_f| ≈ 18 evaluates to exactly 1 tag per frame — 2.9× the entire per-frame recall
difference — so the bound contains zero deterministically at every setting.

**Salvaged:** byte-identical lossless clips with a checksum assertion (strengthened to a
*decoded* round-trip comparison on a random 5% of clips, since the paired argument rests
on both models seeing identical pixels); the radius envelope, with the abstention rule
replaced by "the gate reads the least favourable endpoint".

### Also rejected: reusing `build_training_inventory.py`'s matcher
Its `compute_sleap_match_metrics` does `d2.argmin(axis=1)` then a radius test, with no
exclusivity: `n_matched = in_radius.sum()` while `n_unmatched_aruco` de-duplicates via
`set()`. So `TP + FN ≠ |A|`, recall can exceed 1, and the inflation scales with detection
density — flattering whichever model emits more peaks, which is the very axis under study.
`group_labelling/test_eval_match.py` reproduces this: on a three-detections-one-tag pile
the greedy matcher reports **recall 1.50** where `match_lsap` reports 0.50. Only the two
constants (`DEFAULT_MATCH_RADIUS = 30.0`, `DEFAULT_EDGE_MARGIN = 50.0`) carry over.

---

## 10. Known limitations (read before quoting any number)

1. **Colony/day generalisation is unbounded and unestimable.** All three blocks are
   20260515: one colony, one tag set, one lighting rig, one day. S3 has `n_blocks = 2`,
   which admits no cluster bootstrap. "Never seen" means *never-seen-cameras-within-one-
   recording-day*.
2. **The ant is the inferential unit and there is one draw of ants.** Detectability is
   overwhelmingly ant-level, and tagged ants migrate across the array, so the 75
   camera-runs observe overlapping ant sets. No CI here covers "what would happen with a
   different set of ants".
3. **Adjudication scores location and count, not correspondence.** Inside a genuine pile a
   human cannot always resolve which detection belongs to which of two touching ants;
   those become can't-tell brackets — and that is where the pipeline actually struggles.
4. **The absolute-precision validation is coarser than the bias it validates.** 200
   labelled frames give SE ≈ 0.057 ants/frame against a 0.035 envelope. Reported as a
   coarse smoke test with its resolution printed; never described as validating the
   budget. Closing it needs ~1,500 labelled frames — a different project.
5. **Ants missed by both models are invisible, by construction and irreparably.** They
   cancel identically and no instrument here sees them. Absolute recall claims must not be
   made from this harness.
6. **The queen estimate is conditioned on her visibility**, and visibility correlates with
   detectability. The exclusion rate is reported; the induced selection bias is not
   corrected.
7. **δ_cap = 0.05 is a measurement-resolution argument, not a downstream-cost one.** The
   harness has not determined the trade is worth it; the stored frontier is the mitigation.
8. **No downstream measurement at all.** The harness stops at stage 1. It cannot say
   whether A0's extra detections produce ghost tracks or are harmlessly pruned, nor whether
   all6's extra misses are recovered by track interpolation.
9. **One checkpoint per model, no seed variance.** Given the F1 tie and the warm-start
   provenance, `EQUIVALENT` / `INDETERMINATE` is a realistic and expected outcome.
10. **The capability comparison depends on a usable monotone knob** (Stage 0g). If none
    exists, the harness terminates in `NATIVE-ONLY` and makes no ship recommendation.
11. **The corroborative arm is attenuated by an unknown factor.** Tags producing no
    candidate quad at all are absent at every rung and their prevalence is unmeasurable.
    The arm supports ordering and texture only — never a quantitative correction.
12. **Low-reference cameras.** block02 cam17 yields ~0.1 tags/frame. Such cameras are
    gated out of the corroborative arm and listed in `coverage.json`; they still contribute
    to the primary (adjudication) arm, which needs no reference.

---

## 11. Scripts

| script | role |
|---|---|
| `eval_common.py` | shared core: discovery, strata, sampling, detectors, `match_lsap`, clustering |
| `test_eval_match.py` | matcher/sampler invariants incl. the greedy-pathology regression |
| `eval_preflight.py` | Stage 0 — every gate measured, not assumed → `stage0_report.json` |
| `eval_sample_anchors.py` | deterministic anchor + burst manifests → `anchors/`, `strata.json` |
| `eval_extract_and_aruco.py` | Stage A — clip extraction + two-pass ArUco → ragged `_ref.h5` |
| `eval_build_reference.py` | tier bitfield, roster, two-colony split, `v_max`, `MOBILE` |
| `eval_predict_centroids.py` | Stage B — both models on the *same* clip in one task |
| `eval_cluster_and_stratify.py` | clusters, cancellation, disagreement set X, HT sampling, adjudication packages |
| `eval_estimate.py` | HT estimators, three variances, brackets, frontier, TagRecall |
| `eval_report.py` | report with the honesty rules enforced in code |
| `submit_eval.sh` | SLURM DAG with dedup + logging conventions |

Human work required and not automatable: cluster adjudication, the 200 labelled frames,
queen marking. Estimated ~27 reviewer-hours, needing two people (the 15%
double-adjudication and inter-reviewer κ require it).

Environment: `/apps/unit/ReiterU/sleap-nn/0.3.1/tools/sleap-nn/bin/python`;
`--partition=largegpu --gres=gpu:a100:1 --cpus-per-task=16 --mem=128G
--exclude=saion-gpu25`; `/bucket` is read-only on compute nodes, so all outputs go to
`/work/ReiterU/...` and are copied to `/bucket` from the login node.
