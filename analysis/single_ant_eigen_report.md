# Single-ant eigenposture analysis

This report reuses cells 1–6 and 8–9 of `eigenposture_interactive.py`: body alignment, PCA, feature/model comparisons, behavioral states, ant classes, full-recording speed, S0 occupancy, calibrated motion sleep, and individual time-of-day profiles. The full recording contributes usable observations; position coverage must exceed 60% for the state discovery cohort. The fixed model also predicts every locally valid full-body window from other ants with a confirmed resident identity. Empty and excluded tracks remain in `inventory_all.csv`; discovery selection and prediction eligibility are recorded separately.

Ten-landmark alignment and per-ant length normalization are unchanged. The absolute geometry QC limits are calibrated to the larger ants in this cohort; see `geometry.json` and `analysis_protocol.json`. Positions and speeds retain their original units. Posture interpolation is limited to one second within the same recording and camera. Motion sleep uses raw landmark motion.

Single ants form one cohort. Unmeasured social-contact features are NaN and excluded from every model representation. Colony-specific nest occupancy, trips, and return contacts are omitted. Cup maps and normalized within-cup occupancy are post hoc plots and cannot affect state or class learning.

S0 has the same putative-sleep interpretation as in the colony report. It is a learned state, not independently validated physiological sleep. Model stability, coverage and calibration plots should be inspected before interpreting classes or sleep differences.

If no state meets the unchanged colony descriptive quiet limits (body median
≤0.05 mm/s and antenna median ≤0.15 mm/s), the report keeps S0 membership and
the calibrated-motion computations as relative readouts. Titles and captions
explicitly withdraw the sleep interpretation. `state_interpretation.json`
records the failed criterion; no state or threshold is changed to make it pass.

Known unrecorded gaps are explicitly unknown in both S0 readouts. The trailing
motion window restarts at each source block, requiring its full one-second
span within that recording. This recording-support mask does not change
postures, PCA, state labels or motion thresholds.

The PDF bundles all applicable figures. PNGs and numerical tables are beside it. Eigenposture reconstructions and cup occupancy supplement the shared colony figures. Use `../run_report.sh` to rebuild the report from the published packet on Deigo.

`analysis/sleep_s01_exclusion` contains the separate S0/S1-versus-other export,
with one `labels.npz` per cup and the `s01_every_ant.pdf` report. Category 1 is an
observed S0/S1 assignment, 0 is another observed state, and -1 is ambiguous.
This export uses the newly fitted dictionary, preserves unrecorded gaps and
resident-ID exclusions, and adds no motion-inferred labels. Every cup appears in
the summary, including unassigned and identity-excluded cups. Its numerical
summary also refreshes the displayed speed measurements from the new tracks.
