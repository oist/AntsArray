# Colony returns and sleeping recipients

Open `analysis/exploratory/return_sleep_analysis.py` as a VS Code/Jupyter
interactive script and run its `# %%` cells. It uses the saved cluster IDs
and `sleep_motion_labels`; running `grid_occupancy.py` is unnecessary.

Default dataset: `20260723/block02`. To generate all figures without a GUI:

```bash
MPLBACKEND=Agg /home/sam-reiter/miniforge3/envs/ants/bin/python analysis/exploratory/return_sleep_analysis.py
```

Interaction input is now `block02/interactions_skeleton_0p1mm`, rebuilt from all
finished tracks at a 0.1 mm minimum skeleton-distance threshold. All observed
body/antenna nodes and segments count; crossings have zero distance. The detector
uses the same geometry implementation as the live viewer, with no center cutoff,
antenna-only restriction, or temporal filter. Each unordered pair appears at most
once per frame, with its measured distance. The cluster run writes to flash and a
login-side watcher publishes all 56 chunk/colony files to bucket after success.
The script requires `transfer_complete.ok` and will not silently use the old
1 mm files in `block02/interactions`. Each new file has a parameter sidecar;
`run_manifest.json` records the run and input provenance. Older figure galleries
remain unchanged until the script is rerun with the new input.

Outputs are written to
`block02/analysis_outputs/return_sleep/<settings-hash>/`:

- `index.html`: figure gallery, numerical comparisons, sample counts, and methods.
- `figures/`: PNG and PDF figures for cluster sleep, example trips, return
  activity, first sleep after return, matching balance, recipient responses,
  censored waking, and sensitivity comparisons.
- `matched_triggers.csv`: exact global trigger frames and matching covariates.
- `returns.csv`: exits, returns, resource visits, and observed residence ends.
- `wake_outcomes.csv`: observed waking or censoring time and reason.
- `settings.json`: parameters and source fingerprints for the run.

Edit `SETTINGS` for the minimum outside period, the time after return during
which a partner counts as recently returned, contact gap, prior sleep period,
and matching calipers. Sleep thresholds remain those in the label cache.
Regenerate `sleep_motion_labels` to change the classifier itself.

Cluster curves equally weight ants after excluding bins with insufficient
classified data. Unknown labels are never counted as wake. The analysis uses
the current `panorama_regions.csv` to identify colony and resource positions.
The legacy colony-presence vectors have incompatible boxes for this recording.

Pair contacts continue across chunk boundaries and reciprocal detections are
merged. A continuous contact is not repeatedly counted as new. Sleeping
recipients are selected using labels strictly before contact, and no-contact
times are matched within the same ant using time, position, density, sleep
duration, and motion. Matching does not select on future outcomes. Later
contacts are included in the descriptive response curves and censor follow-up
in the separate isolated-contact waking analysis.

Interpret the results alongside matching balance, contributing-ant counts,
and censoring. Ant-bootstrap intervals require at least five ants; the
Kaplan-Meier curves stop below five events at risk. Resource-visit and
directional subsets are not split in the default plots: all excursion types
are pooled. Figure 4 now has only body motion, antenna motion and contact rate.

Return qualification still requires a stable colony-outside-colony excursion,
but zero is the first observed tracking-anchor crossing into that confirmed
residence, not the majority-inside bin start. Earlier brief touches followed by
a long outside interval do not mark the start of this residence. `returns.csv` includes
`binned_return_frame`, `last_outside_frame` and crossing uncertainty for auditing.
The activity traces retain one-second motion resolution and five-second plot bins.

Figure 5's lower row counts return episodes still observed that have not met the
sustained-sleep criterion. Sleeping, leaving, missing labels and ending follow-up
remove episodes. These are not unique ants or whole-colony awake counts.

The raw detector is undirected: it does not identify an initiator or anatomical
contact type, and `recipient_body_contact` is unknown for these files. Proximity
does not prove physical contact. `contact_audit.csv` counts isolated detections.
`contact_min_detection_frames` filters these after pair and cross-chunk merging;
the default remains 1 pending video validation. The separate analysis bout rule
merges repeated hits separated by at most 2 seconds. Long
tracking dropouts can still manufacture apparently new onsets. Use the all-pair
`tracking/gui/interaction_debug_viewer.py` for frame-by-frame review and local
distance sensitivity before changing analysis defaults.
