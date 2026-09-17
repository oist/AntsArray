# %%
"""Compare the same workers across two windows without running grid_occupancy.

Run these cells in order. Required upstream caches are checked explicitly.
Figures support picking scatter points to identify ants; the final cell selects
an individual for its early/late clock profiles.
"""
try:
    get_ipython().run_line_magic("matplotlib", "qt")  # type: ignore[name-defined]
except Exception:
    pass

from dataclasses import replace
import importlib
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print

for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
    if (candidate / "analysis" / "block_activity_utils.py").is_file():
        sys.path.insert(0, str(candidate))
        break
else:
    raise FileNotFoundError("Run from the AntsArray repository or a subdirectory")

from analysis import block_activity_utils as ba
from analysis.figure_saving import install_auto_savefig

importlib.reload(ba)

# %%
EARLY_BLOCK = Path(os.environ.get("ANTS_EARLY_BLOCK", "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260810/block02-w000-031"))
LATE_BLOCK = Path(os.environ.get("ANTS_LATE_BLOCK", "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260810/block02-w149-197"))
# The date-level directory is not writable on this mount.
OUTPUT_ROOT = Path(os.environ.get("ANTS_COMPARISON_OUTPUT", str(EARLY_BLOCK / "analysis_outputs" / "early_late_comparison")))
MIN_BIN_COVERAGE = 0.50
MIN_MATCHED_HOURS = 6.0
N_BOOTSTRAP = 2000
RANDOM_STATE = 0
READ_WORKERS = 4
REBUILD_COMPARISON_CACHE = False
SETTINGS = ba.ComparisonSettings(MIN_BIN_COVERAGE, MIN_MATCHED_HOURS, N_BOOTSTRAP, RANDOM_STATE)
SETTINGS.validate()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
install_auto_savefig(OUTPUT_ROOT, prefix="early_late", dpi=170)

# %%
# Reuse existing motion, classifier, region and activity caches. Parameter or
# source mismatches raise, rather than silently combining incompatible windows.
early = ba.load_window(EARLY_BLOCK, OUTPUT_ROOT / "cache", label="early",
                       force=REBUILD_COMPARISON_CACHE, max_workers=READ_WORKERS)
late = ba.load_window(LATE_BLOCK, OUTPUT_ROOT / "cache", label="late",
                      force=REBUILD_COMPARISON_CACHE, max_workers=READ_WORKERS)
ba.validate_windows(early, late)
for window in (early, late):
    print(f"{window['info']['label']}: {window['info']['start_time']} to {window['info']['stop_time']}")

# %%
# Only fully recorded clock bins, with enough valid data in both windows.
# Ants and clock slots receive equal weight, not weights proportional to frames.
early_profiles = ba.fold_profiles(early["bins"], SETTINGS)
late_profiles = ba.fold_profiles(late["bins"], SETTINGS)
paired_clock_bins, ant_changes = ba.pair_profiles(early_profiles, late_profiles, SETTINGS)
id_audit = ba.identity_audit(early["inventory"], late["inventory"], ant_changes)
paired_summary = ba.summarize_changes(ant_changes, SETTINGS)
membership_counts, membership_scores = ba.cluster_overlap(id_audit)

id_audit.to_csv(OUTPUT_ROOT / "identity_audit.csv", index=False)
ant_changes.to_csv(OUTPUT_ROOT / "ant_changes.csv", index=False)
paired_summary.to_csv(OUTPUT_ROOT / "paired_summary.csv", index=False)
paired_clock_bins.to_parquet(OUTPUT_ROOT / "paired_clock_bins.parquet", index=False)
pd.concat([early_profiles.assign(window="early"), late_profiles.assign(window="late")]).to_parquet(
    OUTPUT_ROOT / "ant_clock_profiles.parquet", index=False)
membership_counts.to_csv(OUTPUT_ROOT / "cluster_membership_overlap.csv", index=False)
membership_scores.to_csv(OUTPUT_ROOT / "cluster_membership_scores.csv", index=False)
id_audit.loc[id_audit.selected_both, ba.IDENTITY + ["cluster_id_early", "cluster_id_late"]].to_csv(
    OUTPUT_ROOT / "cluster_membership_pairs.csv", index=False)
print("Identity selection (not survival or death):")
display(id_audit.groupby("side")[["selected_early", "selected_late", "selected_both"]].sum())
print("Paired estimates; bootstrap intervals describe ants within these colonies:")
display(paired_summary[["side", "metric", "n_ants", "early_mean", "late_mean", "mean_delta", "spearman_rho"]])
print("Excluded or insufficiently observed IDs remain in identity_audit.csv and ant_changes.csv")

# %%
ba.plot_coverage(id_audit, SETTINGS.min_matched_hours)
plt.show()

# %%
ba.plot_paired_activity(ant_changes, paired_summary, metrics=("speed", "body", "antenna"))
plt.show()
ba.plot_paired_activity(ant_changes, paired_summary, metrics=("sleep", "outside"))
plt.show()

# %%
ba.plot_changes_by_id(ant_changes, id_audit)
plt.show()
clock_figure, clock_summary = ba.plot_clock_changes(paired_clock_bins, ant_changes, early["info"])
clock_summary.to_csv(OUTPUT_ROOT / "paired_colony_clock_profiles.csv", index=False)
plt.show()

# %%
# Whole-window spatial membership is a secondary, time-unmatched description.
ba.plot_cluster_overlap(membership_counts, membership_scores)
plt.show()
display(membership_scores)

# %%
# Sensitivity changes coverage requirements, not the underlying classifier.
sensitivity = []
for coverage in (0.50, 0.75):
    for hours in (3.0, 6.0, 9.0):
        options = replace(SETTINGS, min_bin_coverage=coverage, min_matched_hours=hours)
        _, candidate_changes = ba.pair_profiles(ba.fold_profiles(early["bins"], options),
                                                ba.fold_profiles(late["bins"], options), options)
        sensitivity.append(ba.summarize_changes(candidate_changes, options))
coverage_sensitivity = pd.concat(sensitivity, ignore_index=True)
coverage_sensitivity.to_csv(OUTPUT_ROOT / "coverage_sensitivity.csv", index=False)
display(coverage_sensitivity[["min_bin_coverage", "min_matched_hours", "side", "metric", "n_ants", "mean_delta", "spearman_rho"]])

# %%
# Individual inspection. None chooses the largest supported body-motion change
# as a descriptive example; set a side and ID to inspect a different worker.
SELECTED_SIDE = None  # "left" or "right"
SELECTED_ANT_ID = None
if SELECTED_ANT_ID is None:
    candidates = ant_changes[ant_changes.metric.eq("body") & ant_changes.included]
    if candidates.empty:
        raise ValueError("No ants meet the paired-coverage threshold; inspect the audit or adjust parameters")
    example = candidates.loc[candidates.delta.abs().idxmax()]
    selected_side, selected_id = example.side, int(example.track_id)
else:
    if SELECTED_SIDE not in ("left", "right"):
        raise ValueError("Set SELECTED_SIDE as well as SELECTED_ANT_ID")
    selected_side, selected_id = SELECTED_SIDE, int(SELECTED_ANT_ID)
ba.plot_ant_profiles(early_profiles, late_profiles, side=selected_side, track_id=selected_id, info=early["info"])
plt.show()
display(ant_changes[ant_changes.side.eq(selected_side) & ant_changes.track_id.eq(selected_id)])

# %%
run_stamp = plt._antsarray_auto_savefig_state["run_stamp"]
figure_paths = sorted(OUTPUT_ROOT.glob(f"early_late_{run_stamp}_*.png"))
ba.write_report(OUTPUT_ROOT, early, late, SETTINGS, paired_summary, id_audit,
                ant_changes, membership_scores, coverage_sensitivity, figure_paths)
print(f"Saved {len(figure_paths)} figures, tables and comparison_report.md to {OUTPUT_ROOT}")
