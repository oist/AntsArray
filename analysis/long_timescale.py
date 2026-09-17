# %%
"""Explore behavior across all recordings, interactively or from the command line.

Run this file to create an offline browser dashboard plus PNGs and tables. In an
IPython editor, run cells and call ``run`` with the settings below; returned
tables remain available for further interactive analysis. See long_timescale.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import long_timescale_utils as lt

# %%
# Editable interactive defaults. Folder dates identify recordings, not the
# actual days covered by a tracked window; calendar dates are read from caches.
DATA_FOLDER = Path("/bucket/ReiterU/Ants/basler")
if not DATA_FOLDER.exists():
    DATA_FOLDER = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler")
DATES = ("20260723", "20260724", "20260729")
SETTINGS = lt.Settings()


def default_mappings():
    if Path("/bucket/ReiterU").exists():
        return []
    return ["/bucket/ReiterU=/home/sam-reiter/bucket/ReiterU",
            "/flash/ReiterU=/home/sam-reiter/flash",
            "/home/s/samuel-reiter=/home/sam-reiter/saionHome"]


def find_bundle(blocks, mappings):
    resolve = lt.path_resolver(mappings)
    required = {str(p.resolve()) for p in blocks}
    candidates = []
    for block in blocks:
        for path in (block / "analysis_outputs").glob("*/comparison_manifest.json"):
            manifest = json.loads(path.read_text())
            sources = {str(resolve(w["source"]).resolve()) for w in manifest.get("windows", [])}
            if required.issubset(sources):
                candidates.append(path)
    return max(candidates, key=lambda p: p.stat().st_mtime_ns).parent if candidates else None


def report(tables, manifest, output):
    lines = ["# Long-timescale ant behavior", "",
             "All-recording analysis of activity, sleep and spatial presence. Identities are colony side + tag ID.", "",
             "## Recording coverage", "", "| Source | First tracked time | Last tracked time |", "|---|---|---|"]
    for w in manifest["windows"]:
        lines.append(f"| {Path(w['block']).parent.name}/{Path(w['block']).name} | {w['start']} | {w['stop']} |")
    lines += ["", "## Reading changes over time", "",
              "Calendar plots include every tracked ant, partial days, and block-boundary bins with any observed metric data. No completeness or coverage cutoff hides observations by default. Day/night plots describe actual observed hours; clock profiles average repeated cycles within ants, then weight ants equally.", "",
              f"Standardized trajectories use identical clock slots for each ant in every recording, requiring at least {manifest['settings']['min_shared_hours']:g} shared hours overall or {manifest['settings']['min_phase_hours']:g} within a light/dark phase. Ant membership is fixed across recordings for each metric and phase. Missing or insufficient observations are excluded, never filled with zero.", "",
              "Slopes below use all recording centers on an actual-days axis. They are descriptive linear summaries across sparse observations, not evidence of continuous linear change or a causal aging effect. Intervals resample ants within each colony; they do not quantify between-colony uncertainty.", "",
              "| Colony | Metric | Ants | Mean slope per actual day | 95% ant-bootstrap interval |", "|---|---|---:|---:|---:|"]
    for row in tables["slope_summary"].itertuples():
        unit = lt.METRICS[row.metric][3]
        unit = "percentage points" if unit == "%" else unit
        lines.append(f"| {row.side} | {lt.METRICS[row.metric][2]} ({unit}/day) | {row.n_ants} | {row.mean:.4g} | [{row.ci_low:.4g}, {row.ci_high:.4g}] |")
    if "trip_availability" in tables:
        lines += ["", "## Individual task allocation", "",
                  f"Eight-panel figures cover all {tables['task_bins'].ant.nunique()} tracked identities in individual_ants/. The dashboard follows one ant across all dates, with daily and light/dark profiles, colony-use versus trip-investment paths, and links to the saved figures.", "",
                  "Trip events and positions are reused from the existing grid_occupancy optional trip analysis. No new tracks are stitched. Completed-trip definitions retain the source 30-second minimum, 5-second anchors, 20% observed coverage, and source gap/flicker handling. No minimum trip-count filter is added.", "",
                  "| Block/window | Tracked identities | Existing trip analyses | Completed trips |",
                  "|---|---:|---:|---:|"]
        for row in tables["trip_availability"].itertuples():
            lines.append(f"| {'/'.join(Path(row.block).parts[-2:])} | {row.tracked_ants} | {row.analyzed_ants} | {getattr(row, 'completed_trips', 0):,} |")
        lines += ["", "Trip counts are zero only for analyzed, observed ants with no completed departures. Ants absent from a source trip analysis remain unknown for trip metrics. Each departure is counted once; observed trip seconds are allocated to their actual calendar bins. Trip rates use observed position hours and mean durations use completed-trip counts.", "",
                  "Daily task profiles use metric-specific observation weights and retain partial days. Shared-clock task changes and task_slopes.csv provide an additional check for clock-time confounding across all recordings. These descriptive proxies do not establish discrete tasks, causal aging effects, or behavior during recording gaps. Completed trips omit events censored at gaps/block boundaries."]
    bouts = tables["sleep_bouts"]
    lines += ["", "## Denominators and limitations", "",
              "- Locomotor speed and sleep are binned from existing vectors across the full recording, including partial cycles. Body and antenna motion reuse one-second contexts. Coverage is reported separately from the values.",
              "- Colony/outside presence is the existing one-second majority position state. Food/water presence uses unique exact resource frames divided by all detected position frames. Overlapping annotations of the same resource type are counted once. Resource presence does not establish feeding or drinking.",
              f"- Sleep fragmentation uses existing one-second sleep states: {len(bouts):,} runs, of which {int(bouts.complete.sum()):,} are bounded by known awake states on both ends. Duration plots omit runs touching unknown observations or recording boundaries. These complete-bout summaries are coverage-sensitive.",
              "- Whole-recording spatial maps pool blocks within each ant, normalize in-arena occupancy, then weight ants equally. Their time-of-day coverage is not standardized.",
              "- All finished tracked identities are included, even those excluded from upstream clustering. Missing metric observations remain unknown. The shared-clock comparison is an optional additional view, not a filter on the main timeline.", "",
              "## Outputs", "", "Open index.html for the interactive explorer. PNGs are the complete saved overview set; Parquet/CSV files retain source blocks, actual timestamps, and missingness. run_manifest.json records settings, source fingerprints, software versions, and the code digest."]
    (Path(output) / "report.md").write_text("\n".join(lines) + "\n")


def run(data_folder=DATA_FOLDER, dates=DATES, *, output, analysis_bundle=None,
        settings=SETTINGS, workers=4, path_maps=None):
    from analysis import long_timescale_plots as plots
    from analysis import long_timescale_tasks as tasks
    from analysis import long_timescale_task_plots as task_plots
    import matplotlib
    import pandas
    import numpy
    import plotly

    settings.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Remove completion before any work, so interrupted reruns cannot appear done.
    (output / "COMPLETE.json").unlink(missing_ok=True)
    blocks, discovery = lt.discover_blocks(data_folder, dates)
    mappings = default_mappings() if path_maps is None else path_maps
    bundle = analysis_bundle or find_bundle(blocks, mappings)
    print("Tracked windows:", *blocks, sep="\n", flush=True)
    print("Analysis bundle:", bundle, flush=True)
    windows, resolve = lt.load_windows(blocks, output, bundle=bundle, mappings=mappings, workers=workers)
    tables, manifest = lt.build_tables(windows, resolve, output, settings, workers)
    tasks.build(tables, manifest, windows, resolve, output, settings)
    print("LONG_TIMESCALE_TABLES_COMPLETE", flush=True)
    spatial, grid_sources = plots.save_figures(tables, manifest, windows, output)
    task_plots.save_figures(tables, manifest, output)
    manifest.update(discovery=discovery, analysis_bundle=str(bundle) if bundle else None,
                    grid_sources=grid_sources,
                    software=dict(python=sys.version, pandas=pandas.__version__, numpy=numpy.__version__,
                                  matplotlib=matplotlib.__version__, plotly=plotly.__version__))
    code = [Path(__file__), Path(lt.__file__), Path(plots.__file__), Path(tasks.__file__), Path(task_plots.__file__),
            Path(plots.__file__).with_name("long_timescale_dashboard.html")]
    manifest["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    manifest["figures"] = [p.name for p in sorted(output.glob("*.png"))]
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report(tables, manifest, output)
    plots.write_dashboard(tables, manifest, spatial, output)
    result = dict(recordings=len(manifest["recordings"]), windows=len(windows), figures=len(manifest["figures"]),
                  dashboard=str(output / "index.html"), settings=asdict(settings))
    (output / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    print("LONG_TIMESCALE_COMPLETE", json.dumps(result), flush=True)
    return tables, manifest


# %%
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=DATA_FOLDER)
    parser.add_argument("--dates", nargs="+", default=DATES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-bundle", type=Path, help="Optional previously combined analysis folder; otherwise discover a compatible bundle")
    parser.add_argument("--path-map", action="append", help="OLD_ROOT=NEW_ROOT for shared files mounted at different paths")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-bin-coverage", type=float, default=0.)
    parser.add_argument("--min-phase-hours", type=float, default=0.)
    parser.add_argument("--min-shared-hours", type=float, default=6.)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.headless:
        import matplotlib
        matplotlib.use("Agg")
    if args.workers < 1:
        parser.error("--workers must be positive")
    run(args.data_folder, args.dates, output=args.output, analysis_bundle=args.analysis_bundle,
        settings=lt.Settings(args.min_bin_coverage, args.min_phase_hours, args.min_shared_hours, args.bootstrap),
        workers=args.workers, path_maps=args.path_map)


if __name__ == "__main__" and "get_ipython" not in globals():
    main()

# %%
# In an interactive editor:
# tables, manifest = run(output=DATA_FOLDER / "20260724/block01/analysis_outputs/long_timescale")
# tables["slopes"].query("side == 'left' and metric == 'sleep' and included")
