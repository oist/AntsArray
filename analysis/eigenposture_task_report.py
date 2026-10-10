"""Run the inspectable task-state script headlessly; export static PNGs, PDF and tables.

python -m analysis.eigenposture_task_report --output /path/to/results
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def run_analysis(output):
    source_path = Path(__file__).with_name("eigenposture_interactive.py")
    source = (
        "\n".join(
            line
            for line in source_path.read_text().splitlines()
            if not line.lstrip().startswith("%matplotlib")
        )
        + "\n"
    )
    namespace = {"__name__": "__main__", "__file__": str(source_path)}
    plt.close("all")
    exec(compile(source, str(source_path), "exec"), namespace)
    n = namespace
    assert n["task_input"].shape[1] == 7
    assert len(n["assignments"]) == len(n["ants"])
    np.testing.assert_array_equal(n["tasks"] >= 0, n["valid"])
    np.testing.assert_array_equal(
        n["tasks"][n["valid"]],
        n["state_remap"][n["task_model"].predict(n["task_input"])],
    )
    np.testing.assert_allclose(n["proportions"][n["ants"].eligible].sum(axis=1), 1)
    output.mkdir(parents=True, exist_ok=True)
    (output / "interactive_task_states.py").write_text(source)
    for key, filename in [
        ("task_k_table", "task_k_selection.csv"),
        ("task_summary", "task_summary.csv"),
        ("proportion_table", "ant_task_proportions.csv"),
        ("transition_table", "ant_transition_counts.csv"),
        ("comparison_table", "ant_method_comparison.csv"),
        ("umap_table", "umap_sample.csv"),
    ]:
        n[key].to_csv(output / filename)
    n["bin_table"].to_csv(output / "five_minute_tasks.csv.gz", index=False)
    n["assignments"].to_csv(output / "ant_assignments.csv", index=False)
    pd.DataFrame(n["spatial_results"]).to_csv(
        output / "spatial_comparison.csv", index=False
    )
    for method, tables in [("transitions", n["ant_k_tables"]), ("proportions", n["baseline_k_tables"])]:
        for side, table in tables.items():
            table.to_csv(output / f"{side}_{method}_k_selection.csv")
    np.savez_compressed(
        output / "task_states.npz",
        ants=n["ants"].ant.to_numpy(str),
        **{
            key: n[key]
            for key in (
                "binned",
                "clip_counts",
                "tasks",
                "proportions",
                "ant_groups",
                "baseline_groups",
                "stay_probabilities",
                "pairs",
                "task_means",
                "feature_center",
                "feature_scale",
            )
        },
        umap_sample=n["umap_sample"],
        umap_coordinates=n["umap_coordinates"],
        posture_center=n["coordinate_center"],
        posture_modes=n["coordinate_modes"],
    )
    summary = dict(
        n_identities=len(n["ants"]),
        feature_names=n["names"],
        transition_scales=n["transition_scales"],
        umap=dict(sample_size=len(n["umap_sample"]), n_neighbors=20, min_dist=0.05,
                  n_epochs=300, seed=n["SEED"], purpose="visualization only"),
        proportion_only_groups={side: m.n_components for side, m in n["baseline_models"].items()},
        valid_bins=int(n["valid"].sum()),
        n_task_states=n["task_k"],
        posture_variance=float(n["posture_variance"][:4].sum()),
        ant_groups={
            side: dict(
                k=m.n_components,
                n=int((n["ants"].side.eq(side) & n["ants"].eligible).sum()),
            )
            for side, m in n["ant_models"].items()
        },
        spatial=n["spatial_results"],
        bootstraps=n["BOOTSTRAPS"],
        minimum_clips=n["MIN_CLIPS"],
        minimum_ant_hours=n["MIN_ANT_HOURS"],
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    with PdfPages(output / "task_states.pdf") as pdf:
        for number in plt.get_fignums():
            figure = plt.figure(number)
            figure.savefig(figures / f"figure_{number:02d}.png", dpi=150)
            pdf.savefig(figure)
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_analysis(args.output)
