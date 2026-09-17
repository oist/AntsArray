"""Keep early and late interactive figures in the configured output folder."""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.figure_saving import install_auto_savefig


def test_all_figure_sections_use_configured_output(tmp_path, monkeypatch):
    # Execute only the script's saver setup, without loading the ant dataset.
    script = Path(__file__).with_name("grid_occupancy.py")
    calls = sorted(
        (
            node
            for node in ast.walk(ast.parse(script.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "install_auto_savefig"
        ),
        key=lambda node: node.lineno,
    )
    assert len(calls) == 2
    output_root = tmp_path / "chosen_figure_output"
    cache_root = tmp_path / "cache" / "foraging_summary"
    namespace = {
        "install_auto_savefig": install_auto_savefig,
        "FIGURE_ROOT": output_root,
        "FORAGING_SUMMARY_ROOT": cache_root,
        "FIGURE_DPI": 50,
        "SAVE_FIGURES": True,
    }
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "_antsarray_auto_savefig_state", None, raising=False)
    figures = []
    try:
        for call, label in zip(calls, ("early", "late")):
            eval(compile(ast.Expression(call), str(script), "eval"), namespace)
            fig, ax = plt.subplots()
            figures.append(fig)
            ax.set_title(label)
            ax.plot([0, 1], [0, 1])
            plt.show()

        saved = list(output_root.glob("*.png"))
        assert len(saved) == 2
        assert any(path.name.startswith("grid_occupancy_") for path in saved)
        assert any(path.name.startswith("foraging_summary_") for path in saved)
        assert all(path.stat().st_size > 0 for path in saved)
        assert not cache_root.exists()
    finally:
        for fig in figures:
            plt.close(fig)


def test_posture_tables_create_output_separately_from_reused_cache(tmp_path):
    script = Path(__file__).with_name("grid_occupancy.py")
    calls = sorted(
        (
            node
            for node in ast.walk(ast.parse(script.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and (node.func.value.id, node.func.attr) in {
                ("SLEEP_POSTURE_ROOT", "mkdir"),
                ("sleep_posture_points", "to_parquet"),
            }
        ),
        key=lambda node: node.lineno,
    )
    output = tmp_path / "new_arena_grid" / "sleep_motion_analysis" / "posture"
    points = pd.DataFrame({"x_mm": [0.0], "y_mm": [1.0]})
    namespace = {"SLEEP_POSTURE_ROOT": output, "sleep_posture_points": points}
    for call in calls:
        eval(compile(ast.Expression(call), str(script), "eval"), namespace)
    pd.testing.assert_frame_equal(pd.read_parquet(output / "aligned_posture_points.parquet"), points)
