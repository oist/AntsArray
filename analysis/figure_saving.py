"""Small helper for saving interactive matplotlib figures from notebook scripts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any


def _slugify(text: str, *, max_len: int = 80) -> str:
    text = re.sub(r"\s+", "_", str(text).strip().lower())
    text = re.sub(r"[^a-z0-9_.-]+", "", text)
    text = text.strip("._-")
    return text[:max_len].strip("._-")


def _figure_label(fig: Any) -> str:
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        text = str(suptitle.get_text()).strip()
        if text:
            return text

    for ax in fig.axes:
        title = str(ax.get_title()).strip()
        if title:
            return title

    manager = getattr(getattr(fig, "canvas", None), "manager", None)
    if manager is not None and hasattr(manager, "get_window_title"):
        title = str(manager.get_window_title()).strip()
        if title:
            return title

    return "figure"


def install_auto_savefig(
    output_dir: str | Path,
    *,
    prefix: str,
    dpi: int = 180,
    enabled: bool = True,
) -> None:
    """Patch ``plt.show`` so new interactive figures are saved before display.

    This is intentionally script-level: notebook-style analysis files can enable
    it once after their editable settings are defined, and existing plotting
    functions can keep using ``plt.show()`` normally.
    """
    if not enabled:
        return

    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    state = getattr(plt, "_antsarray_auto_savefig_state", None)
    if state is None:
        state = {
            "original_show": plt.show,
            "saved_figure_ids": set(),
            "counter": 0,
            "output_dir": output_path,
            "prefix": str(prefix),
            "run_stamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "dpi": int(dpi),
        }

        def show_with_save(*args: Any, **kwargs: Any) -> Any:
            save_new_figures(plt)
            return state["original_show"](*args, **kwargs)

        plt.show = show_with_save
        plt._antsarray_auto_savefig_state = state
    else:
        state["output_dir"] = output_path
        state["prefix"] = str(prefix)
        state["dpi"] = int(dpi)
        state["counter"] = 0
        state["run_stamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Figure auto-save enabled: {output_path}", flush=True)


def save_new_figures(plt_module: Any | None = None) -> list[Path]:
    """Save open matplotlib figures that have not already been saved."""
    if plt_module is None:
        import matplotlib.pyplot as plt_module

    state = getattr(plt_module, "_antsarray_auto_savefig_state", None)
    if state is None:
        return []

    saved_paths: list[Path] = []
    saved_figure_ids: set[int] = state["saved_figure_ids"]
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for fig_num in plt_module.get_fignums():
        fig = plt_module.figure(fig_num)
        fig_id = id(fig)
        if fig_id in saved_figure_ids:
            continue

        state["counter"] += 1
        label = _slugify(_figure_label(fig))
        prefix = _slugify(state["prefix"], max_len=48) or "figure"
        filename = f"{prefix}_{state['run_stamp']}_{state['counter']:03d}"
        if label:
            filename = f"{filename}_{label}"
        out_path = output_dir / f"{filename}.png"

        fig.savefig(out_path, dpi=int(state["dpi"]), bbox_inches="tight")
        saved_figure_ids.add(fig_id)
        saved_paths.append(out_path)
        print(f"Saved figure: {out_path}", flush=True)

    return saved_paths
