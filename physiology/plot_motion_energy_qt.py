#!/usr/bin/env python3
"""Interactive Qt/Matplotlib viewer for motion-energy .me HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_ME_PATH = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/physiology/20260707_ant_on_ball/"
    "cam1_2026-07-07-14-55-22.avi.me"
)
DEFAULT_DATASET = "motion_energy"


def _clean_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_motion_energy_path(path: str | Path = DEFAULT_ME_PATH) -> Path:
    """Resolve a .me file, accepting either the file itself or its directory."""
    path = Path(path).expanduser()
    if path.is_dir():
        matches = sorted(path.glob("*.me"))
        if not matches:
            raise FileNotFoundError(f"No .me files found in {path}")
        if len(matches) > 1:
            examples = "\n  ".join(str(match) for match in matches[:10])
            raise ValueError(
                f"Multiple .me files found in {path}. Pass the file explicitly:\n  {examples}"
            )
        return matches[0]

    if path.exists():
        return path

    if path.suffix != ".me":
        me_path = path.with_name(path.name + ".me")
        if me_path.exists():
            return me_path

    return path


def resolve_cli_inputs(me_path: Path, dataset: str) -> tuple[Path, str]:
    """Handle the common mistake of passing a file/folder as --dataset."""
    dataset_path = Path(dataset).expanduser()
    if me_path == DEFAULT_ME_PATH and dataset != DEFAULT_DATASET and dataset_path.exists():
        print(
            f"[INFO] treating --dataset value as the .me path/folder: {dataset_path}. "
            f"Using dataset {DEFAULT_DATASET!r}.",
            flush=True,
        )
        return resolve_motion_energy_path(dataset_path), DEFAULT_DATASET
    return resolve_motion_energy_path(me_path), dataset


def load_motion_energy(
    me_path: str | Path = DEFAULT_ME_PATH,
    *,
    dataset: str = DEFAULT_DATASET,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the motion-energy vector and HDF5 attributes from a .me file."""
    me_path = resolve_motion_energy_path(me_path)
    with h5py.File(me_path, "r") as h5:
        if dataset not in h5:
            raise KeyError(f"{me_path} has no dataset named {dataset!r}; found {list(h5.keys())}")
        motion_energy = h5[dataset][:].astype(np.float32, copy=False)
        attrs = {key: _clean_attr(value) for key, value in h5.attrs.items()}
    attrs["path"] = str(me_path)
    return motion_energy, attrs


def _downsample_mean(values: np.ndarray, time_seconds: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= int(max_points):
        return time_seconds, values

    bin_size = int(np.ceil(len(values) / int(max_points)))
    n_bins = int(np.ceil(len(values) / bin_size))
    pad = n_bins * bin_size - len(values)
    if pad:
        values_padded = np.pad(values.astype(np.float64, copy=False), (0, pad), constant_values=np.nan)
        time_padded = np.pad(time_seconds.astype(np.float64, copy=False), (0, pad), constant_values=np.nan)
    else:
        values_padded = values.astype(np.float64, copy=False)
        time_padded = time_seconds.astype(np.float64, copy=False)

    values_ds = np.nanmean(values_padded.reshape(n_bins, bin_size), axis=1).astype(np.float32)
    time_ds = np.nanmean(time_padded.reshape(n_bins, bin_size), axis=1)
    return time_ds, values_ds


def _moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    window_samples = int(window_samples)
    if window_samples <= 1 or len(values) == 0:
        return values
    window_samples = min(window_samples, len(values))
    kernel = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def plot_motion_energy(
    me_path: str | Path = DEFAULT_ME_PATH,
    *,
    dataset: str = DEFAULT_DATASET,
    max_overview_points: int = 50_000,
    detail_seconds: float = 120.0,
    start_seconds: float = 0.0,
    show: bool = True,
):
    """Open a Qt/Matplotlib interactive plot for a motion-energy .me file.

    The upper axis shows a downsampled overview. Drag across it to update the
    lower axis with the exact-resolution samples for that selected time window.
    """
    import matplotlib

    matplotlib.use("QtAgg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider, SpanSelector

    motion_energy, attrs = load_motion_energy(me_path, dataset=dataset)
    fps = float(attrs.get("fps", np.nan))
    if not np.isfinite(fps) or fps <= 0:
        fps = 1.0

    n_samples = int(len(motion_energy))
    time_seconds = np.arange(n_samples, dtype=np.float64) / fps
    duration_seconds = float(time_seconds[-1]) if n_samples else 0.0
    overview_t, overview_y = _downsample_mean(motion_energy, time_seconds, max_overview_points)

    fig, (ax_overview, ax_detail) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [1, 1.4]},
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.18, hspace=0.35)

    source = Path(str(attrs.get("source_video", attrs.get("video", attrs["path"])))).name
    fig.suptitle(
        f"Motion energy: {Path(attrs['path']).name}\n"
        f"{n_samples:,} samples, {fps:g} fps, {duration_seconds / 3600:.2f} h"
    )

    ax_overview.plot(overview_t / 60.0, overview_y, lw=0.7, color="#1f77b4")
    ax_overview.set_ylabel("Motion energy")
    ax_overview.set_xlabel("Time (min)")
    ax_overview.set_title(source, fontsize=10)
    ax_overview.grid(True, alpha=0.25)

    detail_line, = ax_detail.plot([], [], lw=0.8, color="#222222")
    ax_detail.set_ylabel("Motion energy")
    ax_detail.set_xlabel("Time (min)")
    ax_detail.grid(True, alpha=0.25)

    status = fig.text(0.08, 0.115, "", ha="left", va="center", fontsize=9)
    smooth_ax = fig.add_axes([0.17, 0.055, 0.50, 0.035])
    smooth_slider = Slider(
        smooth_ax,
        "smooth (s)",
        0.0,
        10.0,
        valinit=0.0,
        valstep=0.1,
    )
    reset_ax = fig.add_axes([0.75, 0.052, 0.12, 0.042])
    reset_button = Button(reset_ax, "reset")

    current_window = {"start": float(start_seconds), "stop": float(start_seconds + detail_seconds)}

    def update_detail(start: float, stop: float) -> None:
        start = max(0.0, float(start))
        stop = min(duration_seconds, float(stop))
        if stop <= start:
            stop = min(duration_seconds, start + 1.0)

        current_window["start"] = start
        current_window["stop"] = stop
        first = max(0, int(np.floor(start * fps)))
        last = min(n_samples, int(np.ceil(stop * fps)) + 1)
        x = time_seconds[first:last] / 60.0
        y = motion_energy[first:last]
        window_samples = int(round(float(smooth_slider.val) * fps))
        if window_samples > 1:
            y = _moving_average(y, window_samples)

        detail_line.set_data(x, y)
        ax_detail.set_xlim(start / 60.0, stop / 60.0)
        finite = np.isfinite(y)
        if finite.any():
            ymin = float(np.nanmin(y[finite]))
            ymax = float(np.nanmax(y[finite]))
            if ymin == ymax:
                ymax = ymin + 1.0
            pad = 0.05 * (ymax - ymin)
            ax_detail.set_ylim(ymin - pad, ymax + pad)
        ax_detail.set_title(
            f"Detail: {start / 60:.2f}-{stop / 60:.2f} min "
            f"(frames {first:,}-{max(first, last - 1):,})"
        )
        status.set_text(
            f"Drag across the overview to choose a window. "
            f"Selected duration {(stop - start):.1f} s; smoothing {float(smooth_slider.val):.1f} s."
        )
        fig.canvas.draw_idle()

    def on_select(x_min: float, x_max: float) -> None:
        update_detail(min(x_min, x_max) * 60.0, max(x_min, x_max) * 60.0)

    def on_smooth(_value: float) -> None:
        update_detail(current_window["start"], current_window["stop"])

    def on_reset(_event) -> None:
        update_detail(float(start_seconds), float(start_seconds + detail_seconds))
        ax_overview.set_xlim(float(time_seconds[0]) / 60.0 if n_samples else 0.0, duration_seconds / 60.0)
        smooth_slider.reset()
        fig.canvas.draw_idle()

    selector = SpanSelector(
        ax_overview,
        on_select,
        "horizontal",
        useblit=True,
        props={"facecolor": "#ff7f0e", "alpha": 0.25},
        interactive=True,
        drag_from_anywhere=True,
    )
    smooth_slider.on_changed(on_smooth)
    reset_button.on_clicked(on_reset)

    # Keep widget objects alive through the figure.
    fig._motion_energy_widgets = (selector, smooth_slider, reset_button)  # type: ignore[attr-defined]
    update_detail(float(start_seconds), float(start_seconds + detail_seconds))

    if show:
        plt.show()
    return fig, (ax_overview, ax_detail)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("me_path", nargs="?", type=Path, default=DEFAULT_ME_PATH)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HDF5 dataset name, not the file path.")
    parser.add_argument("--max-overview-points", type=int, default=50_000)
    parser.add_argument("--detail-seconds", type=float, default=120.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    me_path, dataset = resolve_cli_inputs(args.me_path, args.dataset)
    plot_motion_energy(
        me_path,
        dataset=dataset,
        max_overview_points=args.max_overview_points,
        detail_seconds=args.detail_seconds,
        start_seconds=args.start_seconds,
        show=True,
    )


if __name__ == "__main__":
    main()
