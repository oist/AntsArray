#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactively extract stimulus and camera TTL windows from SpikeGLX nidq data.

VS Code workflow:
1. Open this file in VS Code using the repo/conda Python environment.
2. Edit the constants in the "Interactive defaults" section if needed.
3. Run the file with "Python: Run Python File" or from a terminal for a Qt popup.
4. Move the threshold sliders, click Apply to recompute detections, then Save.

Avoid VS Code's Jupyter/Interactive window for the popup mode; ipykernel backend
handling can kill the kernel. If you must use a notebook, run one cell with
`%matplotlib qt` before executing this file.

The saved files keep the historical names used by downstream scripts:
    STIM_WINDOWS.npy
    CAM_TTLs.npy
"""

# %%
from __future__ import annotations

%matplotlib qt
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib


def running_in_ipykernel() -> bool:
    return "ipykernel" in sys.modules or any("ipykernel" in arg for arg in sys.argv)


def configure_matplotlib_backend() -> None:
    requested_backend = os.environ.get("STIM_WINDOWS_MPL_BACKEND")
    if requested_backend:
        matplotlib.use(requested_backend, force=True)
        return

    if running_in_ipykernel():
        return

    try:
        matplotlib.use("QtAgg", force=True)
    except Exception as exc:
        print(f"Could not enable QtAgg backend, using {matplotlib.get_backend()}: {exc}")


configure_matplotlib_backend()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.signal import butter, sosfiltfilt

try:
    import cv2
except ImportError:  # Frame counts are diagnostic only.
    cv2 = None


# %% Interactive defaults
DEFAULT_FOLDER = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/physiology/20260709_antrec/20270709_ant_cb1_g0"
)

# Keep these channel assignments as in the original script.
DEFAULT_STIM_CHANNEL = 2
DEFAULT_CAMERA_CHANNEL = 1

# Target recording defaults. Channel assignments are unchanged from the old script,
# but this recording's filtered channel-1 baseline sits above the old 1000/600 mV
# thresholds, so use the high TTL envelope.
DEFAULT_STIM_HIGH_THRESHOLD_MV = 2000.0
DEFAULT_STIM_LOW_THRESHOLD_MV = 400.0
DEFAULT_STIM_MIN_WIDTH_SAMPLES = 20
DEFAULT_STIM_LOWPASS_HZ = 50.0
DEFAULT_DROP_LAST_STIM_EDGE = True

DEFAULT_CAMERA_THRESHOLD_MV = 3000.0
DEFAULT_CAMERA_MIN_SEPARATION_SAMPLES = 20

# None means the interactive plot opens halfway through the recording.
DEFAULT_PREVIEW_START_SECONDS = None
DEFAULT_PREVIEW_SECONDS = 95.0


@dataclass(frozen=True)
class NidqData:
    bin_path: Path
    meta_path: Path
    meta: dict[str, str]
    raw: np.memmap
    sample_rate: float
    n_channels: int
    n_samples: int

    @property
    def duration_seconds(self) -> float:
        return self.n_samples / self.sample_rate


@dataclass
class DetectionResult:
    stim_edges: np.ndarray
    camera_crossings: np.ndarray
    stim_high_threshold_mv: float
    stim_low_threshold_mv: float
    stim_min_width_samples: int
    camera_threshold_mv: float
    camera_min_separation_samples: int
    saved: bool = False


def read_meta(bin_path: Path) -> tuple[Path, dict[str, str]]:
    meta_path = bin_path.with_suffix(".meta")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file for {bin_path}: {meta_path}")

    meta: dict[str, str] = {}
    for line in meta_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.lstrip("~")] = value
    return meta_path, meta


def find_nidq_bin(path: Path) -> Path:
    if path.is_file():
        if path.name.endswith(".nidq.bin"):
            return path
        raise ValueError(f"Expected a *.nidq.bin file, got {path}")

    matches = sorted(path.glob("*.nidq.bin"))
    if not matches:
        raise FileNotFoundError(f"No *.nidq.bin file found in {path}")
    if len(matches) > 1:
        names = "\n".join(str(match) for match in matches)
        raise ValueError(f"Found multiple *.nidq.bin files; pass one explicitly:\n{names}")
    return matches[0]


def open_nidq(path: Path) -> NidqData:
    bin_path = find_nidq_bin(path)
    meta_path, meta = read_meta(bin_path)

    n_channels = int(meta["nSavedChans"])
    n_samples = int(int(meta["fileSizeBytes"]) / (2 * n_channels))
    sample_rate = float(meta["niSampRate"])
    raw = np.memmap(
        bin_path,
        dtype="int16",
        mode="r",
        shape=(n_channels, n_samples),
        order="F",
    )
    return NidqData(
        bin_path=bin_path,
        meta_path=meta_path,
        meta=meta,
        raw=raw,
        sample_rate=sample_rate,
        n_channels=n_channels,
        n_samples=n_samples,
    )


def ni_channel_counts(meta: dict[str, str]) -> tuple[int, int, int, int]:
    return tuple(int(value) for value in meta["snsMnMaXaDw"].split(","))  # type: ignore[return-value]


def ni_channel_gain(saved_channel: int, meta: dict[str, str]) -> float:
    saved_mn, saved_ma, _saved_xa, _saved_dw = ni_channel_counts(meta)
    if saved_channel < saved_mn:
        return float(meta["niMNGain"])
    if saved_channel < saved_mn + saved_ma:
        return float(meta["niMAGain"])
    return 1.0


def ni_int16_to_mv(data: np.ndarray, saved_channel: int, meta: dict[str, str]) -> np.ndarray:
    int_to_volts = float(meta["niAiRangeMax"]) / 32768.0
    gain = ni_channel_gain(saved_channel, meta)
    return data.astype(np.float32, copy=False) * np.float32(int_to_volts * 1000.0 / gain)


def load_ni_channel(nidq: NidqData, saved_channel: int) -> np.ndarray:
    if saved_channel < 0 or saved_channel >= nidq.n_channels:
        raise ValueError(
            f"Channel {saved_channel} is outside saved channel range "
            f"0..{nidq.n_channels - 1}"
        )
    raw_channel = np.asarray(nidq.raw[saved_channel, :])
    return ni_int16_to_mv(raw_channel, saved_channel, nidq.meta)


def butter_lowpass(data: np.ndarray, highcut_hz: float, sample_rate: float, order: int = 4) -> np.ndarray:
    if highcut_hz <= 0:
        return data
    nyquist = sample_rate / 2.0
    if highcut_hz >= nyquist:
        return data
    sos = butter(order, highcut_hz, btype="low", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, data).astype(np.float32, copy=False)


def ttl_edges(
    signal: np.ndarray,
    hi_thresh: float = DEFAULT_STIM_HIGH_THRESHOLD_MV,
    lo_thresh: float = DEFAULT_STIM_LOW_THRESHOLD_MV,
    min_width: int = DEFAULT_STIM_MIN_WIDTH_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rising/falling sample indices using hysteresis and run cleanup."""
    if lo_thresh >= hi_thresh:
        raise ValueError("lo_thresh must be < hi_thresh")
    if min_width < 1:
        raise ValueError("min_width must be >= 1")

    state = np.zeros(signal.shape, dtype=bool)
    high = False
    for index, value in enumerate(signal):
        if not high and value >= hi_thresh:
            high = True
        elif high and value <= lo_thresh:
            high = False
        state[index] = high

    change_idx = np.flatnonzero(np.diff(state.astype(np.int8)) != 0) + 1
    run_starts = np.r_[0, change_idx]
    run_ends = np.r_[change_idx, len(state)]
    for start, end in zip(run_starts, run_ends):
        if end - start < min_width:
            state[start:end] = ~state[start]

    rising = np.flatnonzero(~state[:-1] & state[1:]) + 1
    falling = np.flatnonzero(state[:-1] & ~state[1:]) + 1
    return rising.astype(np.int64), falling.astype(np.int64), state


def rising_threshold_crossings(
    signal: np.ndarray,
    threshold: float,
    min_separation: int,
) -> np.ndarray:
    above = signal >= threshold
    crossings = np.flatnonzero(~above[:-1] & above[1:]) + 1
    if crossings.size == 0:
        return crossings.astype(np.int64)

    keep = [int(crossings[0])]
    last_crossing = int(crossings[0])
    for crossing in crossings[1:]:
        crossing = int(crossing)
        if crossing - last_crossing >= min_separation:
            keep.append(crossing)
            last_crossing = crossing
    return np.asarray(keep, dtype=np.int64)


def extract_stim_edges(
    stim_signal: np.ndarray,
    sample_rate: float,
    high_threshold_mv: float,
    low_threshold_mv: float,
    min_width_samples: int,
    lowpass_hz: float,
    drop_last_edge: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filtered = butter_lowpass(stim_signal, lowpass_hz, sample_rate)
    rising, falling, clean_state = ttl_edges(
        filtered,
        hi_thresh=high_threshold_mv,
        lo_thresh=low_threshold_mv,
        min_width=min_width_samples,
    )
    stim_edges = np.sort(np.concatenate([rising, falling]))
    if drop_last_edge and stim_edges.size:
        stim_edges = stim_edges[:-1]
    return stim_edges, filtered, clean_state


def extract_camera_ttls(
    camera_signal: np.ndarray,
    threshold_mv: float,
    min_separation_samples: int,
) -> np.ndarray:
    return rising_threshold_crossings(
        camera_signal,
        threshold=threshold_mv,
        min_separation=min_separation_samples,
    )


def preview_slice(sample_rate: float, n_samples: int, start_seconds: float | None, seconds: float) -> slice:
    if start_seconds is None:
        start_seconds = (n_samples / sample_rate) / 2.0
    start = max(0, int(round(start_seconds * sample_rate)))
    stop = min(n_samples, start + max(1, int(round(seconds * sample_rate))))
    return slice(start, stop)


def decimated_preview_indices(preview: slice, max_points: int = 100_000) -> np.ndarray:
    n_points = preview.stop - preview.start
    step = max(1, int(np.ceil(n_points / max_points)))
    return np.arange(preview.start, preview.stop, step, dtype=np.int64)


def preview_events(events: np.ndarray, preview: slice) -> np.ndarray:
    mask = (events >= preview.start) & (events < preview.stop)
    return events[mask]


def slider_limits(signal: np.ndarray, default_values: tuple[float, ...]) -> tuple[float, float]:
    low, high = np.percentile(signal, [0.1, 99.9])
    values = np.asarray(default_values, dtype=float)
    low = min(float(low), float(values.min()))
    high = max(float(high), float(values.max()))
    margin = max((high - low) * 0.08, 1.0)
    return low - margin, high + margin


def print_detection_summary(result: DetectionResult, sample_rate: float) -> None:
    print(f"Detected {len(result.stim_edges)} stimulus edge(s).")
    if len(result.stim_edges):
        print(
            "First stimulus times (s): "
            + np.array2string(result.stim_edges[:10] / sample_rate, precision=3)
        )
    print(f"Detected {len(result.camera_crossings)} camera TTL(s).")
    if len(result.camera_crossings):
        print(
            "First camera TTL times (s): "
            + np.array2string(result.camera_crossings[:10] / sample_rate, precision=3)
        )


def interactive_detection_editor(
    folder: Path,
    stim_filtered: np.ndarray,
    camera_signal: np.ndarray,
    sample_rate: float,
    preview: slice,
    initial: DetectionResult,
    drop_last_stim_edge: bool,
    allow_empty: bool,
    stim_channel: int,
    camera_channel: int,
) -> DetectionResult:
    """Show editable detection thresholds and return the final detection result."""
    result = initial
    dirty = {"value": False}

    plot_idx = decimated_preview_indices(preview)
    x = plot_idx / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.34, hspace=0.32)

    axes[0].plot(x, stim_filtered[plot_idx], color="0.35", linewidth=1)
    axes[1].plot(x, camera_signal[plot_idx], color="0.35", linewidth=1)

    stim_preview_edges = preview_events(result.stim_edges, preview)
    camera_preview_crossings = preview_events(result.camera_crossings, preview)

    stim_scatter = axes[0].scatter(
        stim_preview_edges / sample_rate,
        stim_filtered[stim_preview_edges] if stim_preview_edges.size else [],
        s=48,
        color="red",
        label="stim edges",
    )
    camera_scatter = axes[1].scatter(
        camera_preview_crossings / sample_rate,
        camera_signal[camera_preview_crossings] if camera_preview_crossings.size else [],
        s=18,
        color="red",
        label="camera TTLs",
    )

    stim_hi_line = axes[0].axhline(result.stim_high_threshold_mv, color="tab:orange", linewidth=1.5)
    stim_lo_line = axes[0].axhline(result.stim_low_threshold_mv, color="tab:blue", linewidth=1.5)
    camera_threshold_line = axes[1].axhline(result.camera_threshold_mv, color="tab:orange", linewidth=1.5)

    preview_start_seconds = preview.start / sample_rate
    preview_stop_seconds = preview.stop / sample_rate

    axes[0].set_title(f"Stimulus channel {stim_channel}, filtered")
    axes[0].set_ylabel("mV")
    axes[0].legend(loc="upper right")
    axes[1].set_title(f"Camera channel {camera_channel}")
    axes[1].set_xlabel("seconds")
    axes[1].set_ylabel("mV")
    axes[1].legend(loc="upper right")

    status_text = fig.text(0.08, 0.93, "", fontsize=10)

    stim_min, stim_max = slider_limits(
        stim_filtered[preview],
        (result.stim_low_threshold_mv, result.stim_high_threshold_mv),
    )
    cam_min, cam_max = slider_limits(camera_signal[preview], (result.camera_threshold_mv,))

    ax_stim_hi = fig.add_axes([0.12, 0.255, 0.70, 0.025])
    ax_stim_lo = fig.add_axes([0.12, 0.215, 0.70, 0.025])
    ax_stim_width = fig.add_axes([0.12, 0.175, 0.70, 0.025])
    ax_cam_thresh = fig.add_axes([0.12, 0.135, 0.70, 0.025])
    ax_cam_sep = fig.add_axes([0.12, 0.095, 0.70, 0.025])
    ax_apply = fig.add_axes([0.84, 0.178, 0.10, 0.045])
    ax_save = fig.add_axes([0.84, 0.118, 0.10, 0.045])

    stim_hi_slider = Slider(
        ax_stim_hi,
        "stim high mV",
        stim_min,
        stim_max,
        valinit=result.stim_high_threshold_mv,
    )
    stim_lo_slider = Slider(
        ax_stim_lo,
        "stim low mV",
        stim_min,
        stim_max,
        valinit=result.stim_low_threshold_mv,
    )
    stim_width_slider = Slider(
        ax_stim_width,
        "stim min width samples",
        1,
        max(50_000, result.stim_min_width_samples * 2),
        valinit=result.stim_min_width_samples,
        valstep=1,
        valfmt="%0.0f",
    )
    cam_thresh_slider = Slider(
        ax_cam_thresh,
        "camera threshold mV",
        cam_min,
        cam_max,
        valinit=result.camera_threshold_mv,
    )
    cam_sep_slider = Slider(
        ax_cam_sep,
        "camera min sep samples",
        1,
        max(2_000, result.camera_min_separation_samples * 2),
        valinit=result.camera_min_separation_samples,
        valstep=1,
        valfmt="%0.0f",
    )

    apply_button = Button(ax_apply, "Apply")
    save_button = Button(ax_save, "Save")

    def set_status(message: str) -> None:
        status_text.set_text(message)
        fig.canvas.draw_idle()

    def update_threshold_lines(_value: float | None = None) -> None:
        stim_hi_line.set_ydata([stim_hi_slider.val, stim_hi_slider.val])
        stim_lo_line.set_ydata([stim_lo_slider.val, stim_lo_slider.val])
        camera_threshold_line.set_ydata([cam_thresh_slider.val, cam_thresh_slider.val])
        dirty["value"] = True
        set_status(
            "Thresholds changed. Click Apply to recompute full-file detections; "
            f"current saved set: stim={len(result.stim_edges)}, camera={len(result.camera_crossings)}."
        )

    def current_values() -> tuple[float, float, int, float, int]:
        return (
            float(stim_hi_slider.val),
            float(stim_lo_slider.val),
            int(round(stim_width_slider.val)),
            float(cam_thresh_slider.val),
            int(round(cam_sep_slider.val)),
        )

    def apply_current(_event=None) -> bool:
        nonlocal result
        stim_hi, stim_lo, stim_min_width, camera_threshold, camera_min_sep = current_values()
        if stim_lo >= stim_hi:
            set_status("Stim low threshold must be less than stim high threshold.")
            return False

        set_status("Computing detections on the full recording...")
        fig.canvas.flush_events()

        try:
            rising, falling, _clean_state = ttl_edges(
                stim_filtered,
                hi_thresh=stim_hi,
                lo_thresh=stim_lo,
                min_width=stim_min_width,
            )
        except ValueError as exc:
            set_status(str(exc))
            return False

        stim_edges = np.sort(np.concatenate([rising, falling]))
        if drop_last_stim_edge and stim_edges.size:
            stim_edges = stim_edges[:-1]
        camera_crossings = extract_camera_ttls(
            camera_signal,
            threshold_mv=camera_threshold,
            min_separation_samples=camera_min_sep,
        )

        result = DetectionResult(
            stim_edges=stim_edges,
            camera_crossings=camera_crossings,
            stim_high_threshold_mv=stim_hi,
            stim_low_threshold_mv=stim_lo,
            stim_min_width_samples=stim_min_width,
            camera_threshold_mv=camera_threshold,
            camera_min_separation_samples=camera_min_sep,
            saved=False,
        )

        new_stim_preview = preview_events(result.stim_edges, preview)
        stim_offsets = (
            np.column_stack((new_stim_preview / sample_rate, stim_filtered[new_stim_preview]))
            if new_stim_preview.size
            else np.empty((0, 2))
        )
        stim_scatter.set_offsets(stim_offsets)

        new_camera_preview = preview_events(result.camera_crossings, preview)
        camera_offsets = (
            np.column_stack((new_camera_preview / sample_rate, camera_signal[new_camera_preview]))
            if new_camera_preview.size
            else np.empty((0, 2))
        )
        camera_scatter.set_offsets(camera_offsets)

        dirty["value"] = False
        print_detection_summary(result, sample_rate)
        set_status(
            f"Applied: stim={len(result.stim_edges)}, camera={len(result.camera_crossings)}. "
            "Click Save to write STIM_WINDOWS.npy and CAM_TTLs.npy."
        )
        return True

    def save_current(_event=None) -> None:
        nonlocal result
        if dirty["value"] and not apply_current():
            return
        if len(result.stim_edges) == 0 and not allow_empty:
            set_status("No stimulus edges detected. Adjust thresholds or rerun with --allow-empty.")
            return
        save_outputs(folder, result.stim_edges, result.camera_crossings, sample_rate)
        result.saved = True
        set_status("Saved current detections.")

    for slider in (stim_hi_slider, stim_lo_slider, stim_width_slider, cam_thresh_slider, cam_sep_slider):
        slider.on_changed(update_threshold_lines)
    apply_button.on_clicked(apply_current)
    save_button.on_clicked(save_current)
    fig._stim_window_editor_widgets = (
        stim_hi_slider,
        stim_lo_slider,
        stim_width_slider,
        cam_thresh_slider,
        cam_sep_slider,
        apply_button,
        save_button,
    )

    set_status(
        f"Current detections: stim={len(result.stim_edges)}, camera={len(result.camera_crossings)}. "
        f"Preview is {preview_start_seconds:.1f}-{preview_stop_seconds:.1f}s. "
        "Move sliders, click Apply, then Save."
    )
    plt.show()
    return result


def video_frame_summary(folder: Path) -> list[str]:
    if cv2 is None:
        return ["OpenCV is not installed; skipped AVI frame count diagnostics."]

    lines: list[str] = []
    for video in sorted(folder.glob("*.avi")):
        cap = cv2.VideoCapture(str(video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        suffix = ""
        if fps > 0:
            suffix = f", metadata duration={frame_count / fps:.3f}s at {fps:.3f} fps"
        lines.append(f"{video.name}: {frame_count} frames{suffix}")
    if not lines:
        lines.append("No AVI files found for frame count diagnostics.")
    return lines


def save_outputs(folder: Path, stim_edges: np.ndarray, camera_crossings: np.ndarray, sample_rate: float) -> None:
    stim_seconds = (stim_edges / sample_rate)[:, None]
    camera_seconds = camera_crossings / sample_rate

    stim_path = folder / "STIM_WINDOWS.npy"
    camera_path = folder / "CAM_TTLs.npy"
    np.save(stim_path, stim_seconds)
    np.save(camera_path, camera_seconds)
    print(f"Saved {len(stim_seconds)} stimulus times to {stim_path}")
    print(f"Saved {len(camera_seconds)} camera TTLs to {camera_path}")


def confirm_save() -> bool:
    try:
        answer = input("Save STIM_WINDOWS.npy and CAM_TTLs.npy? [Y/n] ").strip().lower()
    except EOFError:
        return True
    return answer not in {"n", "no"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract STIM_WINDOWS.npy and CAM_TTLs.npy from a SpikeGLX nidq.bin recording.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--stim-channel", type=int, default=DEFAULT_STIM_CHANNEL)
    parser.add_argument("--camera-channel", type=int, default=DEFAULT_CAMERA_CHANNEL)
    parser.add_argument("--stim-high-threshold-mv", type=float, default=DEFAULT_STIM_HIGH_THRESHOLD_MV)
    parser.add_argument("--stim-low-threshold-mv", type=float, default=DEFAULT_STIM_LOW_THRESHOLD_MV)
    parser.add_argument("--stim-min-width-samples", type=int, default=DEFAULT_STIM_MIN_WIDTH_SAMPLES)
    parser.add_argument("--stim-lowpass-hz", type=float, default=DEFAULT_STIM_LOWPASS_HZ)
    parser.add_argument(
        "--keep-last-stim-edge",
        action="store_true",
        help="Do not drop the final stimulus edge before saving.",
    )
    parser.add_argument("--camera-threshold-mv", type=float, default=DEFAULT_CAMERA_THRESHOLD_MV)
    parser.add_argument(
        "--camera-min-separation-samples",
        type=int,
        default=DEFAULT_CAMERA_MIN_SEPARATION_SAMPLES,
    )
    parser.add_argument(
        "--preview-start-seconds",
        type=float,
        default=DEFAULT_PREVIEW_START_SECONDS,
        help="Preview start time in seconds. Omit to open halfway through the recording.",
    )
    parser.add_argument("--preview-seconds", type=float, default=DEFAULT_PREVIEW_SECONDS)
    parser.add_argument("--no-plot", action="store_true", help="Skip diagnostic plots.")
    parser.add_argument("--yes", action="store_true", help="Save without asking for confirmation.")
    parser.add_argument("--dry-run", action="store_true", help="Run detection but do not save outputs.")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow saving an empty STIM_WINDOWS.npy if no stimulus edges are detected.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignoring extra arguments from the current Python session: " + " ".join(unknown))
    return args


def is_jupyter_kernel_json(path: Path) -> bool:
    return path.suffix == ".json" and "jupyter/runtime/kernel-" in str(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    folder = args.folder.expanduser().resolve()
    if is_jupyter_kernel_json(folder):
        print(f"Ignoring Jupyter kernel path passed as folder: {folder}")
        folder = DEFAULT_FOLDER.expanduser().resolve()

    print(f"Opening nidq recording in {folder}")
    nidq = open_nidq(folder)
    print(f"nidq file: {nidq.bin_path}")
    print(
        f"sample_rate={nidq.sample_rate:.6f} Hz, "
        f"n_channels={nidq.n_channels}, n_samples={nidq.n_samples}, "
        f"duration={nidq.duration_seconds:.3f}s"
    )
    print(f"stim channel={args.stim_channel}, camera channel={args.camera_channel}")

    print("Loading stimulus channel...")
    stim_signal = load_ni_channel(nidq, args.stim_channel)
    print("Loading camera channel...")
    camera_signal = load_ni_channel(nidq, args.camera_channel)

    stim_edges, stim_filtered, _clean_state = extract_stim_edges(
        stim_signal,
        sample_rate=nidq.sample_rate,
        high_threshold_mv=args.stim_high_threshold_mv,
        low_threshold_mv=args.stim_low_threshold_mv,
        min_width_samples=args.stim_min_width_samples,
        lowpass_hz=args.stim_lowpass_hz,
        drop_last_edge=not args.keep_last_stim_edge,
    )
    camera_crossings = extract_camera_ttls(
        camera_signal,
        threshold_mv=args.camera_threshold_mv,
        min_separation_samples=args.camera_min_separation_samples,
    )
    result = DetectionResult(
        stim_edges=stim_edges,
        camera_crossings=camera_crossings,
        stim_high_threshold_mv=args.stim_high_threshold_mv,
        stim_low_threshold_mv=args.stim_low_threshold_mv,
        stim_min_width_samples=args.stim_min_width_samples,
        camera_threshold_mv=args.camera_threshold_mv,
        camera_min_separation_samples=args.camera_min_separation_samples,
    )

    print_detection_summary(result, nidq.sample_rate)

    for line in video_frame_summary(folder):
        print(line)

    if not args.no_plot:
        preview = preview_slice(
            nidq.sample_rate,
            nidq.n_samples,
            args.preview_start_seconds,
            args.preview_seconds,
        )
        result = interactive_detection_editor(
            folder,
            stim_filtered,
            camera_signal,
            sample_rate=nidq.sample_rate,
            preview=preview,
            initial=result,
            drop_last_stim_edge=not args.keep_last_stim_edge,
            allow_empty=args.allow_empty,
            stim_channel=args.stim_channel,
            camera_channel=args.camera_channel,
        )

    if args.dry_run:
        print("Dry run requested; outputs were not saved.")
        return

    if result.saved:
        return

    if len(result.stim_edges) == 0 and not args.allow_empty:
        print(
            "No stimulus edges were detected, so outputs were not saved. "
            "Adjust thresholds or pass --allow-empty to save an empty STIM_WINDOWS.npy."
        )
        return

    if args.yes or confirm_save():
        save_outputs(folder, result.stim_edges, result.camera_crossings, nidq.sample_rate)
    else:
        print("Skipped saving outputs.")


if __name__ == "__main__":
    main()
