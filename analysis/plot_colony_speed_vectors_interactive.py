#!/usr/bin/env python3
"""Simple VS Code/Jupyter plot for colony-average speed vectors.

Open this file in VS Code, run the cells from top to bottom, and edit the
settings in the first cell when needed.
"""

from __future__ import annotations

# %%
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
# Settings to edit before running the notebook cells.
SPEED_ROOT = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/speed_vectors"
)
MIN_PRESENT_FRAC = 0.40
BIN_SECONDS = 60.0
SMOOTH_SECONDS = 10 * 60.0


# %%
LIGHT_OFF_CLOCK_SECONDS = 18 * 3600
LIGHT_ON_CLOCK_SECONDS = 6 * 3600


def parse_track_start_seconds(track_name: str) -> int:
    match = re.search(r"_all_(\d{6})_", track_name)
    if match is None:
        match = re.search(r"_(\d{6})_(?:left|right)\.parquet$", track_name)
    if match is None:
        raise ValueError(f"Could not parse HHMMSS start time from {track_name!r}")

    stamp = match.group(1)
    return int(stamp[:2]) * 3600 + int(stamp[2:4]) * 60 + int(stamp[4:6])


def format_clock_time(seconds: int) -> str:
    seconds = int(seconds) % (24 * 3600)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"


def load_tracks(speed_root: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in sorted((speed_root / "per_track").glob("*/speed_metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)

        track_name = str(meta.get("track_name", metadata_path.parent.name))
        if track_name.endswith("_left.parquet") or metadata_path.parent.name.endswith("_left"):
            side = "left"
        elif track_name.endswith("_right.parquet") or metadata_path.parent.name.endswith("_right"):
            side = "right"
        else:
            continue

        n_frames = int(meta["n_frames"])
        n_observed = int(meta["n_observed_frames"])
        rows.append(
            {
                "track_name": track_name,
                "track_id": meta.get("track_id"),
                "side": side,
                "frame_min": int(meta["frame_min"]),
                "n_frames": n_frames,
                "n_observed_frames": n_observed,
                "present_frac": n_observed / n_frames,
                "fps": float(meta["fps"]),
                "speed_path": metadata_path.parent / "speed_mm_s.npy",
            }
        )

    tracks = pd.DataFrame(rows)
    if tracks.empty:
        raise FileNotFoundError(f"No speed metadata found under {speed_root / 'per_track'}")
    return tracks.sort_values(["side", "track_id", "track_name"]).reset_index(drop=True)


def select_tracks(track_table: pd.DataFrame) -> pd.DataFrame:
    selected = track_table[track_table["present_frac"] > MIN_PRESENT_FRAC].copy()
    if selected.empty:
        raise ValueError(f"No tracks passed MIN_PRESENT_FRAC={MIN_PRESENT_FRAC}")
    return selected.reset_index(drop=True)


def accumulate_side_speed(tracks: pd.DataFrame, side: str) -> tuple[np.ndarray, np.ndarray, float]:
    side_tracks = tracks[tracks["side"] == side]
    if side_tracks.empty:
        return np.zeros(0), np.zeros(0, dtype=np.uint16), np.nan

    fps = float(side_tracks["fps"].iloc[0])
    n_global_frames = int((side_tracks["frame_min"] + side_tracks["n_frames"]).max())
    sum_speed = np.zeros(n_global_frames, dtype=np.float64)
    count_speed = np.zeros(n_global_frames, dtype=np.uint16)

    for row in side_tracks.itertuples(index=False):
        print(f"{side}: {row.track_name}")
        speed = np.load(row.speed_path, mmap_mode="r")
        start = int(row.frame_min)
        stop = start + len(speed)
        valid = np.isfinite(speed)

        sum_slice = sum_speed[start:stop]
        count_slice = count_speed[start:stop]
        sum_slice[valid] += speed[valid]
        count_slice[valid] += 1

    return sum_speed, count_speed, fps


def bin_speed(sum_speed: np.ndarray, count_speed: np.ndarray, fps: float) -> pd.DataFrame:
    if len(sum_speed) == 0:
        return pd.DataFrame(columns=["time_h", "mean_speed_mm_s", "n_speed_samples"])

    bin_frames = max(1, int(round(fps * BIN_SECONDS)))
    n_bins = int(np.ceil(len(sum_speed) / bin_frames))
    pad = n_bins * bin_frames - len(sum_speed)

    padded_sum = np.pad(sum_speed, (0, pad), constant_values=0)
    padded_count = np.pad(count_speed.astype(np.uint32), (0, pad), constant_values=0)
    bin_sum = padded_sum.reshape(n_bins, bin_frames).sum(axis=1)
    bin_count = padded_count.reshape(n_bins, bin_frames).sum(axis=1)

    mean_speed = np.full(n_bins, np.nan)
    keep = bin_count > 0
    mean_speed[keep] = bin_sum[keep] / bin_count[keep]

    return pd.DataFrame(
        {
            "time_h": np.arange(n_bins) * bin_frames / fps / 3600,
            "mean_speed_mm_s": mean_speed,
            "n_speed_samples": bin_count,
        }
    )
#test 8

def add_clock_columns(timeseries: pd.DataFrame, start_clock_seconds: int) -> pd.DataFrame:
    out = timeseries.copy()
    clock_seconds = (
        start_clock_seconds + np.rint(out["time_h"].to_numpy() * 3600).astype(np.int64)
    ) % (24 * 3600)
    out.insert(1, "clock_time", [format_clock_time(value) for value in clock_seconds])
    return out


def add_light_lines(ax: plt.Axes, start_clock_seconds: int, max_time_h: float) -> None:
    events = [
        (LIGHT_OFF_CLOCK_SECONDS, "18:00 lights off", "0.20", "--"),
        (LIGHT_ON_CLOCK_SECONDS, "06:00 lights on", "0.65", "-."),
    ]
    for clock_seconds, label, color, linestyle in events:
        first_h = (clock_seconds - start_clock_seconds) / 3600
        while first_h < 0:
            first_h += 24
        for i, time_h in enumerate(np.arange(first_h, max_time_h + 1e-9, 24)):
            ax.axvline(
                time_h,
                color=color,
                linestyle=linestyle,
                lw=1,
                alpha=0.75,
                label=label if i == 0 else None,
            )


def smooth_timeseries(timeseries: pd.DataFrame) -> pd.DataFrame:
    out = timeseries.copy()
    window_bins = max(1, int(round(SMOOTH_SECONDS / BIN_SECONDS)))
    for side in ("left", "right"):
        raw_col = f"{side}_mean_speed_mm_s"
        smooth_col = f"{side}_smoothed_speed_mm_s"
        out[smooth_col] = (
            out[raw_col]
            .rolling(window=window_bins, center=True, min_periods=max(1, window_bins // 4))
            .mean()
        )
    return out


def plot_colony_speed(timeseries: pd.DataFrame, start_clock_seconds: int) -> pd.DataFrame:
    plot_df = smooth_timeseries(timeseries)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"left": "tab:blue", "right": "tab:orange"}
    for side in ("left", "right"):
        ax.plot(
            plot_df["time_h"],
            plot_df[f"{side}_smoothed_speed_mm_s"],
            color=colors[side],
            lw=1.8,
            label=f"{side} colony",
        )

    max_time_h = float(plot_df["time_h"].max())
    add_light_lines(ax, start_clock_seconds, max_time_h)
    ax.set_xlabel(f"Elapsed time from {format_clock_time(start_clock_seconds)} (h)")
    ax.set_ylabel("Mean speed (mm/s)")
    ax.set_title(f"Colony-average ant speed, {SMOOTH_SECONDS / 60:g} min smoothing")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()

    return plot_df


# %%
# Load metadata and keep tracks that are present for enough of the recording.
track_table = load_tracks(SPEED_ROOT)
experiment_start_clock_seconds = parse_track_start_seconds(track_table["track_name"].iloc[0])
tracks = select_tracks(track_table)

print(f"Loaded {len(track_table)} tracks from {SPEED_ROOT}")
print(f"Selected {len(tracks)} tracks with present_frac > {MIN_PRESENT_FRAC}")
print(f"Experiment start clock: {format_clock_time(experiment_start_clock_seconds)}")
print(tracks.groupby("side")["track_name"].count().rename("n_tracks"))

tracks.head()


# %%
# Compute colony-average speed for the left and right colonies.
left_sum_speed, left_count_speed, left_fps = accumulate_side_speed(tracks, "left")
right_sum_speed, right_count_speed, right_fps = accumulate_side_speed(tracks, "right")

left_binned = bin_speed(left_sum_speed, left_count_speed, left_fps).rename(
    columns={
        "mean_speed_mm_s": "left_mean_speed_mm_s",
        "n_speed_samples": "left_n_speed_samples",
    }
)
right_binned = bin_speed(right_sum_speed, right_count_speed, right_fps).rename(
    columns={
        "mean_speed_mm_s": "right_mean_speed_mm_s",
        "n_speed_samples": "right_n_speed_samples",
    }
)

colony_speed_timeseries = pd.merge(left_binned, right_binned, on="time_h", how="outer")
colony_speed_timeseries = colony_speed_timeseries.sort_values("time_h").reset_index(drop=True)
colony_speed_timeseries = add_clock_columns(colony_speed_timeseries, experiment_start_clock_seconds)

colony_speed_timeseries.head()


# %%
# Plot the smoothed colony-average speed.
smoothed_colony_speed_timeseries = plot_colony_speed(
    colony_speed_timeseries,
    experiment_start_clock_seconds,
)

smoothed_colony_speed_timeseries.head()
