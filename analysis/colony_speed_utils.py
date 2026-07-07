"""Helpers for interactive colony speed-vector plotting."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_track_start_seconds(track_name: str) -> int:
    match = re.search(r"_all_(\d{6})_", track_name)
    if match is None:
        match = re.search(r"_(\d{6})_(?:left|right)\.parquet$", track_name)
    if match is None:
        raise ValueError(f"Could not parse HHMMSS start time from {track_name!r}")

    stamp = match.group(1)
    hour = int(stamp[:2])
    minute = int(stamp[2:4])
    second = int(stamp[4:6])
    return hour * 3600 + minute * 60 + second


def format_clock_time(seconds: float) -> str:
    seconds_i = int(round(seconds)) % (24 * 3600)
    return f"{seconds_i // 3600:02d}:{(seconds_i % 3600) // 60:02d}"


def speed_metadata_paths(speed_root: Path) -> list[Path]:
    paths = sorted((speed_root / "per_track").glob("*/speed_metadata.json"))
    if not paths:
        paths = sorted(speed_root.glob("*/speed_metadata.json"))
    return paths


def side_from_track(track_name: str, track_dir: Path) -> str | None:
    if track_name.endswith("_left.parquet") or track_dir.name.endswith("_left"):
        return "left"
    if track_name.endswith("_right.parquet") or track_dir.name.endswith("_right"):
        return "right"
    return None


def load_speed_tracks(speed_root: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in speed_metadata_paths(Path(speed_root)):
        with metadata_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)

        track_name = str(meta.get("track_name", metadata_path.parent.name))
        side = side_from_track(track_name, metadata_path.parent)
        if side is None:
            continue

        n_frames = int(meta["n_frames"])
        n_observed = int(meta["n_observed_frames"])
        rows.append(
            {
                "track_name": track_name,
                "track_id": meta.get("track_id"),
                "side": side,
                "metadata_path": metadata_path,
                "speed_path": metadata_path.parent / "speed_mm_s.npy",
                "frame_min": int(meta["frame_min"]),
                "frame_max": int(meta.get("frame_max", int(meta["frame_min"]) + n_frames - 1)),
                "n_frames": n_frames,
                "n_observed_frames": n_observed,
                "present_frac": n_observed / n_frames if n_frames else np.nan,
                "fps": float(meta["fps"]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(f"No speed_metadata.json files found under {speed_root}")
    return out.sort_values(["side", "track_id", "track_name"]).reset_index(drop=True)


def select_tracks(track_table: pd.DataFrame, min_present_frac: float) -> pd.DataFrame:
    selected = track_table[
        (track_table["present_frac"] > float(min_present_frac))
        & track_table["speed_path"].map(lambda path: Path(path).exists())
    ].copy()
    if selected.empty:
        raise ValueError(f"No tracks passed min_present_frac={min_present_frac}")
    return selected.reset_index(drop=True)


def infer_presence_root(speed_root: Path) -> Path:
    return Path(speed_root).parent / "colony_presence_vectors"


def infer_sleep_prediction_root(speed_root: Path) -> Path:
    return Path(speed_root).parent / "sleep_predictions"


def load_presence_tracks(presence_root: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in sorted((Path(presence_root) / "per_track").glob("*/colony_presence_metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        track_name = str(meta.get("track_name", metadata_path.parent.name))
        rows.append(
            {
                "track_name": track_name,
                "presence_metadata_path": metadata_path,
                "presence_path": metadata_path.parent / "colony_presence_i1.npy",
                "presence_frame_min": int(meta["frame_min"]),
                "presence_frame_max": int(meta["frame_max"]),
                "presence_n_frames": int(meta["n_frames"]),
                "presence_n_valid_position_frames": int(meta["n_valid_position_frames"]),
                "presence_n_inside_colony_frames": int(meta["n_inside_colony_frames"]),
                "presence_n_outside_colony_frames": int(meta["n_outside_colony_frames"]),
                "inside_colony_frac_valid": float(meta["inside_colony_frac_valid"])
                if meta.get("inside_colony_frac_valid") is not None
                else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(f"No colony_presence_metadata.json files found under {presence_root}")
    return out.sort_values("track_name").reset_index(drop=True)


def attach_presence_to_tracks(
    tracks: pd.DataFrame,
    presence_root: Path,
    *,
    require_all: bool = False,
) -> pd.DataFrame:
    presence_table = load_presence_tracks(presence_root)
    out = tracks.merge(presence_table, on="track_name", how="left", validate="one_to_one")
    out["has_colony_presence"] = out["presence_path"].map(lambda path: Path(path).exists() if pd.notna(path) else False)
    missing = int((~out["has_colony_presence"]).sum())
    if missing and require_all:
        missing_names = out.loc[~out["has_colony_presence"], "track_name"].head(10).to_list()
        raise FileNotFoundError(f"Missing colony presence vectors for {missing} tracks, examples: {missing_names}")
    if missing:
        print(f"WARNING: missing colony presence vectors for {missing}/{len(out)} selected tracks")
    return out


def load_sleep_prediction_tracks(sleep_root: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in sorted((Path(sleep_root) / "per_track").glob("*/sleep_prediction_metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        track_name = str(meta.get("track_name", metadata_path.parent.name))
        rows.append(
            {
                "track_name": track_name,
                "sleep_metadata_path": metadata_path,
                "predicted_sleep_path": metadata_path.parent / "predicted_sleep_i1.npy",
                "sleep_probability_path": metadata_path.parent / "sleep_probability_f4.npy",
                "wake_probability_path": metadata_path.parent / "wake_probability_f4.npy",
                "sleep_frame_min": meta.get("frame_min"),
                "sleep_frame_max": meta.get("frame_max"),
                "sleep_n_frames": meta.get("n_frames"),
                "sleep_n_predicted_frames": meta.get("n_predicted_frames"),
                "sleep_n_sleep_frames": meta.get("n_sleep_frames"),
                "sleep_n_wake_frames": meta.get("n_wake_frames"),
                "sleep_fraction_predicted_frames": meta.get("sleep_fraction_predicted_frames"),
                "mean_sleep_probability": meta.get("mean_sleep_probability"),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(f"No sleep_prediction_metadata.json files found under {sleep_root}")
    return out.sort_values("track_name").reset_index(drop=True)


def attach_sleep_predictions_to_tracks(
    tracks: pd.DataFrame,
    sleep_root: Path,
    *,
    require_all: bool = False,
) -> pd.DataFrame:
    sleep_table = load_sleep_prediction_tracks(sleep_root)
    out = tracks.merge(sleep_table, on="track_name", how="left", validate="one_to_one")
    out["has_sleep_predictions"] = out["predicted_sleep_path"].map(
        lambda path: Path(path).exists() if pd.notna(path) else False
    )
    missing = int((~out["has_sleep_predictions"]).sum())
    if missing and require_all:
        missing_names = out.loc[~out["has_sleep_predictions"], "track_name"].head(10).to_list()
        raise FileNotFoundError(f"Missing sleep predictions for {missing} tracks, examples: {missing_names}")
    if missing:
        print(f"WARNING: missing sleep predictions for {missing}/{len(out)} selected tracks")
    return out


def start_time_from_track_table(track_table: pd.DataFrame) -> int:
    return parse_track_start_seconds(str(track_table["track_name"].iloc[0]))


def rolling_nanmean(values: np.ndarray, window_bins: int) -> np.ndarray:
    if window_bins <= 1:
        return values.astype(np.float32, copy=True)
    return (
        pd.Series(values)
        .rolling(window=window_bins, center=True, min_periods=max(1, int(np.ceil(window_bins * 0.25))))
        .mean()
        .to_numpy(dtype=np.float32)
    )


def accumulate_side_speed(tracks: pd.DataFrame, side: str) -> tuple[np.ndarray, np.ndarray, float]:
    side_tracks = tracks[tracks["side"] == side]
    if side_tracks.empty:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.uint16), np.nan

    fps_values = side_tracks["fps"].dropna().unique()
    fps = float(fps_values[0])
    if len(fps_values) > 1:
        print(f"WARNING: multiple FPS values for {side}: {fps_values}; using {fps}")

    n_global_frames = int((side_tracks["frame_min"] + side_tracks["n_frames"]).max())
    sum_speed = np.zeros(n_global_frames, dtype=np.float64)
    count_speed = np.zeros(n_global_frames, dtype=np.uint16)

    for i, row in enumerate(side_tracks.itertuples(index=False), start=1):
        if i == 1 or i == len(side_tracks) or i % 10 == 0:
            print(f"{side}: accumulating {i}/{len(side_tracks)} {row.track_name}")
        speed = np.load(row.speed_path, mmap_mode="r")
        start = int(row.frame_min)
        stop = min(start + len(speed), n_global_frames)
        values = speed[: stop - start]
        valid = np.isfinite(values)

        sum_slice = sum_speed[start:stop]
        count_slice = count_speed[start:stop]
        sum_slice[valid] += values[valid]
        count_slice[valid] += 1

    return sum_speed, count_speed, fps


def bin_speed(
    sum_speed: np.ndarray,
    count_speed: np.ndarray,
    fps: float,
    bin_seconds: float,
) -> pd.DataFrame:
    if len(sum_speed) == 0 or not np.isfinite(fps):
        return pd.DataFrame(columns=["time_h", "mean_speed_mm_s", "n_speed_samples"])

    bin_frames = max(1, int(round(float(fps) * float(bin_seconds))))
    n_bins = int(np.ceil(len(sum_speed) / bin_frames))
    pad = n_bins * bin_frames - len(sum_speed)

    bin_sum = np.pad(sum_speed, (0, pad), constant_values=0).reshape(n_bins, bin_frames).sum(axis=1)
    bin_count = (
        np.pad(count_speed.astype(np.uint32), (0, pad), constant_values=0)
        .reshape(n_bins, bin_frames)
        .sum(axis=1)
    )

    mean_speed = np.full(n_bins, np.nan, dtype=np.float64)
    valid = bin_count > 0
    mean_speed[valid] = bin_sum[valid] / bin_count[valid]
    return pd.DataFrame(
        {
            "time_h": np.arange(n_bins) * bin_frames / float(fps) / 3600.0,
            "mean_speed_mm_s": mean_speed,
            "n_speed_samples": bin_count,
        }
    )


def add_clock_columns(timeseries: pd.DataFrame, start_clock_seconds: int) -> pd.DataFrame:
    out = timeseries.copy()
    clock_seconds = (
        int(start_clock_seconds) + np.rint(out["time_h"].to_numpy() * 3600.0).astype(np.int64)
    ) % (24 * 3600)
    out.insert(1, "clock_time", [format_clock_time(value) for value in clock_seconds])
    return out


def compute_colony_speed_timeseries(
    tracks: pd.DataFrame,
    bin_seconds: float,
    start_clock_seconds: int | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray, float]]]:
    side_data = {}
    binned = {}
    for side in ("left", "right"):
        side_data[side] = accumulate_side_speed(tracks, side)
        binned[side] = bin_speed(*side_data[side], bin_seconds=bin_seconds).rename(
            columns={
                "mean_speed_mm_s": f"{side}_mean_speed_mm_s",
                "n_speed_samples": f"{side}_n_speed_samples",
            }
        )

    timeseries = pd.merge(binned["left"], binned["right"], on="time_h", how="outer")
    timeseries = timeseries.sort_values("time_h").reset_index(drop=True)
    if start_clock_seconds is not None:
        timeseries = add_clock_columns(timeseries, start_clock_seconds)
    return timeseries, side_data


def smooth_colony_timeseries(
    timeseries: pd.DataFrame,
    smooth_seconds: float,
    bin_seconds: float,
) -> pd.DataFrame:
    out = timeseries.copy()
    window_bins = max(1, int(round(float(smooth_seconds) / float(bin_seconds))))
    for side in ("left", "right"):
        raw_col = f"{side}_mean_speed_mm_s"
        smooth_col = f"{side}_smoothed_speed_mm_s"
        out[smooth_col] = rolling_nanmean(out[raw_col].to_numpy(dtype=np.float32), window_bins)
    return out


def add_light_lines(
    ax: plt.Axes,
    start_clock_seconds: int,
    max_time_h: float,
    *,
    light_off_hour: float = 18.0,
    light_on_hour: float = 6.0,
) -> None:
    events = [
        (light_off_hour * 3600, "lights off", "0.20", "--"),
        (light_on_hour * 3600, "lights on", "0.65", "-."),
    ]
    for clock_seconds, label_suffix, color, linestyle in events:
        label = f"{format_clock_time(clock_seconds)} {label_suffix}"
        first_h = (clock_seconds - start_clock_seconds) / 3600.0
        while first_h < 0:
            first_h += 24.0
        for i, time_h in enumerate(np.arange(first_h, max_time_h + 1e-9, 24.0)):
            ax.axvline(
                time_h,
                color=color,
                linestyle=linestyle,
                lw=1,
                alpha=0.75,
                label=label if i == 0 else None,
            )


def plot_colony_speed(
    timeseries: pd.DataFrame,
    start_clock_seconds: int,
    smooth_seconds: float,
    bin_seconds: float,
    *,
    light_off_hour: float = 18.0,
    light_on_hour: float = 6.0,
    ylim: tuple[float, float] | None = None,
) -> pd.DataFrame:
    plot_df = smooth_colony_timeseries(timeseries, smooth_seconds, bin_seconds)

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

    add_light_lines(
        ax,
        start_clock_seconds,
        float(plot_df["time_h"].max()),
        light_off_hour=light_off_hour,
        light_on_hour=light_on_hour,
    )
    ax.set_xlabel(f"Elapsed time from {format_clock_time(start_clock_seconds)} (h)")
    ax.set_ylabel("Mean speed (mm/s)")
    ax.set_title(f"Colony-average ant speed, {smooth_seconds / 60:g} min smoothing")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()
    return plot_df


def choose_tracks(
    tracks: pd.DataFrame,
    side: str | None = "left",
    row_numbers: list[int] | None = None,
    track_ids: list[int | str] | None = None,
    max_tracks: int | None = 6,
    sort_tracks: bool = True,
) -> pd.DataFrame:
    if row_numbers is not None:
        chosen = tracks.loc[row_numbers].copy()
    else:
        chosen = tracks if side in (None, "both") else tracks[tracks["side"] == side]
        if track_ids is not None:
            wanted = {str(value) for value in track_ids}
            chosen = chosen[chosen["track_id"].astype(str).isin(wanted)]
        if max_tracks is not None:
            chosen = chosen.head(int(max_tracks))

    if chosen.empty:
        raise ValueError("No tracks selected")
    if sort_tracks:
        return chosen.sort_values(["side", "track_id", "track_name"])
    return chosen


def binned_track_speed(row: pd.Series, bin_seconds: float) -> pd.DataFrame:
    speed = np.load(row["speed_path"], mmap_mode="r")
    fps = float(row["fps"])
    frame_min = int(row["frame_min"])

    bin_frames = max(1, int(round(fps * float(bin_seconds))))
    first_bin = frame_min // bin_frames
    n_bins = int(np.ceil((frame_min + len(speed)) / bin_frames)) - first_bin
    values = np.full(n_bins, np.nan, dtype=np.float32)

    valid = np.isfinite(speed)
    if valid.any():
        valid_idx = np.flatnonzero(valid)
        local_bin_idx = ((frame_min + valid_idx) // bin_frames) - first_bin
        bin_sum = np.bincount(local_bin_idx, weights=speed[valid_idx], minlength=n_bins)
        bin_count = np.bincount(local_bin_idx, minlength=n_bins)
        keep = bin_count > 0
        values[keep] = (bin_sum[keep] / bin_count[keep]).astype(np.float32)

    return pd.DataFrame(
        {
            "time_h": ((first_bin + np.arange(n_bins)) * bin_frames) / fps / 3600.0,
            "speed_mm_s": values,
            "side": row["side"],
            "track_id": row["track_id"],
            "track_name": row["track_name"],
            "track_row": row.name,
        }
    )


def binned_track_presence(row: pd.Series, bin_seconds: float) -> pd.DataFrame:
    if "presence_path" not in row or pd.isna(row["presence_path"]):
        raise ValueError(f"No presence_path for track {row.get('track_name', '<unknown>')}")

    presence = np.load(row["presence_path"], mmap_mode="r")
    fps = float(row["fps"])
    frame_min = int(row.get("presence_frame_min", row["frame_min"]))

    bin_frames = max(1, int(round(fps * float(bin_seconds))))
    first_bin = frame_min // bin_frames
    n_bins = int(np.ceil((frame_min + len(presence)) / bin_frames)) - first_bin
    inside_frac = np.full(n_bins, np.nan, dtype=np.float32)

    valid = presence >= 0
    if valid.any():
        valid_idx = np.flatnonzero(valid)
        local_bin_idx = ((frame_min + valid_idx) // bin_frames) - first_bin
        inside_sum = np.bincount(local_bin_idx, weights=(presence[valid_idx] == 1).astype(np.float32), minlength=n_bins)
        valid_count = np.bincount(local_bin_idx, minlength=n_bins)
        keep = valid_count > 0
        inside_frac[keep] = (inside_sum[keep] / valid_count[keep]).astype(np.float32)

    return pd.DataFrame(
        {
            "time_h": ((first_bin + np.arange(n_bins)) * bin_frames) / fps / 3600.0,
            "inside_colony_frac": inside_frac,
            "side": row["side"],
            "track_id": row["track_id"],
            "track_name": row["track_name"],
            "track_row": row.name,
        }
    )


def plot_speed_and_presence_for_ant(
    tracks: pd.DataFrame,
    *,
    row_number: int | None = None,
    track_id: int | str | None = None,
    side: str | None = "left",
    bin_seconds: float = 60.0,
    speed_smooth_seconds: float = 10 * 60.0,
    speed_ylim: tuple[float, float] | None = None,
) -> pd.DataFrame:
    row_numbers = [int(row_number)] if row_number is not None else None
    track_ids = [track_id] if track_id is not None else None
    chosen = choose_tracks(
        tracks,
        side=side,
        row_numbers=row_numbers,
        track_ids=track_ids,
        max_tracks=1,
        sort_tracks=False,
    )
    row = chosen.iloc[0]

    speed_df = binned_track_speed(row, bin_seconds)
    window_bins = max(1, int(round(float(speed_smooth_seconds) / float(bin_seconds))))
    speed_df["smoothed_speed_mm_s"] = rolling_nanmean(speed_df["speed_mm_s"].to_numpy(), window_bins)
    presence_df = binned_track_presence(row, bin_seconds)

    out = pd.merge(
        speed_df,
        presence_df[["time_h", "inside_colony_frac"]],
        on="time_h",
        how="outer",
    ).sort_values("time_h")

    fig, (speed_ax, presence_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    label = f"{row['side']} track {row['track_id']} row {row.name}"
    speed_ax.plot(out["time_h"], out["smoothed_speed_mm_s"], lw=1.4, color="tab:blue", label=label)
    speed_ax.set_ylabel("Speed (mm/s)")
    speed_ax.set_title(f"Speed and colony occupancy: {label}")
    if speed_ylim is not None:
        speed_ax.set_ylim(*speed_ylim)
    speed_ax.legend(fontsize=8)
    speed_ax.grid(True, alpha=0.25)

    presence_ax.step(out["time_h"], out["inside_colony_frac"], where="post", color="tab:green", lw=1.0)
    presence_ax.fill_between(
        out["time_h"].to_numpy(float),
        0,
        out["inside_colony_frac"].fillna(0).to_numpy(float),
        step="post",
        color="tab:green",
        alpha=0.25,
    )
    presence_ax.set_ylim(-0.05, 1.05)
    presence_ax.set_yticks([0, 1])
    presence_ax.set_yticklabels(["out", "in"])
    presence_ax.set_xlabel("Elapsed time (h)")
    presence_ax.set_ylabel("Colony")
    presence_ax.grid(True, alpha=0.25)

    fig.tight_layout()
    plt.show()
    return out.reset_index(drop=True)


def plot_individual_speeds(
    tracks: pd.DataFrame,
    *,
    side: str | None = "left",
    row_numbers: list[int] | None = None,
    track_ids: list[int | str] | None = None,
    max_tracks: int | None = 6,
    bin_seconds: float = 60.0,
    smooth_seconds: float = 10 * 60.0,
    ylim: tuple[float, float] | None = None,
) -> pd.DataFrame:
    chosen = choose_tracks(tracks, side, row_numbers, track_ids, max_tracks)
    window_bins = max(1, int(round(float(smooth_seconds) / float(bin_seconds))))

    rows = []
    fig, ax = plt.subplots(figsize=(12, 5))
    for row_idx, row in chosen.iterrows():
        track_df = binned_track_speed(row, bin_seconds)
        track_df["smoothed_speed_mm_s"] = rolling_nanmean(track_df["speed_mm_s"].to_numpy(), window_bins)
        label = f"{row['side']} track {row['track_id']} row {row_idx}"
        ax.plot(track_df["time_h"], track_df["smoothed_speed_mm_s"], lw=1.3, label=label)
        rows.append(track_df)

    ax.set_xlabel("Elapsed time (h)")
    ax.set_ylabel("Speed (mm/s)")
    ax.set_title(f"Individual ant speed, {smooth_seconds / 60:g} min smoothing")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()
    return pd.concat(rows, ignore_index=True)


def build_speed_image(
    tracks: pd.DataFrame,
    *,
    side: str | None = "both",
    bin_seconds: float = 60.0,
    smooth_seconds: float = 10 * 60.0,
    order_by_colony_frac: bool = False,
    colony_frac_ascending: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    chosen = choose_tracks(tracks, side=side, row_numbers=None, track_ids=None, max_tracks=None)
    if order_by_colony_frac:
        if "inside_colony_frac_valid" not in chosen.columns:
            raise ValueError("tracks must have inside_colony_frac_valid; run attach_presence_to_tracks first")
        chosen = chosen.sort_values(
            "inside_colony_frac_valid",
            ascending=bool(colony_frac_ascending),
            na_position="last",
            kind="mergesort",
        )
    fps = float(chosen["fps"].iloc[0])
    bin_frames = max(1, int(round(fps * float(bin_seconds))))
    n_bins = int(np.ceil((chosen["frame_min"] + chosen["n_frames"]).max() / bin_frames))
    image = np.full((len(chosen), n_bins), np.nan, dtype=np.float32)

    for image_row, (_, row) in enumerate(chosen.iterrows()):
        if image_row == 0 or image_row == len(chosen) - 1 or (image_row + 1) % 25 == 0:
            print(f"image: loading {image_row + 1}/{len(chosen)} {row['track_name']}")
        speed = np.load(row["speed_path"], mmap_mode="r")
        valid = np.isfinite(speed)
        if not valid.any():
            continue
        valid_idx = np.flatnonzero(valid)
        bin_idx = (int(row["frame_min"]) + valid_idx) // bin_frames
        bin_sum = np.bincount(bin_idx, weights=speed[valid_idx], minlength=n_bins)
        bin_count = np.bincount(bin_idx, minlength=n_bins)
        keep = bin_count > 0
        image[image_row, keep] = (bin_sum[keep] / bin_count[keep]).astype(np.float32)

    window_bins = max(1, int(round(float(smooth_seconds) / float(bin_seconds))))
    if window_bins > 1:
        for image_row in range(image.shape[0]):
            image[image_row] = rolling_nanmean(image[image_row], window_bins)

    image_tracks = chosen.copy()
    image_tracks.insert(0, "track_row", image_tracks.index)
    image_tracks = image_tracks.reset_index(drop=True)
    time_h = np.arange(n_bins) * bin_frames / fps / 3600.0
    return image, image_tracks, time_h


def plot_speed_image(
    tracks: pd.DataFrame,
    *,
    side: str | None = "both",
    bin_seconds: float = 60.0,
    smooth_seconds: float = 10 * 60.0,
    order_by_colony_frac: bool = False,
    colony_frac_ascending: bool = False,
    vmin: float | None = 0.0,
    vmax: float | None = None,
    vmax_percentile: float | None = 99.0,
    cmap: str = "viridis",
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    image, image_tracks, time_h = build_speed_image(
        tracks,
        side=side,
        bin_seconds=bin_seconds,
        smooth_seconds=smooth_seconds,
        order_by_colony_frac=order_by_colony_frac,
        colony_frac_ascending=colony_frac_ascending,
    )
    if vmax is None and vmax_percentile is not None:
        vmax = float(np.nanpercentile(image, vmax_percentile))

    plt.figure(figsize=(12, 6))
    extent = [float(time_h[0]), float(time_h[-1]), image.shape[0] - 0.5, -0.5]
    im = plt.imshow(
        image,
        aspect="auto",
        interpolation="none",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    if not order_by_colony_frac and side in (None, "both") and image_tracks["side"].nunique() > 1:
        side_values = image_tracks["side"].to_numpy()
        boundaries = np.flatnonzero(side_values[1:] != side_values[:-1]) + 0.5
        for boundary in boundaries:
            plt.axhline(boundary, color="white", lw=1, alpha=0.8)

    plt.xlabel("Elapsed time (h)")
    plt.ylabel("Ant track")
    title = f"All ant speeds, {smooth_seconds / 60:g} min smoothing"
    if order_by_colony_frac:
        direction = "low to high" if colony_frac_ascending else "high to low"
        title += f", colony use {direction}"
    plt.title(title)
    plt.colorbar(im, label="Speed (mm/s)")
    plt.tight_layout()
    plt.show()
    return image, image_tracks, time_h


def infer_conductor_path(path: Path) -> Path:
    """Find the conductor log for a block from a block, stitched, or speed root."""
    root = Path(path)
    if root.is_file():
        if root.name.startswith("conductor_") and root.suffix == ".txt":
            return root
        root = root.parent

    for base in [root, *root.parents]:
        matches = sorted(base.glob("conductor_*.txt"))
        if matches:
            return matches[-1]

    raise FileNotFoundError(f"Could not find conductor_*.txt above {path}")


def _conductor_message(line: str) -> str:
    parts = line.rstrip("\n").split("\t", 2)
    if len(parts) == 3:
        return parts[2].strip()
    return line.strip()


def _last_session_lines(lines: list[str]) -> tuple[list[str], str | None, int]:
    start_indices = [
        idx for idx, line in enumerate(lines)
        if "\tSESS\t==== START " in line or "\tSESS\t==== START" in line
    ]
    if not start_indices:
        return lines, None, 0

    start_idx = start_indices[-1]
    session_name = None
    match = re.search(r"START\s+(sess_\d{8}_\d{6})", lines[start_idx])
    if match is not None:
        session_name = match.group(1)

    stop_idx = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        if "\tSESS\t==== STOP " in lines[idx]:
            stop_idx = idx
            break
    return lines[start_idx:stop_idx], session_name, start_idx + 1


def _coerce_pulse_columns(pulses: pd.DataFrame) -> pd.DataFrame:
    out = pulses.copy()
    for col in out.columns:
        if col in {"iso_time", "session_name", "conductor_path"}:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().all() and np.all(np.isclose(numeric, np.rint(numeric))):
            out[col] = numeric.astype("int64")
        elif numeric.notna().any():
            out[col] = numeric

    if "iso_time" in out.columns:
        out["iso_datetime"] = pd.to_datetime(out["iso_time"], errors="coerce")
    return out


def load_last_session_csv_pulses(conductor_path: Path) -> pd.DataFrame:
    """Load CSV_PULSE rows from the final recording session in a conductor log."""
    conductor_path = Path(conductor_path)
    lines = conductor_path.read_text(encoding="utf-8", errors="replace").splitlines()
    session_lines, session_name, session_start_line = _last_session_lines(lines)

    header: list[str] | None = None
    rows: list[list[str]] = []
    for line in session_lines:
        message = _conductor_message(line)
        if not message.startswith("CSV_PULSE,"):
            continue

        fields = next(csv.reader([message]))
        if len(fields) < 2:
            continue
        if fields[1] == "iso_time":
            header = fields[1:]
        else:
            rows.append(fields[1:])

    if not rows:
        raise ValueError(
            f"No CSV_PULSE rows found in the last session of {conductor_path}"
        )

    if header is None:
        header = [
            "iso_time",
            "trial",
            "duty",
            "dur_s",
            "interval_s",
            "camFrameStart",
            "camFrameEnd",
            "fs_hz",
            "samples",
            "gyro_rms_dps",
            "gyro_peak_dps",
            "acc_rms_g",
            "acc_peak_g",
            "temp_mean_C",
        ]

    width = min(len(header), min(len(row) for row in rows))
    out = pd.DataFrame([row[:width] for row in rows], columns=header[:width])
    out = _coerce_pulse_columns(out)
    out.insert(0, "pulse_row", np.arange(len(out), dtype=np.int64))
    out["session_name"] = session_name
    out["session_start_line"] = session_start_line
    out["conductor_path"] = str(conductor_path)
    return out


def _pulse_frame_column(pulses: pd.DataFrame, frame_col: str | None) -> str:
    if frame_col is not None:
        if frame_col not in pulses.columns:
            raise KeyError(f"Pulse frame column {frame_col!r} is not in the pulse table")
        return frame_col

    for candidate in ("camFrameStart", "cam_frame_start", "cam_frame", "CamFrame"):
        if candidate in pulses.columns:
            return candidate
    raise KeyError("Could not find a pulse frame column in the pulse table")


def _pulse_stimulus_strength_column(
    pulses: pd.DataFrame,
    stimulus_strength_col: str | None = "auto",
) -> str:
    requested_col = "auto" if stimulus_strength_col in {None, ""} else stimulus_strength_col
    if requested_col != "auto":
        if requested_col not in pulses.columns:
            raise KeyError(
                f"Pulse stimulus-strength column {requested_col!r} is not in the pulse table"
            )
        values = pd.to_numeric(pulses[requested_col], errors="coerce")
        if not values.notna().any():
            raise ValueError(
                f"Pulse stimulus-strength column {requested_col!r} has no finite numeric values"
            )
        return requested_col

    candidates = (
        "duty",
        "duty_pct",
        "duty_percent",
        "duty_cycle",
        "stimulus_strength",
        "strength",
        "gyro_rms_dps",
        "gyro_peak_dps",
        "acc_rms_g",
        "acc_peak_g",
    )
    for candidate in candidates:
        if candidate not in pulses.columns:
            continue
        values = pd.to_numeric(pulses[candidate], errors="coerce")
        if values.notna().any():
            return candidate
    raise KeyError(
        "Could not find a numeric pulse stimulus-strength column. "
        f"Tried: {', '.join(candidates)}"
    )


def sort_pulses_by_stimulus_strength(
    pulses: pd.DataFrame,
    *,
    stimulus_strength_col: str | None = "auto",
    ascending: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Return pulses sorted by stimulus strength, keeping row order stable within ties."""
    resolved_col = _pulse_stimulus_strength_column(pulses, stimulus_strength_col)
    out = pulses.copy()
    sort_value_col = "__stimulus_strength_sort_value"
    sort_order_col = "__stimulus_strength_sort_order"
    out[sort_value_col] = pd.to_numeric(out[resolved_col], errors="coerce")
    out[sort_order_col] = np.arange(len(out), dtype=np.int64)
    out = out.sort_values(
        [sort_value_col, sort_order_col],
        ascending=[ascending, True],
        na_position="last",
        kind="mergesort",
    )
    out["stimulus_strength"] = out[sort_value_col]
    out["stimulus_strength_col"] = resolved_col
    out = out.drop(columns=[sort_value_col, sort_order_col])
    return out, resolved_col


def accumulate_colony_speed_frames(
    tracks: pd.DataFrame,
    *,
    side: str | None = "both",
) -> tuple[np.ndarray, np.ndarray, float, pd.DataFrame]:
    """Average speed over selected tracks at the original frame resolution."""
    chosen = choose_tracks(
        tracks,
        side=side,
        row_numbers=None,
        track_ids=None,
        max_tracks=None,
    )
    fps_values = chosen["fps"].dropna().unique()
    if len(fps_values) == 0:
        raise ValueError("Selected tracks do not have FPS metadata")
    fps = float(fps_values[0])
    if len(fps_values) > 1:
        print(f"WARNING: multiple FPS values in selected tracks: {fps_values}; using {fps}")

    n_global_frames = int((chosen["frame_min"] + chosen["n_frames"]).max())
    sum_speed = np.zeros(n_global_frames, dtype=np.float64)
    count_speed = np.zeros(n_global_frames, dtype=np.uint32)

    for i, row in enumerate(chosen.itertuples(index=False), start=1):
        if i == 1 or i == len(chosen) or i % 25 == 0:
            print(f"pulse response: accumulating {i}/{len(chosen)} {row.track_name}")
        speed = np.load(row.speed_path, mmap_mode="r")
        start = int(row.frame_min)
        stop = min(start + len(speed), n_global_frames)
        if stop <= start:
            continue

        values = speed[: stop - start]
        valid = np.isfinite(values)
        sum_slice = sum_speed[start:stop]
        count_slice = count_speed[start:stop]
        sum_slice[valid] += values[valid]
        count_slice[valid] += 1

    mean_speed = np.full(n_global_frames, np.nan, dtype=np.float32)
    valid = count_speed > 0
    mean_speed[valid] = (sum_speed[valid] / count_speed[valid]).astype(np.float32)
    return mean_speed, count_speed, fps, chosen.reset_index(drop=True)


def _average_pulse_response(
    response: np.ndarray,
    response_count: np.ndarray,
    relative_time_s: np.ndarray,
) -> pd.DataFrame:
    finite = np.isfinite(response)
    n_pulses_per_frame = finite.sum(axis=0)
    mean_speed = np.full(response.shape[1], np.nan, dtype=np.float32)
    std_speed = np.full(response.shape[1], np.nan, dtype=np.float32)
    keep = n_pulses_per_frame > 0
    if keep.any():
        response_sum = np.where(finite, response, 0.0).sum(axis=0, dtype=np.float64)
        mean_speed[keep] = (response_sum[keep] / n_pulses_per_frame[keep]).astype(np.float32)
        centered = np.where(finite, response - mean_speed, 0.0)
        variance = (centered * centered).sum(axis=0, dtype=np.float64)
        std_speed[keep] = np.sqrt(variance[keep] / n_pulses_per_frame[keep]).astype(np.float32)
    sem_speed = np.full_like(mean_speed, np.nan, dtype=np.float32)
    sem_speed[keep] = (std_speed[keep] / np.sqrt(n_pulses_per_frame[keep])).astype(np.float32)
    track_count_values = np.where(response_count > 0, response_count, np.nan)
    finite_track_count = np.isfinite(track_count_values)
    mean_track_samples = np.full(response.shape[1], np.nan, dtype=np.float32)
    track_keep = finite_track_count.sum(axis=0) > 0
    if track_keep.any():
        track_sum = np.where(finite_track_count, track_count_values, 0.0).sum(axis=0, dtype=np.float64)
        mean_track_samples[track_keep] = (
            track_sum[track_keep] / finite_track_count.sum(axis=0)[track_keep]
        ).astype(np.float32)
    return pd.DataFrame(
        {
            "relative_time_s": relative_time_s,
            "mean_speed_mm_s": mean_speed,
            "sem_speed_mm_s": sem_speed,
            "n_pulses": n_pulses_per_frame,
            "mean_track_samples": mean_track_samples,
        }
    )


def _nanmean_rows(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    means = np.full(values.shape[0], np.nan, dtype=np.float32)
    keep = counts > 0
    if keep.any():
        row_sum = np.where(finite, values, 0.0).sum(axis=1, dtype=np.float64)
        means[keep] = (row_sum[keep] / counts[keep]).astype(np.float32)
    return means


def _centered_time_extent(relative_time_s: np.ndarray) -> tuple[float, float]:
    times = np.asarray(relative_time_s, dtype=np.float64)
    if times.size == 0:
        return 0.0, 0.0
    if times.size == 1:
        half_step = 0.5
    else:
        diffs = np.diff(times)
        finite_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        half_step = 0.5 * float(np.median(finite_diffs)) if finite_diffs.size else 0.5
    return float(times[0] - half_step), float(times[-1] + half_step)


def build_pulse_triggered_colony_speed(
    tracks: pd.DataFrame,
    pulses: pd.DataFrame,
    *,
    pre_seconds: float,
    post_seconds: float,
    side: str | None = "both",
    frame_col: str | None = "camFrameStart",
    smooth_seconds: float = 0.0,
    sort_by_stimulus_strength: bool = False,
    stimulus_strength_col: str | None = "auto",
    stimulus_sort_ascending: bool = True,
) -> dict[str, object]:
    """Build pulse x time colony-speed responses averaged over ant tracks."""
    resolved_frame_col = _pulse_frame_column(pulses, frame_col)
    pulse_table = pulses.copy()
    pulse_table["pulse_frame"] = pd.to_numeric(
        pulse_table[resolved_frame_col],
        errors="coerce",
    )
    pulse_table = pulse_table[pulse_table["pulse_frame"].notna()].copy()
    if pulse_table.empty:
        raise ValueError(f"No finite pulse frames found in {resolved_frame_col}")
    pulse_table["pulse_frame"] = pulse_table["pulse_frame"].round().astype("int64")
    resolved_stimulus_strength_col = None
    if sort_by_stimulus_strength:
        pulse_table, resolved_stimulus_strength_col = sort_pulses_by_stimulus_strength(
            pulse_table,
            stimulus_strength_col=stimulus_strength_col,
            ascending=stimulus_sort_ascending,
        )
    pulse_table = pulse_table.reset_index(drop=True)
    pulse_table.insert(0, "response_row", np.arange(len(pulse_table), dtype=np.int64))

    colony_speed, colony_count, fps, selected_tracks = accumulate_colony_speed_frames(
        tracks,
        side=side,
    )
    pre_frames = int(round(float(pre_seconds) * fps))
    post_frames = int(round(float(post_seconds) * fps))
    relative_frames = np.arange(-pre_frames, post_frames + 1, dtype=np.int64)
    relative_time_s = relative_frames.astype(np.float64) / float(fps)

    response = np.full((len(pulse_table), len(relative_frames)), np.nan, dtype=np.float32)
    response_count = np.zeros(response.shape, dtype=np.uint32)

    for row_idx, pulse_frame in enumerate(pulse_table["pulse_frame"].to_numpy(dtype=np.int64)):
        frame_idx = pulse_frame + relative_frames
        valid = (frame_idx >= 0) & (frame_idx < len(colony_speed))
        response[row_idx, valid] = colony_speed[frame_idx[valid]]
        response_count[row_idx, valid] = colony_count[frame_idx[valid]]

    if smooth_seconds > 0:
        window_frames = max(1, int(round(float(smooth_seconds) * fps)))
        if window_frames > 1:
            for row_idx in range(response.shape[0]):
                response[row_idx] = rolling_nanmean(response[row_idx], window_frames)

    pulse_table["valid_fraction"] = np.isfinite(response).mean(axis=1)
    pulse_table["mean_speed_mm_s"] = _nanmean_rows(response)

    average = _average_pulse_response(response, response_count, relative_time_s)

    return {
        "average": average,
        "matrix": response,
        "matrix_track_counts": response_count,
        "relative_time_s": relative_time_s,
        "pulse_table": pulse_table,
        "track_table": selected_tracks,
        "fps": fps,
        "side": side,
        "frame_col": resolved_frame_col,
        "sort_by_stimulus_strength": sort_by_stimulus_strength,
        "stimulus_strength_col": resolved_stimulus_strength_col,
        "stimulus_sort_ascending": stimulus_sort_ascending,
    }


def build_sleep_split_pulse_triggered_colony_speed(
    tracks: pd.DataFrame,
    pulses: pd.DataFrame,
    *,
    pre_seconds: float,
    post_seconds: float,
    side: str | None = "both",
    frame_col: str | None = "camFrameStart",
    smooth_seconds: float = 0.0,
    sort_by_stimulus_strength: bool = False,
    stimulus_strength_col: str | None = "auto",
    stimulus_sort_ascending: bool = True,
) -> dict[str, object]:
    """Build pulse responses split by each ant's predicted sleep state at pulse onset."""
    required_cols = {"predicted_sleep_path", "sleep_frame_min"}
    missing_cols = sorted(required_cols - set(tracks.columns))
    if missing_cols:
        raise KeyError(
            "Tracks are missing sleep prediction columns "
            f"{missing_cols}; call attach_sleep_predictions_to_tracks first."
        )

    resolved_frame_col = _pulse_frame_column(pulses, frame_col)
    pulse_table = pulses.copy()
    pulse_table["pulse_frame"] = pd.to_numeric(
        pulse_table[resolved_frame_col],
        errors="coerce",
    )
    pulse_table = pulse_table[pulse_table["pulse_frame"].notna()].copy()
    if pulse_table.empty:
        raise ValueError(f"No finite pulse frames found in {resolved_frame_col}")
    pulse_table["pulse_frame"] = pulse_table["pulse_frame"].round().astype("int64")
    resolved_stimulus_strength_col = None
    if sort_by_stimulus_strength:
        pulse_table, resolved_stimulus_strength_col = sort_pulses_by_stimulus_strength(
            pulse_table,
            stimulus_strength_col=stimulus_strength_col,
            ascending=stimulus_sort_ascending,
        )
    pulse_table = pulse_table.reset_index(drop=True)
    pulse_table.insert(0, "response_row", np.arange(len(pulse_table), dtype=np.int64))

    chosen = choose_tracks(
        tracks,
        side=side,
        row_numbers=None,
        track_ids=None,
        max_tracks=None,
    ).reset_index(drop=True)
    fps_values = chosen["fps"].dropna().unique()
    if len(fps_values) == 0:
        raise ValueError("Selected tracks do not have FPS metadata")
    fps = float(fps_values[0])
    if len(fps_values) > 1:
        print(f"WARNING: multiple FPS values in selected tracks: {fps_values}; using {fps}")

    pre_frames = int(round(float(pre_seconds) * fps))
    post_frames = int(round(float(post_seconds) * fps))
    relative_frames = np.arange(-pre_frames, post_frames + 1, dtype=np.int64)
    relative_time_s = relative_frames.astype(np.float64) / float(fps)
    pulse_frames = pulse_table["pulse_frame"].to_numpy(dtype=np.int64)

    state_info = {
        "sleeping": {"value": 1, "label": "sleeping at pulse", "color": "tab:orange"},
        "not_sleeping": {"value": 0, "label": "awake at pulse", "color": "tab:blue"},
    }
    response_sum = {
        key: np.zeros((len(pulse_table), len(relative_frames)), dtype=np.float64)
        for key in state_info
    }
    response_count = {
        key: np.zeros((len(pulse_table), len(relative_frames)), dtype=np.uint32)
        for key in state_info
    }
    state_counts = {
        "sleeping": np.zeros(len(pulse_table), dtype=np.uint32),
        "not_sleeping": np.zeros(len(pulse_table), dtype=np.uint32),
        "unknown": np.zeros(len(pulse_table), dtype=np.uint32),
    }

    for i, row in enumerate(chosen.itertuples(index=False), start=1):
        if i == 1 or i == len(chosen) or i % 25 == 0:
            print(f"sleep split pulse response: loading {i}/{len(chosen)} {row.track_name}")

        states = np.full(len(pulse_table), -1, dtype=np.int8)
        sleep_path = Path(row.predicted_sleep_path) if pd.notna(row.predicted_sleep_path) else None
        sleep_frame_min = int(row.sleep_frame_min) if pd.notna(row.sleep_frame_min) else None
        if sleep_path is not None and sleep_path.exists() and sleep_frame_min is not None:
            sleep_state = np.load(sleep_path, mmap_mode="r")
            sleep_idx = pulse_frames - sleep_frame_min
            valid_sleep = (sleep_idx >= 0) & (sleep_idx < len(sleep_state))
            if valid_sleep.any():
                states[valid_sleep] = sleep_state[sleep_idx[valid_sleep]]

        state_counts["sleeping"] += (states == 1)
        state_counts["not_sleeping"] += (states == 0)
        state_counts["unknown"] += ~((states == 1) | (states == 0))
        if not ((states == 1) | (states == 0)).any():
            continue

        speed = np.load(row.speed_path, mmap_mode="r")
        speed_frame_min = int(row.frame_min)
        for pulse_idx, pulse_frame in enumerate(pulse_frames):
            if states[pulse_idx] == 1:
                state_key = "sleeping"
            elif states[pulse_idx] == 0:
                state_key = "not_sleeping"
            else:
                continue

            local_idx = pulse_frame + relative_frames - speed_frame_min
            in_bounds = (local_idx >= 0) & (local_idx < len(speed))
            if not in_bounds.any():
                continue
            response_cols = np.flatnonzero(in_bounds)
            values = speed[local_idx[in_bounds]]
            finite = np.isfinite(values)
            if not finite.any():
                continue
            response_cols = response_cols[finite]
            response_sum[state_key][pulse_idx, response_cols] += values[finite]
            response_count[state_key][pulse_idx, response_cols] += 1

    responses: dict[str, np.ndarray] = {}
    averages: dict[str, pd.DataFrame] = {}
    for state_key in state_info:
        response = np.full(response_sum[state_key].shape, np.nan, dtype=np.float32)
        valid = response_count[state_key] > 0
        response[valid] = (response_sum[state_key][valid] / response_count[state_key][valid]).astype(np.float32)
        if smooth_seconds > 0:
            window_frames = max(1, int(round(float(smooth_seconds) * fps)))
            if window_frames > 1:
                for row_idx in range(response.shape[0]):
                    response[row_idx] = rolling_nanmean(response[row_idx], window_frames)
        responses[state_key] = response
        averages[state_key] = _average_pulse_response(response, response_count[state_key], relative_time_s)
        pulse_table[f"{state_key}_valid_fraction"] = np.isfinite(response).mean(axis=1)
        pulse_table[f"{state_key}_mean_speed_mm_s"] = _nanmean_rows(response)

    pulse_table["n_sleeping_tracks_at_pulse"] = state_counts["sleeping"]
    pulse_table["n_not_sleeping_tracks_at_pulse"] = state_counts["not_sleeping"]
    pulse_table["n_unknown_sleep_tracks_at_pulse"] = state_counts["unknown"]
    pulse_table["n_sleep_classified_tracks_at_pulse"] = (
        pulse_table["n_sleeping_tracks_at_pulse"] + pulse_table["n_not_sleeping_tracks_at_pulse"]
    )

    return {
        "average_by_state": averages,
        "matrix_by_state": responses,
        "matrix_track_counts_by_state": response_count,
        "relative_time_s": relative_time_s,
        "pulse_table": pulse_table,
        "track_table": chosen,
        "state_info": state_info,
        "fps": fps,
        "side": side,
        "frame_col": resolved_frame_col,
        "sort_by_stimulus_strength": sort_by_stimulus_strength,
        "stimulus_strength_col": resolved_stimulus_strength_col,
        "stimulus_sort_ascending": stimulus_sort_ascending,
    }


def plot_pulse_triggered_colony_speed(
    tracks: pd.DataFrame,
    pulses: pd.DataFrame,
    *,
    pre_seconds: float,
    post_seconds: float,
    side: str | None = "both",
    frame_col: str | None = "camFrameStart",
    smooth_seconds: float = 0.0,
    vmin: float | None = 0.0,
    vmax: float | None = None,
    vmax_percentile: float | None = 99.0,
    cmap: str = "viridis",
    ylim: tuple[float, float] | None = None,
    sort_by_stimulus_strength: bool = False,
    stimulus_strength_col: str | None = "auto",
    stimulus_sort_ascending: bool = True,
) -> dict[str, object]:
    result = build_pulse_triggered_colony_speed(
        tracks,
        pulses,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        side=side,
        frame_col=frame_col,
        smooth_seconds=smooth_seconds,
        sort_by_stimulus_strength=sort_by_stimulus_strength,
        stimulus_strength_col=stimulus_strength_col,
        stimulus_sort_ascending=stimulus_sort_ascending,
    )

    average = result["average"]
    matrix = result["matrix"]
    pulse_table = result["pulse_table"]
    relative_time_s = result["relative_time_s"]

    if vmax is None and vmax_percentile is not None:
        vmax = float(np.nanpercentile(matrix, vmax_percentile))

    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[2, 3],
        width_ratios=[1, 0.035],
        hspace=0.08,
        wspace=0.04,
    )
    avg_ax = fig.add_subplot(grid[0, 0])
    matrix_ax = fig.add_subplot(grid[1, 0], sharex=avg_ax)
    colorbar_ax = fig.add_subplot(grid[1, 1])
    avg_ax.tick_params(labelbottom=False)

    x = average["relative_time_s"].to_numpy(dtype=float)
    y = average["mean_speed_mm_s"].to_numpy(dtype=float)
    sem = average["sem_speed_mm_s"].to_numpy(dtype=float)
    avg_ax.plot(x, y, color="tab:blue", lw=1.8, label="mean over pulses")
    avg_ax.fill_between(x, y - sem, y + sem, color="tab:blue", alpha=0.20, lw=0)
    avg_ax.axvline(0, color="black", lw=1.0, alpha=0.8)

    if "dur_s" in pulse_table.columns and pulse_table["dur_s"].notna().any():
        pulse_duration = float(pd.to_numeric(pulse_table["dur_s"], errors="coerce").median())
        avg_ax.axvspan(0, pulse_duration, color="tab:red", alpha=0.12, label="pulse duration")
    else:
        pulse_duration = None

    avg_ax.set_ylabel("Speed (mm/s)")
    title = (
        f"Pulse-triggered colony speed ({side}, "
        f"{len(pulse_table)} pulses, {len(result['track_table'])} tracks)"
    )
    if result["sort_by_stimulus_strength"]:
        direction = "ascending" if result["stimulus_sort_ascending"] else "descending"
        title += f", sorted by {result['stimulus_strength_col']} ({direction})"
    if smooth_seconds > 0:
        title += f", {smooth_seconds:g}s smoothing"
    avg_ax.set_title(title)
    if ylim is not None:
        avg_ax.set_ylim(*ylim)
    avg_ax.legend(fontsize=8)
    avg_ax.grid(True, alpha=0.25)

    time_left, time_right = _centered_time_extent(relative_time_s)
    extent = [time_left, time_right, matrix.shape[0] - 0.5, -0.5]
    im = matrix_ax.imshow(
        matrix,
        aspect="auto",
        interpolation="none",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    matrix_ax.axvline(0, color="white", lw=1.0, alpha=0.9)
    if pulse_duration is not None:
        matrix_ax.axvline(pulse_duration, color="white", lw=0.8, alpha=0.65, linestyle="--")
    matrix_ax.set_xlabel(f"Time from CSV_PULSE {result['frame_col']} (s)")
    matrix_ax.set_ylabel("Pulse")
    matrix_title = "Single-pulse responses, each averaged over ant speed tracks"
    if result["sort_by_stimulus_strength"]:
        matrix_title += f" (rows sorted by {result['stimulus_strength_col']})"
    matrix_ax.set_title(matrix_title)
    fig.colorbar(im, cax=colorbar_ax, label="Speed (mm/s)")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08)
    plt.show()
    return result


def plot_sleep_split_pulse_triggered_colony_speed(
    tracks: pd.DataFrame,
    pulses: pd.DataFrame,
    *,
    pre_seconds: float,
    post_seconds: float,
    side: str | None = "both",
    frame_col: str | None = "camFrameStart",
    smooth_seconds: float = 0.0,
    vmin: float | None = 0.0,
    vmax: float | None = None,
    vmax_percentile: float | None = 99.0,
    cmap: str = "viridis",
    ylim: tuple[float, float] | None = None,
    sort_by_stimulus_strength: bool = False,
    stimulus_strength_col: str | None = "auto",
    stimulus_sort_ascending: bool = True,
) -> dict[str, object]:
    result = build_sleep_split_pulse_triggered_colony_speed(
        tracks,
        pulses,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        side=side,
        frame_col=frame_col,
        smooth_seconds=smooth_seconds,
        sort_by_stimulus_strength=sort_by_stimulus_strength,
        stimulus_strength_col=stimulus_strength_col,
        stimulus_sort_ascending=stimulus_sort_ascending,
    )

    averages = result["average_by_state"]
    matrices = result["matrix_by_state"]
    pulse_table = result["pulse_table"]
    relative_time_s = result["relative_time_s"]
    state_info = result["state_info"]
    state_order = ["sleeping", "not_sleeping"]

    if vmax is None and vmax_percentile is not None:
        finite_arrays = [
            matrix[np.isfinite(matrix)]
            for matrix in matrices.values()
            if np.isfinite(matrix).any()
        ]
        if finite_arrays:
            finite_values = np.concatenate(finite_arrays)
            vmax = float(np.nanpercentile(finite_values, vmax_percentile))

    fig = plt.figure(figsize=(12, 10))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[2, 3, 3],
        width_ratios=[1, 0.035],
        hspace=0.10,
        wspace=0.04,
    )
    avg_ax = fig.add_subplot(grid[0, 0])
    matrix_axes = {
        "sleeping": fig.add_subplot(grid[1, 0], sharex=avg_ax),
        "not_sleeping": fig.add_subplot(grid[2, 0], sharex=avg_ax),
    }
    colorbar_ax = fig.add_subplot(grid[1:, 1])
    avg_ax.tick_params(labelbottom=False)
    matrix_axes["sleeping"].tick_params(labelbottom=False)

    for state_key in state_order:
        avg = averages[state_key]
        info = state_info[state_key]
        x = avg["relative_time_s"].to_numpy(dtype=float)
        y = avg["mean_speed_mm_s"].to_numpy(dtype=float)
        sem = avg["sem_speed_mm_s"].to_numpy(dtype=float)
        pulse_counts = pulse_table[f"n_{state_key}_tracks_at_pulse"].to_numpy(dtype=float)
        n_pulses = int(np.sum(pulse_counts > 0))
        median_tracks = float(np.nanmedian(pulse_counts[pulse_counts > 0])) if np.any(pulse_counts > 0) else 0.0
        label = f"{info['label']} ({n_pulses} pulses, median {median_tracks:g} ants)"
        avg_ax.plot(x, y, color=info["color"], lw=1.8, label=label)
        avg_ax.fill_between(x, y - sem, y + sem, color=info["color"], alpha=0.18, lw=0)

    avg_ax.axvline(0, color="black", lw=1.0, alpha=0.8)
    if "dur_s" in pulse_table.columns and pulse_table["dur_s"].notna().any():
        pulse_duration = float(pd.to_numeric(pulse_table["dur_s"], errors="coerce").median())
        avg_ax.axvspan(0, pulse_duration, color="tab:red", alpha=0.12, label="pulse duration")
    else:
        pulse_duration = None

    avg_ax.set_ylabel("Speed (mm/s)")
    title = (
        f"CSV-pulse speed by sleep state at {result['frame_col']} "
        f"({side}, {len(pulse_table)} pulses, {len(result['track_table'])} tracks)"
    )
    if result["sort_by_stimulus_strength"]:
        direction = "ascending" if result["stimulus_sort_ascending"] else "descending"
        title += f", sorted by {result['stimulus_strength_col']} ({direction})"
    if smooth_seconds > 0:
        title += f", {smooth_seconds:g}s smoothing"
    avg_ax.set_title(title)
    if ylim is not None:
        avg_ax.set_ylim(*ylim)
    avg_ax.legend(fontsize=8)
    avg_ax.grid(True, alpha=0.25)

    time_left, time_right = _centered_time_extent(relative_time_s)
    extent = [time_left, time_right, len(pulse_table) - 0.5, -0.5]
    im = None
    for state_key in state_order:
        ax = matrix_axes[state_key]
        info = state_info[state_key]
        im = ax.imshow(
            matrices[state_key],
            aspect="auto",
            interpolation="none",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.axvline(0, color="white", lw=1.0, alpha=0.9)
        if pulse_duration is not None:
            ax.axvline(pulse_duration, color="white", lw=0.8, alpha=0.65, linestyle="--")
        ax.set_ylabel("Pulse")
        matrix_title = f"Single-pulse responses: ants {info['label']}"
        if result["sort_by_stimulus_strength"]:
            matrix_title += f" (rows sorted by {result['stimulus_strength_col']})"
        ax.set_title(matrix_title)

    matrix_axes["not_sleeping"].set_xlabel(f"Time from CSV_PULSE {result['frame_col']} (s)")
    if im is not None:
        fig.colorbar(im, cax=colorbar_ax, label="Speed (mm/s)")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.07)
    plt.show()
    return result
