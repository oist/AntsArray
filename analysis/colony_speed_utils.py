"""Helpers for interactive colony speed-vector plotting."""

from __future__ import annotations

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
