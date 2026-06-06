"""Quick spontaneous-activity screen for motion-corrected 2P recordings.

The goal is not final ROI extraction. This script makes fast QC products that
answer: "Is there localized calcium-like activity worth annotating?"
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib
import numpy as np
import tifffile as tf
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_frame_rate(experiment_xml: Path | None) -> float | None:
    if experiment_xml is None or not experiment_xml.exists():
        return None
    root = ET.parse(experiment_xml).getroot()
    lsm = root.find("LSM")
    if lsm is None:
        return None
    value = lsm.attrib.get("frameRate")
    return float(value) if value is not None else None


def scale_to_uint8(image: np.ndarray, low_pct: float = 1, high_pct: float = 99.5) -> np.ndarray:
    low, high = np.nanpercentile(image, [low_pct, high_pct])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(image)), float(np.nanmax(image))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)
    return (scaled * 255).astype(np.uint8)


def load_binned_movie(movie_path: Path, spatial_downsample: int, temporal_bin: int) -> tuple[np.ndarray, tuple[int, int], np.dtype]:
    with tf.TiffFile(movie_path) as tif:
        total = len(tif.pages)
        first = tif.pages[0].asarray()
        raw_shape = first.shape
        raw_dtype = first.dtype
        out_h = raw_shape[0] // spatial_downsample
        out_w = raw_shape[1] // spatial_downsample
        n_bins = total // temporal_bin
        movie = np.empty((n_bins, out_h, out_w), dtype=np.float32)

        acc = np.zeros(raw_shape, dtype=np.float32)
        acc_n = 0
        out_i = 0
        for frame_i, page in enumerate(tif.pages[: n_bins * temporal_bin]):
            acc += page.asarray().astype(np.float32)
            acc_n += 1
            if acc_n == temporal_bin:
                averaged = acc / temporal_bin
                movie[out_i] = cv2.resize(averaged, (out_w, out_h), interpolation=cv2.INTER_AREA)
                acc.fill(0)
                acc_n = 0
                out_i += 1
            if frame_i % 500 == 0:
                print(f"loaded frame {frame_i}/{n_bins * temporal_bin}")
    return movie, raw_shape, raw_dtype


def compute_dff(movie: np.ndarray, baseline_pct: float) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.percentile(movie, baseline_pct, axis=0)
    floor = max(1.0, float(np.percentile(baseline, 5)))
    dff = (movie - baseline) / np.maximum(baseline, floor)
    return dff.astype(np.float32), baseline.astype(np.float32)


def local_correlation(movie: np.ndarray) -> np.ndarray:
    centered = movie - movie.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, keepdims=True)
    z = centered / np.maximum(std, 1e-6)
    h, w = movie.shape[1:]
    corr_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    for dy, dx in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        y_src = slice(max(0, -dy), min(h, h - dy))
        x_src = slice(max(0, -dx), min(w, w - dx))
        y_dst = slice(max(0, dy), min(h, h + dy))
        x_dst = slice(max(0, dx), min(w, w + dx))
        corr = np.mean(z[:, y_src, x_src] * z[:, y_dst, x_dst], axis=0)
        corr_sum[y_src, x_src] += corr
        count[y_src, x_src] += 1

    return corr_sum / np.maximum(count, 1)


def find_candidate_peaks(
    score: np.ndarray,
    max_candidates: int,
    min_distance: int,
    percentile: float,
    valid_mask: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    smooth = ndimage.gaussian_filter(score, 1.0)
    if valid_mask is not None:
        smooth = smooth.copy()
        smooth[~valid_mask] = 0
        pool = smooth[valid_mask & np.isfinite(smooth)]
    else:
        pool = smooth[np.isfinite(smooth)]
    threshold = np.percentile(pool, percentile) if pool.size else np.inf
    maxima = smooth == ndimage.maximum_filter(smooth, size=min_distance * 2 + 1)
    keep = maxima & (smooth >= threshold)
    if valid_mask is not None:
        keep &= valid_mask
    coords = np.argwhere(keep)
    peaks = [(int(y), int(x), float(smooth[y, x])) for y, x in coords]
    peaks.sort(key=lambda p: p[2], reverse=True)

    selected: list[tuple[int, int, float]] = []
    for y, x, val in peaks:
        if all((y - yy) ** 2 + (x - xx) ** 2 >= min_distance**2 for yy, xx, _ in selected):
            selected.append((y, x, val))
        if len(selected) >= max_candidates:
            break
    return selected


def aperture_trace(movie: np.ndarray, y: int, x: int, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[: movie.shape[1], : movie.shape[2]]
    mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
    return movie[:, mask].mean(axis=1)


def robust_event_count(trace: np.ndarray, min_dff: float) -> tuple[int, float]:
    med = float(np.median(trace))
    mad = float(np.median(np.abs(trace - med)))
    sigma = max(1e-6, 1.4826 * mad)
    threshold = max(min_dff, med + 3.0 * sigma)
    active = trace > threshold
    labeled, n_labels = ndimage.label(active)
    return int(n_labels), float(threshold)


def save_overlay(path: Path, mean_image: np.ndarray, peaks: list[tuple[int, int, float]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(mean_image, cmap="gray")
    for i, (y, x, _) in enumerate(peaks, start=1):
        ax.plot(x, y, "o", ms=8, markerfacecolor="none", markeredgecolor="tab:red", markeredgewidth=1.5)
        ax.text(x + 2, y + 2, str(i), color="yellow", fontsize=8)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_trace_plot(
    path: Path,
    traces: list[np.ndarray],
    frame_rate_hz: float | None,
    temporal_bin: int,
) -> None:
    if not traces:
        return
    dt = temporal_bin / frame_rate_hz if frame_rate_hz else 1.0
    x = np.arange(len(traces[0])) * dt
    fig, ax = plt.subplots(figsize=(12, 7))
    offset = 0.0
    for i, trace in enumerate(traces, start=1):
        centered = trace - np.percentile(trace, 10)
        ax.plot(x, centered + offset, lw=0.8)
        ax.text(x[-1] + dt, offset, str(i), va="center", fontsize=8)
        offset += max(0.05, float(np.percentile(centered, 99) - np.percentile(centered, 1)) * 1.2)
    ax.set_xlabel("time (s)" if frame_rate_hz else "binned frame")
    ax.set_ylabel("candidate traces, offset")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_map_grid(path: Path, maps: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (title, image) in zip(axes.ravel(), maps.items()):
        im = ax.imshow(image, cmap="magma")
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes.ravel()[len(maps) :]:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(
    path: Path,
    movie_path: Path,
    n_bins: int,
    raw_shape: tuple[int, int],
    frame_rate_hz: float | None,
    temporal_bin: int,
    spatial_downsample: int,
    summary: dict[str, float | int],
    files: dict[str, str],
) -> None:
    links = "\n".join(f'<li><a href="{html.escape(v)}">{html.escape(k)}</a></li>' for k, v in files.items())
    effective_hz = frame_rate_hz / temporal_bin if frame_rate_hz else None
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Spontaneous activity quick screen</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1c1c1c; }}
    img {{ max-width: min(1100px, 100%); height: auto; display: block; margin: 14px 0 26px; }}
    code {{ background: #eee; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Spontaneous activity quick screen</h1>
  <p><b>Movie:</b> <code>{html.escape(str(movie_path))}</code></p>
  <p><b>Raw shape:</b> {raw_shape[0]} x {raw_shape[1]}; <b>binned frames:</b> {n_bins}; <b>effective rate:</b> {effective_hz or "unknown"} Hz</p>
  <p><b>Spatial downsample:</b> {spatial_downsample}x; <b>temporal bin:</b> {temporal_bin} frames</p>
  <p><b>Candidate count:</b> {summary["candidate_count"]}; <b>median candidate events:</b> {summary["median_candidate_events"]:.1f}; <b>max residual dF/F:</b> {summary["max_residual_dff"]:.3f}</p>
  <h2>Outputs</h2>
  <ul>{links}</ul>
  <h2>Candidate locations</h2>
  <img src="{html.escape(files["Candidate overlay"])}" alt="Candidate overlay">
  <h2>Candidate traces</h2>
  <img src="{html.escape(files["Candidate traces"])}" alt="Candidate traces">
  <h2>Activity maps</h2>
  <img src="{html.escape(files["Activity map grid"])}" alt="Activity maps">
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="spontaneous_qc")
    parser.add_argument("--experiment-xml", type=Path)
    parser.add_argument("--spatial-downsample", type=int, default=4)
    parser.add_argument("--temporal-bin", type=int, default=3)
    parser.add_argument("--baseline-percentile", type=float, default=20)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--candidate-percentile", type=float, default=99.2)
    parser.add_argument("--min-distance", type=int, default=4)
    parser.add_argument("--aperture-radius", type=int, default=3)
    parser.add_argument("--min-event-dff", type=float, default=0.03)
    parser.add_argument("--exclude-border", type=int, default=8, help="Exclude this many downsampled pixels at each border.")
    parser.add_argument("--tissue-percentile", type=float, default=30, help="Keep pixels above this mean-intensity percentile.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_rate_hz = read_frame_rate(args.experiment_xml)
    movie, raw_shape, raw_dtype = load_binned_movie(args.movie, args.spatial_downsample, args.temporal_bin)
    dff, baseline = compute_dff(movie, args.baseline_percentile)
    global_trace = np.median(dff.reshape(dff.shape[0], -1), axis=1)
    residual = dff - global_trace[:, None, None]

    mean_image = movie.mean(axis=0)
    std_dff = dff.std(axis=0)
    std_residual = residual.std(axis=0)
    p99_residual = np.percentile(residual, 99, axis=0)
    p95_minus_p20 = np.percentile(residual, 95, axis=0) - np.percentile(residual, 20, axis=0)
    local_corr = local_correlation(residual)

    med = np.median(residual, axis=0)
    mad = np.median(np.abs(residual - med), axis=0)
    sigma = np.maximum(1e-6, 1.4826 * mad)
    event_mask = residual > np.maximum(args.min_event_dff, med + 3.0 * sigma)
    event_fraction = event_mask.mean(axis=0)

    score = (
        0.35 * (std_residual / np.nanpercentile(std_residual, 99.5))
        + 0.30 * np.clip(local_corr, 0, None) / max(1e-6, np.nanpercentile(np.clip(local_corr, 0, None), 99.5))
        + 0.20 * (p95_minus_p20 / np.nanpercentile(p95_minus_p20, 99.5))
        + 0.15 * (event_fraction / max(1e-6, np.nanpercentile(event_fraction, 99.5)))
    )
    score = np.nan_to_num(score, nan=0, posinf=0, neginf=0)
    valid_mask = mean_image > np.percentile(mean_image, args.tissue_percentile)
    if args.exclude_border > 0:
        valid_mask[: args.exclude_border, :] = False
        valid_mask[-args.exclude_border :, :] = False
        valid_mask[:, : args.exclude_border] = False
        valid_mask[:, -args.exclude_border :] = False

    peaks = find_candidate_peaks(score, args.max_candidates, args.min_distance, args.candidate_percentile, valid_mask)
    traces = [aperture_trace(residual, y, x, args.aperture_radius) for y, x, _ in peaks]

    candidates_csv = args.output_dir / f"{args.prefix}_candidate_traces.csv"
    event_counts: list[int] = []
    with candidates_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["binned_frame", "time_s", "global_dff"] + [f"candidate_{i}" for i in range(1, len(traces) + 1)]
        writer.writerow(header)
        dt = args.temporal_bin / frame_rate_hz if frame_rate_hz else math.nan
        for i in range(movie.shape[0]):
            writer.writerow(
                [i, i * dt if frame_rate_hz else "", float(global_trace[i])]
                + [float(trace[i]) for trace in traces]
            )

    candidates_summary_csv = args.output_dir / f"{args.prefix}_candidates.csv"
    with candidates_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "y_downsampled",
                "x_downsampled",
                "y_raw_px",
                "x_raw_px",
                "score",
                "events",
                "event_threshold_dff",
                "max_residual_dff",
                "std_residual_dff",
            ],
        )
        writer.writeheader()
        for i, ((y, x, val), trace) in enumerate(zip(peaks, traces), start=1):
            events, threshold = robust_event_count(trace, args.min_event_dff)
            event_counts.append(events)
            writer.writerow(
                {
                    "candidate": i,
                    "y_downsampled": y,
                    "x_downsampled": x,
                    "y_raw_px": y * args.spatial_downsample,
                    "x_raw_px": x * args.spatial_downsample,
                    "score": val,
                    "events": events,
                    "event_threshold_dff": threshold,
                    "max_residual_dff": float(np.max(trace)),
                    "std_residual_dff": float(np.std(trace)),
                }
            )

    mean_png = args.output_dir / f"{args.prefix}_mean.png"
    maps_png = args.output_dir / f"{args.prefix}_activity_maps.png"
    overlay_png = args.output_dir / f"{args.prefix}_candidate_overlay.png"
    traces_png = args.output_dir / f"{args.prefix}_candidate_traces.png"
    global_png = args.output_dir / f"{args.prefix}_global_trace.png"
    report_html = args.output_dir / f"{args.prefix}_report.html"
    metadata_json = args.output_dir / f"{args.prefix}_metadata.json"

    iio.imwrite(mean_png, scale_to_uint8(mean_image))
    save_map_grid(
        maps_png,
        {
            "std dF/F": std_dff,
            "std residual dF/F": std_residual,
            "99th pct residual": p99_residual,
            "p95 - p20 residual": p95_minus_p20,
            "local correlation": local_corr,
            "event fraction": event_fraction,
        },
    )
    save_overlay(overlay_png, scale_to_uint8(mean_image), peaks)
    save_trace_plot(traces_png, traces, frame_rate_hz, args.temporal_bin)

    dt = args.temporal_bin / frame_rate_hz if frame_rate_hz else 1.0
    x = np.arange(len(global_trace)) * dt
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, global_trace, lw=0.8)
    ax.set_xlabel("time (s)" if frame_rate_hz else "binned frame")
    ax.set_ylabel("global median dF/F")
    fig.tight_layout()
    fig.savefig(global_png, dpi=160)
    plt.close(fig)

    summary = {
        "candidate_count": len(peaks),
        "median_candidate_events": float(np.median(event_counts)) if event_counts else 0.0,
        "max_residual_dff": float(np.max([np.max(trace) for trace in traces])) if traces else float(np.max(residual)),
        "raw_dtype": str(raw_dtype),
        "binned_frames": int(movie.shape[0]),
        "frame_rate_hz": frame_rate_hz or "",
    }
    files = {
        "Mean image": mean_png.name,
        "Activity map grid": maps_png.name,
        "Candidate overlay": overlay_png.name,
        "Candidate traces": traces_png.name,
        "Global trace": global_png.name,
        "Candidate summary CSV": candidates_summary_csv.name,
        "Candidate trace CSV": candidates_csv.name,
        "Metadata JSON": metadata_json.name,
    }
    write_report(
        report_html,
        args.movie,
        movie.shape[0],
        raw_shape,
        frame_rate_hz,
        args.temporal_bin,
        args.spatial_downsample,
        summary,
        files,
    )
    metadata_json.write_text(
        json.dumps(
            {
                "movie": str(args.movie),
                "experiment_xml": str(args.experiment_xml) if args.experiment_xml else None,
                "parameters": vars(args),
                "summary": summary,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"Loaded binned movie: {movie.shape}")
    print(f"Frame rate: {frame_rate_hz or 'unknown'} Hz")
    print(f"Candidates: {len(peaks)}")
    print(f"Median candidate events: {summary['median_candidate_events']:.1f}")
    print(f"Max residual dF/F among candidates: {summary['max_residual_dff']:.3f}")
    print(f"Wrote report: {report_html}")


if __name__ == "__main__":
    main()
