"""Stabilize a 2-photon time-series TIFF and estimate XYZ motion.

This follows the practical core of CaImAn/NoRMCorre rigid correction:
build a template, estimate FFT phase-correlation shifts for each frame, and
apply the shifts to the raw movie. If a z-stack is provided, each corrected
frame is also matched to the calibrated stack to estimate axial drift.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib
import numpy as np
import tifffile as tf
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def natural_key(path: Path) -> tuple:
    nums = tuple(int(part) for part in re.findall(r"\d+", path.stem))
    return (*nums, path.name)


def read_thor_xml(path: Path) -> dict[str, float | int | str | None]:
    meta: dict[str, float | int | str | None] = {
        "pixel_size_um": None,
        "frame_rate_hz": None,
        "timepoints": None,
    }
    if not path.exists():
        return meta
    root = ET.parse(path).getroot()
    lsm = root.find("LSM")
    if lsm is not None:
        for key, attr in (("pixel_size_um", "pixelSizeUM"), ("frame_rate_hz", "frameRate")):
            value = lsm.attrib.get(attr)
            if value is not None:
                meta[key] = float(value)
    timelapse = root.find("Timelapse")
    if timelapse is not None and timelapse.attrib.get("timepoints") is not None:
        meta["timepoints"] = int(timelapse.attrib["timepoints"])
    return meta


def ome_physical_sizes(tif_path: Path) -> dict[str, float | None]:
    sizes: dict[str, float | None] = {"x": None, "y": None, "z": None}
    with tf.TiffFile(tif_path) as tif:
        description = tif.pages[0].tags.get("ImageDescription")
        if description is None:
            return sizes
        text = str(description.value)
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return sizes
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2010-06"}
    pixels = root.find(".//ome:Pixels", ns)
    if pixels is None:
        return sizes
    for axis, attr in (("x", "PhysicalSizeX"), ("y", "PhysicalSizeY"), ("z", "PhysicalSizeZ")):
        value = pixels.attrib.get(attr)
        if value is not None:
            try:
                sizes[axis] = float(value)
            except ValueError:
                pass
    return sizes


def find_recording_tiff(input_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    candidates = [p for p in candidates if "preview" not in p.name.lower() and p.name.lower() != "image.tif"]
    if not candidates:
        raise FileNotFoundError(f"No recording TIFF found in {input_dir}")
    return max(candidates, key=lambda p: p.stat().st_size)


def sorted_zstack_tiffs(zstack_dir: Path) -> list[Path]:
    paths = sorted(zstack_dir.glob("*.tif"), key=natural_key) + sorted(zstack_dir.glob("*.tiff"), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No z-stack TIFFs found in {zstack_dir}")
    return paths


def load_zstack(zstack_dir: Path) -> tuple[np.ndarray, dict[str, float | None]]:
    paths = sorted_zstack_tiffs(zstack_dir)
    first = np.squeeze(tf.imread(paths[0]))
    sizes = ome_physical_sizes(paths[0])
    if first.ndim == 3:
        return first, sizes
    if first.ndim != 2:
        raise ValueError(f"Expected 2D or 3D z-stack TIFF, got {first.shape}")
    planes = [first]
    for path in paths[1:]:
        plane = np.squeeze(tf.imread(path))
        if plane.ndim != 2:
            raise ValueError(f"Expected 2D z plane in {path}, got {plane.shape}")
        planes.append(plane)
    return np.stack(planes, axis=0), sizes


def registration_image(frame: np.ndarray, low_pct: float, high_pct: float, highpass_sigma: float) -> np.ndarray:
    image = frame.astype(np.float32, copy=False)
    low, high = np.percentile(image, [low_pct, high_pct])
    if high > low:
        image = np.clip(image, low, high)
    if highpass_sigma > 0:
        image = image - cv2.GaussianBlur(image, (0, 0), highpass_sigma)
    image = image - float(np.mean(image))
    std = float(np.std(image))
    if std > 0:
        image = image / std
    return np.ascontiguousarray(image.astype(np.float32, copy=False))


def estimate_shift_to_template(frame_reg: np.ndarray, template_reg: np.ndarray, hann: np.ndarray) -> tuple[float, float, float]:
    (dx, dy), response = cv2.phaseCorrelate(frame_reg, template_reg, hann)
    return float(dx), float(dy), float(response)


def shift_frame(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        frame,
        matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float32, copy=False).ravel()
    bb = b.astype(np.float32, copy=False).ravel()
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom == 0:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def iter_sample_indices(total: int, sample_stride: int, max_samples: int) -> list[int]:
    indices = list(range(0, total, max(1, sample_stride)))
    if len(indices) > max_samples:
        pick = np.linspace(0, len(indices) - 1, max_samples).round().astype(int)
        indices = [indices[i] for i in pick]
    return indices


def build_template(
    tif_path: Path,
    sample_stride: int,
    max_samples: int,
    template_iters: int,
    low_pct: float,
    high_pct: float,
    highpass_sigma: float,
) -> np.ndarray:
    with tf.TiffFile(tif_path) as tif:
        total = len(tif.pages)
        indices = iter_sample_indices(total, sample_stride, max_samples)
        frames = [tif.pages[i].asarray() for i in indices]

    template = np.mean(np.stack(frames).astype(np.float32), axis=0)
    hann = cv2.createHanningWindow((template.shape[1], template.shape[0]), cv2.CV_32F)
    for _ in range(template_iters):
        template_reg = registration_image(template, low_pct, high_pct, highpass_sigma)
        aligned = []
        for frame in frames:
            frame_reg = registration_image(frame, low_pct, high_pct, highpass_sigma)
            dx, dy, _ = estimate_shift_to_template(frame_reg, template_reg, hann)
            aligned.append(shift_frame(frame, dx, dy).astype(np.float32))
        template = np.mean(np.stack(aligned), axis=0)
    return template


def center_crop_or_pad(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    out = image
    if out.shape[0] < target_h or out.shape[1] < target_w:
        pad_h = max(0, target_h - out.shape[0])
        pad_w = max(0, target_w - out.shape[1])
        out = np.pad(
            out,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode="edge",
        )
    y0 = max(0, (out.shape[0] - target_h) // 2)
    x0 = max(0, (out.shape[1] - target_w) // 2)
    return out[y0 : y0 + target_h, x0 : x0 + target_w]


def prepare_z_references(
    zstack_dir: Path,
    output_shape: tuple[int, int],
    rec_pixel_um: float | None,
    template: np.ndarray,
    low_pct: float,
    high_pct: float,
    highpass_sigma: float,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | None]]:
    zstack, sizes = load_zstack(zstack_dir)
    z_pixel_um = sizes.get("x") or sizes.get("y")
    template_reg = registration_image(template, low_pct, high_pct, highpass_sigma)
    hann = cv2.createHanningWindow((template.shape[1], template.shape[0]), cv2.CV_32F)

    refs = []
    for plane in zstack:
        plane_f = plane.astype(np.float32)
        if rec_pixel_um and z_pixel_um:
            scale = z_pixel_um / rec_pixel_um
            if not math.isclose(scale, 1.0, rel_tol=0.03, abs_tol=0.03):
                plane_f = cv2.resize(plane_f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        plane_f = center_crop_or_pad(plane_f, output_shape)
        plane_reg = registration_image(plane_f, low_pct, high_pct, highpass_sigma)
        dx, dy, _ = estimate_shift_to_template(plane_reg, template_reg, hann)
        plane_aligned = shift_frame(plane_f, dx, dy)
        small = cv2.resize(
            plane_aligned,
            (output_shape[1] // downsample, output_shape[0] // downsample),
            interpolation=cv2.INTER_AREA,
        )
        small_reg = registration_image(small, low_pct, high_pct, highpass_sigma=0)
        refs.append(small_reg.ravel())

    ref_matrix = np.stack(refs).astype(np.float32)
    norms = np.linalg.norm(ref_matrix, axis=1)
    norms[norms == 0] = 1
    ref_matrix = ref_matrix / norms[:, None]
    z_um = np.arange(zstack.shape[0], dtype=np.float32) * float(sizes.get("z") or 1.0)
    return ref_matrix, z_um, sizes


def estimate_z(frame: np.ndarray, refs: np.ndarray, downsample: int, low_pct: float, high_pct: float) -> tuple[int, float]:
    small = cv2.resize(
        frame.astype(np.float32),
        (frame.shape[1] // downsample, frame.shape[0] // downsample),
        interpolation=cv2.INTER_AREA,
    )
    reg = registration_image(small, low_pct, high_pct, highpass_sigma=0).ravel()
    norm = float(np.linalg.norm(reg))
    if norm == 0:
        return -1, float("nan")
    scores = refs @ (reg / norm)
    idx = int(np.argmax(scores))
    return idx, float(scores[idx])


def scale_to_uint8(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        high = low + 1
    scaled = np.clip((frame.astype(np.float32) - low) / (high - low), 0, 1)
    return (scaled * 255).astype(np.uint8)


def add_label(image: np.ndarray, label: str) -> Image.Image:
    out = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((6, 6, 170, 28), fill=(0, 0, 0))
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)
    return out


def write_qc_gif(path: Path, pairs: list[tuple[int, np.ndarray, np.ndarray]], low: float, high: float) -> None:
    frames: list[Image.Image] = []
    for idx, raw, corrected in pairs:
        raw_u8 = scale_to_uint8(raw, low, high)
        cor_u8 = scale_to_uint8(corrected, low, high)
        combined = np.concatenate([raw_u8, cor_u8], axis=1)
        image = add_label(combined, f"frame {idx}: raw | XY corrected")
        image.thumbnail((900, 450), Image.Resampling.LANCZOS)
        frames.append(image)
    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)


def write_plots(path: Path, rows: list[dict[str, float | int | str]], frame_rate_hz: float | None) -> None:
    frames = np.array([int(row["frame"]) for row in rows])
    time = frames / frame_rate_hz if frame_rate_hz else frames
    xlabel = "time (s)" if frame_rate_hz else "frame"

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time, [row["dx_px"] for row in rows], lw=0.8, label="dx")
    axes[0].plot(time, [row["dy_px"] for row in rows], lw=0.8, label="dy")
    axes[0].set_ylabel("XY shift (px)")
    axes[0].legend(loc="upper right")
    axes[1].plot(time, [row["corr_before"] for row in rows], lw=0.8, label="before")
    axes[1].plot(time, [row["corr_after"] for row in rows], lw=0.8, label="after")
    axes[1].set_ylabel("template corr.")
    axes[1].legend(loc="lower right")
    axes[2].plot(time, [row["z_um"] if row["z_um"] != "" else np.nan for row in rows], lw=0.8, marker=".", ms=2)
    axes[2].set_ylabel("estimated z (um)")
    axes[3].plot(time, [row["z_corr"] if row["z_corr"] != "" else np.nan for row in rows], lw=0.8)
    axes[3].set_ylabel("z corr.")
    axes[3].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(
    path: Path,
    recording: Path,
    zstack: Path | None,
    total_frames: int,
    shape: tuple[int, int],
    rec_meta: dict[str, float | int | str | None],
    z_meta: dict[str, float | None],
    files: dict[str, str],
    summary: dict[str, float],
) -> None:
    links = "\n".join(f'<li><a href="{html.escape(v)}">{html.escape(k)}</a></li>' for k, v in files.items())
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>rec1 motion stabilization QC</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1c1c1c; }}
    img {{ max-width: min(1000px, 100%); height: auto; display: block; margin: 14px 0 26px; background: #000; }}
    code {{ background: #eee; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>rec1 motion stabilization QC</h1>
  <p><b>Recording:</b> <code>{html.escape(str(recording))}</code></p>
  <p><b>Z-stack:</b> <code>{html.escape(str(zstack)) if zstack else "not used"}</code></p>
  <p><b>Frames:</b> {total_frames}; <b>shape:</b> {shape[0]} x {shape[1]}; <b>frame rate:</b> {rec_meta.get("frame_rate_hz") or "unknown"} Hz</p>
  <p><b>Recording pixel size:</b> {rec_meta.get("pixel_size_um") or "unknown"} um; <b>z-stack voxel:</b> {z_meta.get("x") or "unknown"} x {z_meta.get("y") or "unknown"} x {z_meta.get("z") or "unknown"} um</p>
  <p><b>Median template correlation:</b> before {summary["median_corr_before"]:.3f}, after {summary["median_corr_after"]:.3f}</p>
  <p><b>XY shift range:</b> dx {summary["min_dx"]:.2f}..{summary["max_dx"]:.2f} px, dy {summary["min_dy"]:.2f}..{summary["max_dy"]:.2f} px; <b>accepted shifts:</b> {summary["accepted_fraction"]:.1%}</p>
  <h2>Outputs</h2>
  <ul>{links}</ul>
  <h2>Motion traces</h2>
  <img src="{html.escape(files["Motion trace plot"])}" alt="Motion trace plot">
  <h2>Mean projections</h2>
  <img src="{html.escape(files["Mean before/after"])}" alt="Mean before and after motion correction">
  <h2>Preview movie</h2>
  <img src="{html.escape(files["QC GIF"])}" alt="Raw and corrected preview movie">
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--recording-tif", type=Path)
    parser.add_argument("--zstack-dir", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="rec1")
    parser.add_argument("--sample-stride", type=int, default=25)
    parser.add_argument("--max-template-samples", type=int, default=250)
    parser.add_argument("--template-iters", type=int, default=2)
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.7)
    parser.add_argument("--highpass-sigma", type=float, default=0.0)
    parser.add_argument("--max-shift-px", type=float, default=60.0)
    parser.add_argument("--online-template-alpha", type=float, default=0.0)
    parser.add_argument("--min-phase-response", type=float, default=0.0)
    parser.add_argument("--min-corr-improvement", type=float, default=-1.0)
    parser.add_argument("--z-every", type=int, default=5)
    parser.add_argument("--z-downsample", type=int, default=4)
    parser.add_argument("--qc-every", type=int, default=100)
    parser.add_argument("--no-save-movie", action="store_true")
    parser.add_argument("--max-frames", type=int, help="Process only the first N frames for testing/QC.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recording = find_recording_tiff(args.input_dir, args.recording_tif)
    rec_meta = read_thor_xml(args.input_dir / "Experiment.xml")

    with tf.TiffFile(recording) as tif:
        total_frames = len(tif.pages)
        first = tif.pages[0].asarray()
        shape = first.shape
        dtype = first.dtype
        preview_low, preview_high = np.percentile(first, [args.low_percentile, args.high_percentile])
    frames_to_process = min(total_frames, args.max_frames) if args.max_frames else total_frames

    template = build_template(
        recording,
        args.sample_stride,
        args.max_template_samples,
        args.template_iters,
        args.low_percentile,
        args.high_percentile,
        args.highpass_sigma,
    )
    template_reg = registration_image(template, args.low_percentile, args.high_percentile, args.highpass_sigma)
    hann = cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)

    z_refs = None
    z_um = None
    z_meta: dict[str, float | None] = {"x": None, "y": None, "z": None}
    if args.zstack_dir is not None:
        z_refs, z_um, z_meta = prepare_z_references(
            args.zstack_dir,
            shape,
            rec_meta.get("pixel_size_um") if isinstance(rec_meta.get("pixel_size_um"), float) else None,
            template,
            args.low_percentile,
            args.high_percentile,
            args.highpass_sigma,
            args.z_downsample,
        )

    corrected_tif = args.output_dir / f"{args.prefix}_xy_corrected.bigtif"
    motion_csv = args.output_dir / f"{args.prefix}_motion_xyz.csv"
    template_png = args.output_dir / f"{args.prefix}_template.png"
    mean_png = args.output_dir / f"{args.prefix}_mean_before_after.png"
    plot_png = args.output_dir / f"{args.prefix}_motion_traces.png"
    qc_gif = args.output_dir / f"{args.prefix}_raw_vs_corrected_preview.gif"
    report_html = args.output_dir / f"{args.prefix}_motion_report.html"
    meta_json = args.output_dir / f"{args.prefix}_motion_metadata.json"

    template_u8 = scale_to_uint8(template, preview_low, preview_high)
    iio.imwrite(template_png, template_u8)

    rows: list[dict[str, float | int | str]] = []
    qc_pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    mean_raw = np.zeros(shape, dtype=np.float64)
    mean_corrected = np.zeros(shape, dtype=np.float64)

    writer = None if args.no_save_movie else tf.TiffWriter(corrected_tif, bigtiff=True)
    try:
        with tf.TiffFile(recording) as tif:
            for idx, page in enumerate(tif.pages):
                if idx >= frames_to_process:
                    break
                raw = page.asarray()
                frame_reg = registration_image(raw, args.low_percentile, args.high_percentile, args.highpass_sigma)
                dx, dy, response = estimate_shift_to_template(frame_reg, template_reg, hann)
                accepted = True
                if abs(dx) > args.max_shift_px or abs(dy) > args.max_shift_px:
                    dx, dy = 0.0, 0.0
                    response = float("nan")
                    accepted = False
                corrected = shift_frame(raw, dx, dy).astype(dtype, copy=False)

                corr_before = corrcoef(frame_reg, template_reg)
                corr_after = corrcoef(
                    registration_image(corrected, args.low_percentile, args.high_percentile, args.highpass_sigma),
                    template_reg,
                )
                if response < args.min_phase_response or corr_after < corr_before + args.min_corr_improvement:
                    dx, dy = 0.0, 0.0
                    corrected = raw
                    corr_after = corr_before
                    accepted = False

                z_index: int | str = ""
                z_value_um: float | str = ""
                z_corr: float | str = ""
                if z_refs is not None and z_um is not None and idx % max(1, args.z_every) == 0:
                    z_index_i, z_corr_f = estimate_z(corrected, z_refs, args.z_downsample, args.low_percentile, args.high_percentile)
                    z_index = z_index_i
                    z_value_um = float(z_um[z_index_i]) if z_index_i >= 0 else ""
                    z_corr = z_corr_f

                rows.append(
                    {
                        "frame": idx,
                        "time_s": idx / float(rec_meta["frame_rate_hz"]) if rec_meta.get("frame_rate_hz") else "",
                        "dx_px": dx,
                        "dy_px": dy,
                        "dx_um": dx * float(rec_meta["pixel_size_um"]) if rec_meta.get("pixel_size_um") else "",
                        "dy_um": dy * float(rec_meta["pixel_size_um"]) if rec_meta.get("pixel_size_um") else "",
                        "phase_response": response,
                        "corr_before": corr_before,
                        "corr_after": corr_after,
                        "z_index": z_index,
                        "z_um": z_value_um,
                        "z_corr": z_corr,
                        "accepted_xy_shift": int(accepted),
                    }
                )

                if writer is not None:
                    writer.write(corrected, photometric="minisblack", contiguous=True)
                if args.online_template_alpha > 0 and accepted:
                    alpha = min(max(args.online_template_alpha, 0.0), 1.0)
                    template = (1.0 - alpha) * template + alpha * corrected.astype(np.float32)
                    template_reg = registration_image(template, args.low_percentile, args.high_percentile, args.highpass_sigma)
                mean_raw += raw
                mean_corrected += corrected
                if idx % max(1, args.qc_every) == 0:
                    qc_pairs.append((idx, raw.copy(), corrected.copy()))
                if idx % 250 == 0:
                    print(f"processed {idx}/{frames_to_process}")
    finally:
        if writer is not None:
            writer.close()

    fieldnames = list(rows[0].keys())
    with motion_csv.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    mean_raw /= frames_to_process
    mean_corrected /= frames_to_process
    before_u8 = scale_to_uint8(mean_raw, np.percentile(mean_raw, 1), np.percentile(mean_raw, 99.7))
    after_u8 = scale_to_uint8(mean_corrected, np.percentile(mean_corrected, 1), np.percentile(mean_corrected, 99.7))
    iio.imwrite(mean_png, np.concatenate([before_u8, after_u8], axis=1))
    write_qc_gif(qc_gif, qc_pairs, preview_low, preview_high)
    write_plots(plot_png, rows, rec_meta.get("frame_rate_hz") if isinstance(rec_meta.get("frame_rate_hz"), float) else None)

    corr_before_vals = np.array([float(row["corr_before"]) for row in rows])
    corr_after_vals = np.array([float(row["corr_after"]) for row in rows])
    dx_vals = np.array([float(row["dx_px"]) for row in rows])
    dy_vals = np.array([float(row["dy_px"]) for row in rows])
    summary = {
        "median_corr_before": float(np.nanmedian(corr_before_vals)),
        "median_corr_after": float(np.nanmedian(corr_after_vals)),
        "min_dx": float(np.nanmin(dx_vals)),
        "max_dx": float(np.nanmax(dx_vals)),
        "min_dy": float(np.nanmin(dy_vals)),
        "max_dy": float(np.nanmax(dy_vals)),
        "accepted_fraction": float(np.mean([int(row["accepted_xy_shift"]) for row in rows])),
    }

    files = {
        "XY-corrected BigTIFF": corrected_tif.name if not args.no_save_movie else "",
        "Motion CSV": motion_csv.name,
        "Template PNG": template_png.name,
        "Mean before/after": mean_png.name,
        "Motion trace plot": plot_png.name,
        "QC GIF": qc_gif.name,
        "Metadata JSON": meta_json.name,
    }
    files = {k: v for k, v in files.items() if v}
    write_report(report_html, recording, args.zstack_dir, frames_to_process, shape, rec_meta, z_meta, files, summary)
    meta_json.write_text(
        json.dumps(
            {
                "recording": str(recording),
                "zstack_dir": str(args.zstack_dir) if args.zstack_dir else None,
                "total_frames": total_frames,
                "processed_frames": frames_to_process,
                "shape_yx": shape,
                "dtype": str(dtype),
                "recording_metadata": rec_meta,
                "zstack_metadata": z_meta,
                "parameters": vars(args) | {"input_dir": str(args.input_dir), "output_dir": str(args.output_dir)},
                "summary": summary,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"Recording: {recording}")
    print(f"Frames: {frames_to_process}/{total_frames}, shape: {shape}, dtype: {dtype}")
    print(f"Median corr before/after: {summary['median_corr_before']:.3f}/{summary['median_corr_after']:.3f}")
    print(f"XY shift range dx: {summary['min_dx']:.2f}..{summary['max_dx']:.2f}, dy: {summary['min_dy']:.2f}..{summary['max_dy']:.2f}")
    for label, rel in files.items():
        print(f"Wrote {label}: {args.output_dir / rel}")
    print(f"Wrote report: {report_html}")


if __name__ == "__main__":
    main()
