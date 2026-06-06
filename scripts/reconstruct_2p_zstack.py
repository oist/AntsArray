"""Build quick 3D reconstruction products from a 2-photon TIFF z-stack.

Outputs:
  - a single OME-TIFF volume
  - XY/XZ/YZ maximum-intensity projection PNGs
  - a simple rotating maximum-intensity projection GIF
  - an HTML summary report
"""

from __future__ import annotations

import argparse
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile as tf
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


def sorted_tiffs(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if not paths:
        raise FileNotFoundError(f"No .tif/.tiff files found in {input_dir}")

    def key(path: Path) -> tuple:
        nums = tuple(int(part) for part in re.findall(r"\d+", path.stem))
        return (*nums, path.name)

    return sorted(paths, key=key)


def physical_sizes(first_tif: Path) -> dict[str, float | None]:
    sizes: dict[str, float | None] = {"x": None, "y": None, "z": None}
    with tf.TiffFile(first_tif) as tif:
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


def load_volume(paths: list[Path]) -> np.ndarray:
    first = tf.imread(paths[0])
    squeezed = np.squeeze(first)
    if squeezed.ndim == 3:
        return squeezed
    if squeezed.ndim != 2:
        raise ValueError(f"Expected 2D plane or 3D volume, got shape {squeezed.shape}")

    planes = [squeezed]
    for path in paths[1:]:
        plane = np.squeeze(tf.imread(path))
        if plane.ndim != 2:
            raise ValueError(f"Expected 2D plane in {path}, got shape {plane.shape}")
        planes.append(plane)
    return np.stack(planes, axis=0)


def scale_to_uint8(volume: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    low, high = np.percentile(volume, [low_pct, high_pct])
    if high <= low:
        low, high = float(volume.min()), float(volume.max())
    if high <= low:
        return np.zeros(volume.shape, dtype=np.uint8)
    scaled = np.clip((volume.astype(np.float32) - low) / (high - low), 0, 1)
    return (scaled * 255).astype(np.uint8)


def save_projection(path: Path, image: np.ndarray) -> None:
    iio.imwrite(path, image.astype(np.uint8))


def save_side_projection(path: Path, image: np.ndarray, z_um: float | None, lateral_um: float | None) -> None:
    image_u8 = image.astype(np.uint8)
    if not z_um or not lateral_um:
        save_projection(path, image_u8)
        return

    height = max(1, round(image_u8.shape[0] * z_um / lateral_um))
    resized = Image.fromarray(image_u8).resize((image_u8.shape[1], height), Image.Resampling.BILINEAR)
    iio.imwrite(path, np.asarray(resized, dtype=np.uint8))


def add_label(frame: Image.Image, label: str) -> Image.Image:
    out = frame.convert("RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((8, 8, 94, 34), fill=(0, 0, 0))
    draw.text((14, 13), label, fill=(255, 255, 255), font=font)
    return out


def save_turntable_gif(volume_u8: np.ndarray, sizes: dict[str, float | None], path: Path) -> None:
    z_um = sizes.get("z") or 1.0
    xy_um = sizes.get("y") or sizes.get("x") or 1.0
    z_zoom = max(0.2, min(8.0, z_um / xy_um))
    isotropic = ndimage.zoom(volume_u8, (z_zoom, 1.0, 1.0), order=1)

    frames: list[Image.Image] = []
    for angle in np.linspace(0, 360, 36, endpoint=False):
        rotated = ndimage.rotate(isotropic, angle, axes=(0, 2), reshape=True, order=1, mode="constant")
        mip = rotated.max(axis=0)
        image = Image.fromarray(mip).convert("L")
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (640, 640), 0)
        canvas.paste(image, ((640 - image.width) // 2, (640 - image.height) // 2))
        frames.append(add_label(canvas, f"{int(angle):03d} deg"))

    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0)


def write_html_report(
    path: Path,
    input_dir: Path,
    volume: np.ndarray,
    sizes: dict[str, float | None],
    files: dict[str, str],
) -> None:
    sx = sizes.get("x")
    sy = sizes.get("y")
    sz = sizes.get("z")
    physical = "unknown"
    if sx and sy and sz:
        physical = f"{volume.shape[2] * sx:.1f} x {volume.shape[1] * sy:.1f} x {volume.shape[0] * sz:.1f} um"

    rows = "\n".join(
        f'<li><a href="{html.escape(rel)}">{html.escape(label)}</a></li>' for label, rel in files.items()
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>2P Z-stack reconstruction</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1b1b1b; }}
    img {{ max-width: min(900px, 100%); height: auto; background: #000; display: block; margin: 12px 0 24px; }}
    code {{ background: #eee; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>2P Z-stack reconstruction</h1>
  <p><b>Input:</b> <code>{html.escape(str(input_dir))}</code></p>
  <p><b>Volume:</b> {volume.shape[0]} z planes, {volume.shape[1]} x {volume.shape[2]} pixels, {volume.dtype}</p>
  <p><b>Voxel size:</b> X={sx or "unknown"} um, Y={sy or "unknown"} um, Z={sz or "unknown"} um</p>
  <p><b>Physical extent:</b> {physical}</p>
  <h2>Outputs</h2>
  <ul>{rows}</ul>
  <h2>Turntable</h2>
  <img src="{html.escape(files["turntable_gif"])}" alt="Rotating volume projection">
  <h2>Maximum-intensity projections</h2>
  <img src="{html.escape(files["mip_xy"])}" alt="XY maximum-intensity projection">
  <img src="{html.escape(files["mip_xz"])}" alt="XZ maximum-intensity projection">
  <img src="{html.escape(files["mip_yz"])}" alt="YZ maximum-intensity projection">
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("analysis/zstack_reconstruction"))
    parser.add_argument("--prefix", default="zstack_20260518")
    parser.add_argument("--low-percentile", type=float, default=0.2)
    parser.add_argument("--high-percentile", type=float, default=99.8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted_tiffs(args.input_dir)
    volume = load_volume(paths)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume after loading, got shape {volume.shape}")

    sizes = physical_sizes(paths[0])
    u8 = scale_to_uint8(volume, args.low_percentile, args.high_percentile)

    ome_tif = args.output_dir / f"{args.prefix}_volume.ome.tif"
    tf.imwrite(
        ome_tif,
        volume,
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": sizes.get("x"),
            "PhysicalSizeY": sizes.get("y"),
            "PhysicalSizeZ": sizes.get("z"),
            "PhysicalSizeXUnit": "um",
            "PhysicalSizeYUnit": "um",
            "PhysicalSizeZUnit": "um",
        },
    )

    mip_xy = args.output_dir / f"{args.prefix}_mip_xy.png"
    mip_xz = args.output_dir / f"{args.prefix}_mip_xz.png"
    mip_yz = args.output_dir / f"{args.prefix}_mip_yz.png"
    turntable = args.output_dir / f"{args.prefix}_turntable.gif"

    save_projection(mip_xy, u8.max(axis=0))
    save_side_projection(mip_xz, u8.max(axis=1), sizes.get("z"), sizes.get("x"))
    save_side_projection(mip_yz, u8.max(axis=2), sizes.get("z"), sizes.get("y"))
    save_turntable_gif(u8, sizes, turntable)

    report = args.output_dir / f"{args.prefix}_report.html"
    files = {
        "OME-TIFF volume": ome_tif.name,
        "mip_xy": mip_xy.name,
        "mip_xz": mip_xz.name,
        "mip_yz": mip_yz.name,
        "turntable_gif": turntable.name,
    }
    write_html_report(report, args.input_dir, volume, sizes, files)

    print(f"Loaded {len(paths)} TIFF files")
    print(f"Volume shape ZYX: {volume.shape}, dtype: {volume.dtype}")
    print(f"Voxel size um XYZ: {sizes.get('x')}, {sizes.get('y')}, {sizes.get('z')}")
    print(f"Wrote: {ome_tif}")
    print(f"Wrote: {mip_xy}")
    print(f"Wrote: {mip_xz}")
    print(f"Wrote: {mip_yz}")
    print(f"Wrote: {turntable}")
    print(f"Wrote: {report}")


if __name__ == "__main__":
    main()
