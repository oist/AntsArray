#!/usr/bin/env python3
"""
Generate the AntsArray ArUco calibration board as crisp, print-ready output:
a high-resolution PNG plus vector SVG and vector PDF.

Why vector
----------
Each marker's bit grid is drawn as exact rectangles in millimetres, so the board
prints sharp at any physical size. The earlier version rendered every marker at
1 pixel per module (e.g. an 8x8 marker into 8 px) and relied on the print
pipeline to scale it up; bilinear/bicubic interpolation then smeared adjacent
modules together and corrupted the codes, which hurt detection. Vector output
removes that failure mode entirely.

Default board (reproduces the 25-camera calibration rig used in 20260414)
------------------------------------------------------------------------
  46 x 35 markers, DICT_APRILTAG_36h10, IDs 0..1609 row-major,
  marker 2.5 mm, pitch 10 mm (gap 7.5 mm), 7.5 mm quiet margin
  -> page 467.5 x 357.5 mm.

Examples
--------
  python GenerateArucoGrid.py                       # default 2.5 mm board (PNG+SVG+PDF)
  python GenerateArucoGrid.py --marker-mm 3.0 --pitch-mm 12
  python GenerateArucoGrid.py --dict DICT_APRILTAG_36h11   # higher Hamming distance
  python GenerateArucoGrid.py --cols 20 --rows 14 --no-pdf

Notes
-----
* IDs are assigned row-major as id = row * cols + col, starting at 0.
* When printing, choose "Actual size" / 100% scale (NOT "fit to page"), then
  verify with a ruler: 10 markers span exactly 10 * pitch_mm.
* PDF export uses matplotlib; if it is missing, PNG and SVG are still written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2.aruco as aruco
import numpy as np

DEFAULT_DICT = "DICT_APRILTAG_36h10"


# ---------------------------------------------------------------------------
# Marker geometry
# ---------------------------------------------------------------------------

def resolve_dictionary(name: str) -> aruco.Dictionary:
    attr = name if name.startswith("DICT_") else f"DICT_{name}"
    if not hasattr(aruco, attr):
        known = sorted(k for k in dir(aruco) if k.startswith("DICT_"))
        raise SystemExit(f"Unknown ArUco dictionary {name!r}. Known: {', '.join(known)}")
    return aruco.getPredefinedDictionary(getattr(aruco, attr))


def module_matrix(dictionary: aruco.Dictionary, marker_id: int, total_modules: int,
                  border_bits: int) -> np.ndarray:
    """Canonical (total_modules x total_modules) grid; True = black module."""
    img = np.zeros((total_modules, total_modules), np.uint8)
    aruco.generateImageMarker(dictionary, marker_id, total_modules, img, border_bits)
    return img == 0


def build_rects(dictionary, cols, rows, marker_mm, pitch_mm, margin_mm,
                total_modules, border_bits):
    """List of (x, y, w, h) black rectangles in mm, run-length merged per row."""
    module_mm = marker_mm / total_modules
    rects: list[tuple[float, float, float, float]] = []
    marker_id = 0
    for r in range(rows):
        for c in range(cols):
            mat = module_matrix(dictionary, marker_id, total_modules, border_bits)
            x0 = margin_mm + c * pitch_mm
            y0 = margin_mm + r * pitch_mm
            for mr in range(total_modules):
                mc = 0
                while mc < total_modules:
                    if mat[mr, mc]:
                        run = mc
                        while run < total_modules and mat[mr, run]:
                            run += 1
                        rects.append((x0 + mc * module_mm, y0 + mr * module_mm,
                                      (run - mc) * module_mm, module_mm))
                        mc = run
                    else:
                        mc += 1
            marker_id += 1
    return rects


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def write_svg(path: Path, rects, page_w, page_h, caption: str | None) -> None:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{_fmt(page_w)}mm" height="{_fmt(page_h)}mm" '
        f'viewBox="0 0 {_fmt(page_w)} {_fmt(page_h)}">',
        f'<rect x="0" y="0" width="{_fmt(page_w)}" height="{_fmt(page_h)}" fill="#ffffff"/>',
        '<g fill="#000000" stroke="none" shape-rendering="crispEdges">',
    ]
    parts += [f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}"/>'
              for x, y, w, h in rects]
    parts.append('</g>')
    if caption:
        parts.append(
            f'<text x="{_fmt(page_w / 2)}" y="{_fmt(page_h - 2.2)}" '
            f'font-family="sans-serif" font-size="3" fill="#000000" '
            f'text-anchor="middle">{caption}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_pdf(path: Path, rects, page_w, page_h, caption: str | None) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib not available - skipping PDF (PNG/SVG still written).")
        return False
    fig = plt.figure(figsize=(page_w / 25.4, page_h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(page_h, 0)  # origin top-left, y down
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_collection(PatchCollection([Rectangle((x, y), w, h) for x, y, w, h in rects],
                                      facecolor="black", edgecolor="none", antialiased=False))
    if caption:
        ax.text(page_w / 2, page_h - 2.2, caption, ha="center", va="top",
                fontsize=8, family="sans-serif", color="black")
    fig.savefig(str(path), format="pdf", facecolor="white")
    plt.close(fig)
    return True


def write_png(path: Path, rects, page_w, page_h, dpi: int) -> None:
    import cv2
    s = dpi / 25.4  # px per mm
    canvas = np.full((round(page_h * s), round(page_w * s)), 255, np.uint8)
    for x, y, w, h in rects:
        canvas[round(y * s):round((y + h) * s), round(x * s):round((x + w) * s)] = 0
    cv2.imwrite(str(path), canvas)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a crisp ArUco calibration board (PNG + SVG + PDF).")
    ap.add_argument("--dict", default=DEFAULT_DICT, help=f"ArUco dictionary (default: {DEFAULT_DICT}).")
    ap.add_argument("--cols", type=int, default=46, help="Markers per row (default: 46).")
    ap.add_argument("--rows", type=int, default=35, help="Markers per column (default: 35).")
    ap.add_argument("--marker-mm", type=float, default=2.5, help="Marker side in mm (default: 2.5).")
    ap.add_argument("--pitch-mm", type=float, default=10.0, help="Center-to-center spacing in mm (default: 10).")
    ap.add_argument("--margin-mm", type=float, default=7.5, help="White quiet border in mm (default: 7.5).")
    ap.add_argument("--border-bits", type=int, default=1, help="Marker border thickness in modules (default: 1).")
    ap.add_argument("--png-dpi", type=int, default=300, help="Raster PNG resolution (default: 300).")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent), help="Output directory.")
    ap.add_argument("--basename", default=None, help="Output basename (default: aruco_grid_<DICT>_<marker>mm).")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--no-svg", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-caption", action="store_true")
    args = ap.parse_args()

    dictionary = resolve_dictionary(args.dict)
    dict_name = args.dict if args.dict.startswith("DICT_") else f"DICT_{args.dict}"
    total_modules = int(dictionary.markerSize) + 2 * args.border_bits
    n_markers = args.cols * args.rows
    capacity = len(dictionary.bytesList)

    module_mm = args.marker_mm / total_modules
    page_w = 2 * args.margin_mm + (args.cols - 1) * args.pitch_mm + args.marker_mm
    page_h = 2 * args.margin_mm + (args.rows - 1) * args.pitch_mm + args.marker_mm

    print(f"dict={dict_name}  data-bits={dictionary.markerSize}  total-modules={total_modules}")
    print(f"grid={args.cols}x{args.rows}  markers={n_markers} (IDs 0..{n_markers - 1})  capacity={capacity}")
    print(f"marker={args.marker_mm}mm  module={module_mm:.4f}mm  pitch={args.pitch_mm}mm  "
          f"gap={args.pitch_mm - args.marker_mm}mm  margin={args.margin_mm}mm")
    print(f"page = {page_w:.2f} x {page_h:.2f} mm")
    if n_markers > capacity:
        print(f"WARNING: {n_markers} markers exceeds dictionary capacity ({capacity}); "
              f"IDs >= {capacity} are invalid. Reduce grid or pick a larger dictionary.")

    caption = None if args.no_caption else (
        f"{dict_name}  {args.cols}x{args.rows}  marker={args.marker_mm}mm  "
        f"pitch={args.pitch_mm}mm  IDs 0-{n_markers - 1} row-major")

    print("Building marker rectangles ...")
    rects = build_rects(dictionary, args.cols, args.rows, args.marker_mm, args.pitch_mm,
                        args.margin_mm, total_modules, args.border_bits)
    print(f"rectangles (run-length merged): {len(rects)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = args.basename or f"aruco_grid_{dict_name}_{_fmt(args.marker_mm)}mm"

    if not args.no_svg:
        p = outdir / f"{base}.svg"
        write_svg(p, rects, page_w, page_h, caption)
        print(f"wrote {p}  ({p.stat().st_size / 1024:.1f} KB)")
    if not args.no_pdf:
        p = outdir / f"{base}.pdf"
        if write_pdf(p, rects, page_w, page_h, caption):
            print(f"wrote {p}  ({p.stat().st_size / 1024:.1f} KB)")
    if not args.no_png:
        p = outdir / f"{base}.png"
        write_png(p, rects, page_w, page_h, args.png_dpi)
        print(f"wrote {p}  ({p.stat().st_size / 1024:.1f} KB, {args.png_dpi} dpi)")

    print("DONE.")


if __name__ == "__main__":
    main()
