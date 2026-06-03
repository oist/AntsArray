#!/usr/bin/env python3
"""Generate a printable, cuttable tag sheet for custom ArUco dictionary A.

Layout:
  - 10x10 grid of markers (IDs 0-99)
  - Each marker at specified physical size (default 1.5mm)
  - White quiet zone (margin) around each marker (default 0.5mm)
  - ID labels positioned outside the margin
  - Cutting crop marks at row/column boundaries for precise cutting

Usage:
    python aruco_detection/custom_dicts/generate_cuttable_sheet.py \
        --npz aruco_detection/custom_dicts/custom_4x4_A100_d4_20260410_103938.npz \
        --output aruco_detection/custom_dicts/custom_4x4_A100_d4_cuttable.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np


def load_custom_dictionary(npz_path):
    data = np.load(str(npz_path), allow_pickle=True)
    d = aruco.Dictionary()
    d.bytesList = data["bytesList"]
    d.markerSize = 4
    d.maxCorrectionBits = int(data["max_correction_bits"])
    patterns = data["patterns"] if "patterns" in data.files else None
    return d, d.bytesList.shape[0], int(data["min_distance"]), patterns


def extract_marker_pattern(dictionary, marker_id, marker_size=4):
    """Extract the (markerSize, markerSize) bit pattern (1=black, 0=white) for a marker."""
    img = aruco.generateImageMarker(dictionary, marker_id, marker_size + 2)
    data = img[1:marker_size + 1, 1:marker_size + 1]
    return (data < 128).astype(np.uint8)


def mm_to_px(mm, dpi):
    return round(mm * dpi / 25.4)


def generate_cuttable_sheet(
    dictionary: aruco.Dictionary,
    n_markers: int,
    min_d: int,
    output_path: Path,
    marker_mm: float = 1.5,
    margin_mm: float = 0.5,
    label_gap_mm: float = 1.8,
    col_gap_mm: float = 1.2,
    crop_mark_mm: float = 2.0,
    page_margin_mm: float = 8.0,
    dpi: int = 600,
    cols: int = 10,
):
    """Generate a cuttable tag sheet with crop marks.

    Physical layout per tag:
        [margin][marker][margin]
         0.5mm   1.5mm   0.5mm  = 2.5mm wide

    Between tags:
        Horizontal gap (col_gap_mm) for vertical crop marks
        Vertical gap (label_gap_mm) for ID labels + horizontal crop marks

    Crop marks: short lines at every row/column cut boundary,
    extending outside the grid area.
    """
    rows = (n_markers + cols - 1) // cols

    # Convert dimensions to pixels
    marker_px = mm_to_px(marker_mm, dpi)
    margin_px = mm_to_px(margin_mm, dpi)
    label_gap_px = mm_to_px(label_gap_mm, dpi)
    col_gap_px = mm_to_px(col_gap_mm, dpi)
    crop_mark_px = mm_to_px(crop_mark_mm, dpi)
    page_margin_px = mm_to_px(page_margin_mm, dpi)

    # Tag cell = marker + 2 margins (the cut boundary)
    tag_w = marker_px + 2 * margin_px
    tag_h = tag_w  # square tags

    # Cell = tag + gaps for labels/crop marks
    cell_w = tag_w + col_gap_px
    cell_h = tag_h + label_gap_px

    # Grid area
    grid_w = cols * cell_w - col_gap_px  # no gap after last column
    grid_h = rows * cell_h - label_gap_px  # no gap after last row... actually keep it for labels

    # Total sheet with page margins and crop mark space
    sheet_w = grid_w + 2 * page_margin_px + 2 * crop_mark_px
    sheet_h = rows * cell_h + 2 * page_margin_px + 2 * crop_mark_px + mm_to_px(6, dpi)  # extra for title

    # Title area
    title_h = mm_to_px(6, dpi)

    # Origin of grid (top-left of first tag's cut boundary)
    ox = page_margin_px + crop_mark_px
    oy = page_margin_px + crop_mark_px + title_h

    # Create white sheet
    sheet = np.ones((sheet_h, sheet_w), dtype=np.uint8) * 255

    # -- Title --
    title = f"Custom_A  {n_markers} markers  min_d={min_d}  marker={marker_mm}mm  margin={margin_mm}mm"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = mm_to_px(2.0, dpi) / 30  # scale relative to desired text height
    cv2.putText(sheet, title, (ox, oy - mm_to_px(2, dpi)),
                font, font_scale, 0, max(1, dpi // 300), cv2.LINE_AA)

    # Scale ruler (10mm)
    ruler_x = ox + grid_w - mm_to_px(12, dpi)
    ruler_y = oy - mm_to_px(5, dpi)
    ruler_len = mm_to_px(10, dpi)
    cv2.line(sheet, (ruler_x, ruler_y), (ruler_x + ruler_len, ruler_y), 0, max(1, dpi // 300))
    for tick_mm in range(11):
        tx = ruler_x + mm_to_px(tick_mm, dpi)
        th = mm_to_px(0.8 if tick_mm % 5 == 0 else 0.4, dpi)
        cv2.line(sheet, (tx, ruler_y - th), (tx, ruler_y + th), 0, max(1, dpi // 300))
    cv2.putText(sheet, "10 mm", (ruler_x + ruler_len + mm_to_px(1, dpi), ruler_y + mm_to_px(0.5, dpi)),
                font, font_scale * 0.6, 0, max(1, dpi // 300), cv2.LINE_AA)

    # -- Draw markers and ID labels --
    for mid in range(n_markers):
        r, c = divmod(mid, cols)

        # Tag cut boundary top-left
        tag_x = ox + c * cell_w
        tag_y = oy + r * cell_h

        # Marker position (centered in tag with margins)
        mx = tag_x + margin_px
        my = tag_y + margin_px

        # Generate marker
        img = aruco.generateImageMarker(dictionary, mid, marker_px)
        sheet[my:my + marker_px, mx:mx + marker_px] = img

        # ID label — centered below the tag, in the gap area
        label = str(mid)
        label_font_scale = font_scale * 0.5
        (tw, th), _ = cv2.getTextSize(label, font, label_font_scale, max(1, dpi // 300))
        lx = tag_x + (tag_w - tw) // 2
        ly = tag_y + tag_h + mm_to_px(1.2, dpi)
        cv2.putText(sheet, label, (lx, ly), font, label_font_scale, 0,
                    max(1, dpi // 300), cv2.LINE_AA)

    # -- Crop marks: L-shaped corner marks at all 4 corners of each tag --
    crop_color = 100  # dark gray
    crop_thickness = max(1, dpi // 600)
    arm_len = mm_to_px(0.6, dpi)  # length of each arm of the L
    corner_gap = mm_to_px(0.15, dpi)  # tiny gap between corner mark and tag edge

    for mid in range(n_markers):
        r, c = divmod(mid, cols)
        tag_x = ox + c * cell_w
        tag_y = oy + r * cell_h

        # Four corners of the tag cut boundary
        corners = [
            (tag_x, tag_y),                      # top-left
            (tag_x + tag_w, tag_y),              # top-right
            (tag_x, tag_y + tag_h),              # bottom-left
            (tag_x + tag_w, tag_y + tag_h),      # bottom-right
        ]
        # Direction each arm extends outward from the tag
        directions = [
            ((-1, 0), (0, -1)),  # top-left: arms go left and up
            ((1, 0), (0, -1)),   # top-right: arms go right and up
            ((-1, 0), (0, 1)),   # bottom-left: arms go left and down
            ((1, 0), (0, 1)),    # bottom-right: arms go right and down
        ]
        for (cx, cy), ((dx1, dy1), (dx2, dy2)) in zip(corners, directions):
            # Offset start point slightly away from corner
            sx = cx + dx1 * corner_gap
            sy = cy + dy1 * corner_gap
            sx2 = cx + dx2 * corner_gap
            sy2 = cy + dy2 * corner_gap
            # Horizontal arm
            cv2.line(sheet, (cx + dx1 * corner_gap, cy),
                     (cx + dx1 * (corner_gap + arm_len), cy),
                     crop_color, crop_thickness)
            # Vertical arm
            cv2.line(sheet, (cx, cy + dy2 * corner_gap),
                     (cx, cy + dy2 * (corner_gap + arm_len)),
                     crop_color, crop_thickness)

    # -- Print info footer --
    footer_y = sheet_h - page_margin_px
    footer_text = f"DPI: {dpi}  |  Cut along crop marks  |  Each tag: {marker_mm + 2*margin_mm:.1f}mm x {marker_mm + 2*margin_mm:.1f}mm"
    cv2.putText(sheet, footer_text, (ox, footer_y),
                font, font_scale * 0.5, 120, max(1, dpi // 300), cv2.LINE_AA)

    cv2.imwrite(str(output_path), sheet)

    # Print physical dimensions
    sheet_w_mm = sheet_w * 25.4 / dpi
    sheet_h_mm = sheet_h * 25.4 / dpi
    print(f"  Sheet: {sheet_w} x {sheet_h} px ({sheet_w_mm:.1f} x {sheet_h_mm:.1f} mm)")
    print(f"  Tag cut size: {marker_mm + 2*margin_mm:.1f} x {marker_mm + 2*margin_mm:.1f} mm")
    print(f"  Grid: {cols} x {rows}")
    print(f"  DPI: {dpi}")
    print(f"  Saved: {output_path}")


def generate_cuttable_svg(
    dictionary: aruco.Dictionary,
    n_markers: int,
    min_d: int,
    output_path: Path,
    marker_mm: float = 1.5,
    margin_mm: float = 0.3,
    label_gap_mm: float = 1.8,
    col_gap_mm: float = 1.2,
    crop_arm_mm: float = 0.6,
    crop_gap_mm: float = 0.15,
    page_margin_mm: float = 8.0,
    cols: int = 10,
    name: str = "Custom_A",
):
    """Generate a cuttable tag sheet as SVG (vector format).

    All coordinates and sizes are in millimeters (SVG user units = mm via viewBox).
    Output is true vector — markers are rectangles, labels are text, crop marks are lines.
    """
    rows = (n_markers + cols - 1) // cols

    # Tag cut boundary (marker + 2 margins)
    tag_w = marker_mm + 2 * margin_mm
    tag_h = tag_w

    # Per cell with label gap and column gap
    cell_w = tag_w + col_gap_mm
    cell_h = tag_h + label_gap_mm

    grid_w = cols * cell_w - col_gap_mm

    # Sheet dimensions in mm
    title_h = 6.0
    sheet_w = grid_w + 2 * page_margin_mm + 2 * crop_arm_mm
    sheet_h = rows * cell_h + 2 * page_margin_mm + 2 * crop_arm_mm + title_h + 6.0  # extra footer

    # Origin of grid (top-left of first tag)
    ox = page_margin_mm + crop_arm_mm
    oy = page_margin_mm + crop_arm_mm + title_h

    # Cell size for marker bits: marker_mm / 6 (4 data + 1 border on each side)
    cell_mm = marker_mm / 6.0

    # SVG header — viewBox in mm
    svg = []
    svg.append(f'<?xml version="1.0" encoding="UTF-8"?>')
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{sheet_w}mm" height="{sheet_h}mm" '
        f'viewBox="0 0 {sheet_w} {sheet_h}" '
        f'shape-rendering="crispEdges">'
    )
    svg.append(f'  <rect width="{sheet_w}" height="{sheet_h}" fill="white"/>')

    # Title
    title = f"{name}  {n_markers} markers  min_d={min_d}  marker={marker_mm}mm  margin={margin_mm}mm"
    svg.append(
        f'  <text x="{ox}" y="{oy - 2}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="2.0" fill="black">{title}</text>'
    )

    # Scale ruler (10mm) above title
    rx = ox + grid_w - 12.0
    ry = oy - 5.0
    svg.append(
        f'  <line x1="{rx}" y1="{ry}" x2="{rx + 10}" y2="{ry}" '
        f'stroke="black" stroke-width="0.15"/>'
    )
    for tick_mm in range(11):
        tx = rx + tick_mm
        th = 0.8 if tick_mm % 5 == 0 else 0.4
        svg.append(
            f'  <line x1="{tx}" y1="{ry - th}" x2="{tx}" y2="{ry + th}" '
            f'stroke="black" stroke-width="0.15"/>'
        )
    svg.append(
        f'  <text x="{rx + 11}" y="{ry + 0.7}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="1.5" fill="black">10 mm</text>'
    )

    # Markers, labels, and crop marks
    for mid in range(n_markers):
        r, c = divmod(mid, cols)
        tag_x = ox + c * cell_w
        tag_y = oy + r * cell_h

        # --- Marker: render as nested rectangles ---
        # 6x6 cells: 1-cell black border + 4x4 data
        # Black border (full marker square)
        mx = tag_x + margin_mm
        my = tag_y + margin_mm
        # Render full black square first
        svg.append(
            f'  <rect x="{mx}" y="{my}" width="{marker_mm}" height="{marker_mm}" fill="black"/>'
        )
        # Then render WHITE cells where the data bit is 0 (in the inner 4x4 region)
        pattern = extract_marker_pattern(dictionary, mid, 4)
        # Inner data starts at 1 cell offset
        for ri in range(4):
            for ci in range(4):
                if pattern[ri, ci] == 0:  # white cell
                    cx = mx + (ci + 1) * cell_mm
                    cy = my + (ri + 1) * cell_mm
                    svg.append(
                        f'  <rect x="{cx}" y="{cy}" width="{cell_mm}" height="{cell_mm}" '
                        f'fill="white"/>'
                    )

        # --- L-shaped corner crop marks at 4 corners of tag cut boundary ---
        corners = [
            (tag_x, tag_y, -1, -1),                          # top-left
            (tag_x + tag_w, tag_y, +1, -1),                  # top-right
            (tag_x, tag_y + tag_h, -1, +1),                  # bottom-left
            (tag_x + tag_w, tag_y + tag_h, +1, +1),          # bottom-right
        ]
        for cx, cy, dx, dy in corners:
            # Horizontal arm
            svg.append(
                f'  <line x1="{cx + dx * crop_gap_mm}" y1="{cy}" '
                f'x2="{cx + dx * (crop_gap_mm + crop_arm_mm)}" y2="{cy}" '
                f'stroke="#666" stroke-width="0.1"/>'
            )
            # Vertical arm
            svg.append(
                f'  <line x1="{cx}" y1="{cy + dy * crop_gap_mm}" '
                f'x2="{cx}" y2="{cy + dy * (crop_gap_mm + crop_arm_mm)}" '
                f'stroke="#666" stroke-width="0.1"/>'
            )

        # --- ID label centered below tag ---
        label = str(mid)
        label_y = tag_y + tag_h + 1.3
        svg.append(
            f'  <text x="{tag_x + tag_w / 2}" y="{label_y}" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="1.2" '
            f'fill="black" text-anchor="middle">{label}</text>'
        )

    # Footer
    footer_y = sheet_h - page_margin_mm
    footer = (f"Vector SVG  |  Cut along corner marks  |  "
              f"Each tag: {tag_w:.1f}mm x {tag_h:.1f}mm")
    svg.append(
        f'  <text x="{ox}" y="{footer_y}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="1.2" fill="#555">{footer}</text>'
    )

    svg.append('</svg>')

    output_path.write_text("\n".join(svg), encoding="utf-8")
    print(f"  Sheet (SVG): {sheet_w:.1f} x {sheet_h:.1f} mm")
    print(f"  Tag cut size: {tag_w:.1f} x {tag_h:.1f} mm")
    print(f"  Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Generate cuttable ArUco tag sheet")
    p.add_argument("--npz", required=True, help="Custom dictionary NPZ file")
    p.add_argument("--output", required=True,
                   help="Output path. Extension determines format: .png, .svg, or .pdf "
                        "(if extension is .png, also saves matching .svg next to it)")
    p.add_argument("--marker-mm", type=float, default=1.5, help="Marker size in mm (default: 1.5)")
    p.add_argument("--margin-mm", type=float, default=0.3, help="White margin in mm (default: 0.3)")
    p.add_argument("--dpi", type=int, default=600, help="Output DPI for PNG (default: 600)")
    p.add_argument("--cols", type=int, default=10, help="Columns (default: 10)")
    p.add_argument("--name", default="Custom_A", help="Dictionary name shown in title")
    p.add_argument("--no-svg", action="store_true",
                   help="Skip SVG output when generating PNG (default: also save SVG)")
    args = p.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dictionary, n_markers, min_d, _patterns = load_custom_dictionary(args.npz)
    print(f"Loaded: {n_markers} markers, min_d={min_d}")

    ext = output_path.suffix.lower()
    if ext == ".svg":
        generate_cuttable_svg(
            dictionary, n_markers, min_d, output_path,
            marker_mm=args.marker_mm,
            margin_mm=args.margin_mm,
            cols=args.cols,
            name=args.name,
        )
    elif ext == ".png":
        generate_cuttable_sheet(
            dictionary, n_markers, min_d, output_path,
            marker_mm=args.marker_mm,
            margin_mm=args.margin_mm,
            dpi=args.dpi,
            cols=args.cols,
        )
        if not args.no_svg:
            svg_path = output_path.with_suffix(".svg")
            generate_cuttable_svg(
                dictionary, n_markers, min_d, svg_path,
                marker_mm=args.marker_mm,
                margin_mm=args.margin_mm,
                cols=args.cols,
                name=args.name,
            )
    else:
        raise ValueError(f"Unsupported output extension: {ext}. Use .png or .svg")


if __name__ == "__main__":
    main()
