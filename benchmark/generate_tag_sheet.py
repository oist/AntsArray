#!/usr/bin/env python3
"""
Generate a printable tag sheet with ArUco markers at multiple physical sizes.

Each size section is labeled and includes a 10 mm ruler bar for print-scale
verification.  Output is a high-DPI PNG ready for printing at "actual size".

Example:
    python generate_tag_sheet.py --dpi 1200 --sizes 1.5,2.0,2.5 --ids 0-29
"""

import argparse
import cv2
import cv2.aruco as aruco
import numpy as np


def parse_id_range(s: str) -> list[int]:
    """Parse '0-29' or '0,1,5' into a list of ints."""
    ids = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def mm_to_px(mm: float, dpi: int) -> int:
    return round(mm / 25.4 * dpi)


def generate_tag_sheet(
    sizes_mm: list[float],
    ids: list[int],
    dpi: int = 1200,
    aruco_dict_type: int = aruco.DICT_4X4_1000,
    cols: int = 10,
    output_file: str = "tag_sheet.png",
):
    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)

    margin_mm = 2.0
    section_gap_mm = 5.0
    label_height_mm = 4.0
    ruler_height_mm = 3.0

    margin_px = mm_to_px(margin_mm, dpi)
    section_gap_px = mm_to_px(section_gap_mm, dpi)
    label_h_px = mm_to_px(label_height_mm, dpi)
    ruler_h_px = mm_to_px(ruler_height_mm, dpi)

    # Pre-compute section dimensions
    sections = []
    max_width = 0

    for size_mm in sizes_mm:
        tag_px = mm_to_px(size_mm, dpi)
        n_tags = len(ids)
        n_cols = min(cols, n_tags)
        n_rows = (n_tags + n_cols - 1) // n_cols

        sec_w = n_cols * tag_px + (n_cols + 1) * margin_px
        sec_h = label_h_px + n_rows * tag_px + (n_rows + 1) * margin_px + ruler_h_px
        sections.append((size_mm, tag_px, n_cols, n_rows, sec_w, sec_h))
        max_width = max(max_width, sec_w)

    total_height = sum(s[5] for s in sections) + section_gap_px * (len(sections) - 1)

    # Create white canvas
    img = 255 * np.ones((total_height, max_width), dtype=np.uint8)

    y_offset = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_base = dpi / 300  # scale text relative to DPI

    for size_mm, tag_px, n_cols, n_rows, sec_w, sec_h in sections:
        # Section label
        label = f"{size_mm} mm  (IDs {ids[0]}-{ids[-1]})"
        font_scale = font_scale_base * 0.5
        thickness = max(1, int(font_scale * 2))
        cv2.putText(
            img,
            label,
            (margin_px, y_offset + label_h_px - margin_px // 2),
            font,
            font_scale,
            0,
            thickness,
            cv2.LINE_AA,
        )
        grid_y0 = y_offset + label_h_px

        # Draw tags
        for idx, tag_id in enumerate(ids):
            r = idx // n_cols
            c = idx % n_cols
            tag_img = aruco.generateImageMarker(aruco_dict, tag_id, tag_px)
            x0 = c * tag_px + (c + 1) * margin_px
            y0 = grid_y0 + r * tag_px + (r + 1) * margin_px
            img[y0 : y0 + tag_px, x0 : x0 + tag_px] = tag_img

            # Tag ID label below each tag
            id_label = str(tag_id)
            id_font_scale = font_scale * 0.5
            id_thickness = max(1, int(id_font_scale * 2))
            (tw, th), _ = cv2.getTextSize(id_label, font, id_font_scale, id_thickness)
            tx = x0 + (tag_px - tw) // 2
            ty = y0 + tag_px + th + margin_px // 4
            if ty < img.shape[0]:
                cv2.putText(
                    img, id_label, (tx, ty), font, id_font_scale, 0, id_thickness, cv2.LINE_AA
                )

        # 10 mm ruler bar
        ruler_y = grid_y0 + n_rows * tag_px + (n_rows + 1) * margin_px + margin_px
        ruler_10mm_px = mm_to_px(10.0, dpi)
        ruler_thick = max(2, dpi // 300)

        # Main bar
        cv2.line(
            img,
            (margin_px, ruler_y),
            (margin_px + ruler_10mm_px, ruler_y),
            0,
            ruler_thick,
        )
        # End ticks
        tick_h = mm_to_px(1.5, dpi)
        cv2.line(img, (margin_px, ruler_y - tick_h), (margin_px, ruler_y + tick_h), 0, ruler_thick)
        cv2.line(
            img,
            (margin_px + ruler_10mm_px, ruler_y - tick_h),
            (margin_px + ruler_10mm_px, ruler_y + tick_h),
            0,
            ruler_thick,
        )
        # 1 mm ticks
        for i in range(1, 10):
            x_tick = margin_px + mm_to_px(float(i), dpi)
            small_tick = tick_h // 2 if i != 5 else tick_h
            cv2.line(img, (x_tick, ruler_y - small_tick), (x_tick, ruler_y + small_tick), 0, max(1, ruler_thick // 2))

        # Label
        ruler_label = "10 mm"
        cv2.putText(
            img,
            ruler_label,
            (margin_px + ruler_10mm_px + margin_px, ruler_y + tick_h),
            font,
            font_scale * 0.4,
            0,
            max(1, thickness // 2),
            cv2.LINE_AA,
        )

        y_offset += sec_h + section_gap_px

    cv2.imwrite(output_file, img)
    print(f"Tag sheet saved: {output_file}  ({img.shape[1]}x{img.shape[0]} px, {dpi} DPI)")
    print(f"Sizes: {sizes_mm} mm | IDs: {ids[0]}-{ids[-1]} | Print at 'actual size'")


def main():
    p = argparse.ArgumentParser(description="Generate printable multi-size ArUco tag sheet")
    p.add_argument("--dpi", type=int, default=1200, help="Print DPI (default: 1200)")
    p.add_argument(
        "--sizes",
        type=str,
        default="1.5,2.0,2.5",
        help="Comma-separated tag sizes in mm (default: 1.5,2.0,2.5)",
    )
    p.add_argument(
        "--ids",
        type=str,
        default="0-29",
        help="Tag IDs: '0-29' or '0,1,5,10' (default: 0-29)",
    )
    p.add_argument("--cols", type=int, default=10, help="Tags per row (default: 10)")
    p.add_argument("--output", type=str, default="tag_sheet.png", help="Output filename")
    args = p.parse_args()

    sizes_mm = [float(s.strip()) for s in args.sizes.split(",")]
    ids = parse_id_range(args.ids)

    generate_tag_sheet(
        sizes_mm=sizes_mm,
        ids=ids,
        dpi=args.dpi,
        cols=args.cols,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
