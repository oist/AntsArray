#!/usr/bin/env python3
"""Interactive GUI for reviewing ArUco crop quality.

Shows crops in a scrollable grid. Click to toggle accept/reject.
Green border = accepted (GT), red border = rejected.
All crops start as accepted. Click bad ones to reject.

Keyboard:
    Scroll       — mouse wheel or Page Up/Down
    A            — accept all visible
    R            — reject all visible
    S            — save and exit
    Q / Esc      — quit without saving
    F            — toggle showing only rejected (to double-check rejects)

On save, writes:
    _accepted.txt  — list of accepted .json filenames (use for training)
    _rejected.txt  — list of rejected .json filenames

Usage:
    python nn-aruco-detection-test/review_gui.py `
        --input-dir nn-aruco-detection-test/corner_gt_review
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


THUMB_SIZE = 128
BORDER = 4
COLS = 12
VISIBLE_ROWS = 6
CELL = THUMB_SIZE + BORDER * 2
WINDOW_W = COLS * CELL
WINDOW_H = VISIBLE_ROWS * (CELL + 16)  # +16 for ID text


class ReviewGUI:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir

        # Load all crop entries
        jsons = sorted(Path(input_dir).glob("*.json"))
        self.entries: list[dict] = []
        for jp in jsons:
            png = jp.with_suffix(".png")
            if not png.exists():
                continue
            with open(jp) as f:
                meta = json.load(f)
            meta["_json_path"] = str(jp)
            meta["_png_path"] = str(png)
            meta["_accepted"] = True
            self.entries.append(meta)

        print(f"Loaded {len(self.entries)} crops")

        # Load previous state if exists
        accepted_path = os.path.join(input_dir, "_accepted.txt")
        rejected_path = os.path.join(input_dir, "_rejected.txt")
        if os.path.isfile(rejected_path):
            with open(rejected_path) as f:
                rejected_set = set(f.read().strip().splitlines())
            for e in self.entries:
                if os.path.basename(e["_json_path"]) in rejected_set:
                    e["_accepted"] = False
            n_rej = sum(1 for e in self.entries if not e["_accepted"])
            print(f"  Restored previous state: {n_rej} rejected")

        self.scroll_offset = 0
        self.show_rejected_only = False

        # Preload thumbnails
        print("Loading thumbnails...")
        self.thumbs: dict[str, np.ndarray] = {}
        corner_colors = [
            (0, 0, 255),    # red = corner 0
            (0, 255, 0),    # green = corner 1
            (255, 0, 0),    # blue = corner 2
            (0, 255, 255),  # yellow = corner 3
        ]
        for e in self.entries:
            img = cv2.imread(e["_png_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((THUMB_SIZE, THUMB_SIZE), dtype=np.uint8)
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Draw corners
            for i, (cx, cy) in enumerate(e.get("corners_local", [])):
                cv2.circle(bgr, (int(cx), int(cy)), 3, corner_colors[i], -1)
            self.thumbs[e["_png_path"]] = bgr
        print("Ready.")

    def _visible_entries(self) -> list[dict]:
        if self.show_rejected_only:
            filtered = [e for e in self.entries if not e["_accepted"]]
        else:
            filtered = self.entries
        start = self.scroll_offset * COLS
        end = start + VISIBLE_ROWS * COLS
        return filtered[start:end]

    def _total_rows(self) -> int:
        if self.show_rejected_only:
            n = sum(1 for e in self.entries if not e["_accepted"])
        else:
            n = len(self.entries)
        return max(1, (n + COLS - 1) // COLS)

    def _render(self) -> np.ndarray:
        canvas = np.full((WINDOW_H, WINDOW_W, 3), 50, dtype=np.uint8)
        visible = self._visible_entries()

        for idx, entry in enumerate(visible):
            row = idx // COLS
            col = idx % COLS
            x = col * CELL
            y = row * (CELL + 16)

            # Border color
            color = (0, 180, 0) if entry["_accepted"] else (0, 0, 220)
            cv2.rectangle(canvas, (x, y), (x + CELL - 1, y + CELL - 1), color, BORDER)

            # Thumbnail
            thumb = self.thumbs.get(entry["_png_path"])
            if thumb is not None:
                tx = x + BORDER
                ty = y + BORDER
                canvas[ty:ty + THUMB_SIZE, tx:tx + THUMB_SIZE] = thumb

            # ID text
            mid = entry.get("marker_id", "?")
            cv2.putText(
                canvas, str(mid),
                (x + BORDER, y + CELL + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
            )

        # Status bar at bottom
        n_acc = sum(1 for e in self.entries if e["_accepted"])
        n_rej = len(self.entries) - n_acc
        total_rows = self._total_rows()
        mode = "REJECTED ONLY" if self.show_rejected_only else "ALL"
        status = (
            f"[{mode}]  Accepted: {n_acc}  Rejected: {n_rej}  "
            f"Row {self.scroll_offset+1}/{total_rows}  "
            f"Click=toggle  A=accept all  R=reject all  S=save  Q=quit  F=filter"
        )
        cv2.putText(
            canvas, status,
            (5, WINDOW_H - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
        )

        return canvas

    def _entry_at(self, mx: int, my: int) -> dict | None:
        col = mx // CELL
        row = my // (CELL + 16)
        if col >= COLS or row >= VISIBLE_ROWS:
            return None
        idx = row * COLS + col
        visible = self._visible_entries()
        if idx >= len(visible):
            return None
        return visible[idx]

    def _save(self):
        accepted = []
        rejected = []
        for e in self.entries:
            fname = os.path.basename(e["_json_path"])
            if e["_accepted"]:
                accepted.append(fname)
            else:
                rejected.append(fname)

        acc_path = os.path.join(self.input_dir, "_accepted.txt")
        rej_path = os.path.join(self.input_dir, "_rejected.txt")
        with open(acc_path, "w") as f:
            f.write("\n".join(accepted))
        with open(rej_path, "w") as f:
            f.write("\n".join(rejected))
        print(f"Saved: {len(accepted)} accepted, {len(rejected)} rejected")
        print(f"  {acc_path}")
        print(f"  {rej_path}")

    def run(self):
        win = "ArUco GT Review"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                entry = self._entry_at(x, y)
                if entry is not None:
                    entry["_accepted"] = not entry["_accepted"]
            elif event == cv2.EVENT_MOUSEWHEEL:
                if flags > 0:
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                else:
                    self.scroll_offset = min(self._total_rows() - 1, self.scroll_offset + 1)

        cv2.setMouseCallback(win, on_mouse)

        while True:
            canvas = self._render()
            cv2.imshow(win, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:  # Q or Esc
                break
            elif key == ord("s"):
                self._save()
                break
            elif key == ord("a"):
                # Accept all visible
                for e in self._visible_entries():
                    e["_accepted"] = True
            elif key == ord("r"):
                # Reject all visible
                for e in self._visible_entries():
                    e["_accepted"] = False
            elif key == ord("f"):
                self.show_rejected_only = not self.show_rejected_only
                self.scroll_offset = 0
            elif key == 0:  # Page Up (special key)
                self.scroll_offset = max(0, self.scroll_offset - VISIBLE_ROWS)
            elif key == 1:  # Page Down
                self.scroll_offset = min(self._total_rows() - 1, self.scroll_offset + VISIBLE_ROWS)

        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Interactive ArUco crop review GUI")
    p.add_argument("--input-dir", default="nn-aruco-detection-test/corner_gt_review")
    args = p.parse_args()
    gui = ReviewGUI(args.input_dir)
    gui.run()


if __name__ == "__main__":
    main()
