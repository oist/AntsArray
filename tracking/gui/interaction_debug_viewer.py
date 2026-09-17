#!/usr/bin/env python3
"""Live distance-only contact review using finished tracks projected onto video."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.contact_review_utils import build_review_clip
from tracking.colony.interaction_one_chunk import DEFAULT_MICRO_DISTANCE_MM
from tracking.gui.contact_distance import FinishedTrackClip, skeleton_pair_distances, validate_distance
from tracking.gui.aruco_sleep_viewer import (
    ArucoPoint, ArucoSleepViewer, OpenCvVideoReader, SleepMotionStore, SleepParameters,
    SLEAP_SKELETON_EDGES, color_for_id, infer_hmats_path, infer_x_split,
    parse_camera_index, resolve_chunk_specs,
)
from tracking.gui.multicam_tracking_viewer import apply_homography_points


def scale_bar_endpoints(panorama_h, *, origin, length_mm, mm_per_pixel):
    """A horizontal camera-image bar with an exact calibrated panorama length."""
    if not np.isfinite([length_mm, mm_per_pixel]).all() or min(length_mm, mm_per_pixel) <= 0:
        raise ValueError("Scale-bar length and mm_per_pixel must be positive and finite")
    origin = np.asarray(origin, dtype=float)
    projected = apply_homography_points(np.array([origin, origin + [1, 0]]), panorama_h)
    direction = projected[1] - projected[0]
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid scale-bar calibration")
    end = projected[0] + direction / norm * (length_mm / mm_per_pixel)
    camera_end = apply_homography_points(end[None], np.linalg.inv(panorama_h))[0]
    if not np.isfinite(camera_end).all() or camera_end[0] <= origin[0]:
        raise ValueError("Scale bar crosses the calibration's horizon")
    return np.array([origin, camera_end])


def finished_track_frame(review, frame, inverse_h, *, width, height):
    """Project cached final TrackID/TrackX/TrackY and X/Y; never match raw detections."""
    local = int(frame) - review.start
    if not 0 <= local < review.stop - review.start:
        raise ValueError("Frame outside finished-track review interval")
    anchors = apply_homography_points(review.anchors[local], inverse_h)
    xy = review.xy[local]
    poses = apply_homography_points(xy.reshape(-1, 2), inverse_h).reshape(xy.shape)
    visible = (np.isfinite(anchors).all(axis=1)
               & (anchors[:, 0] >= 0) & (anchors[:, 0] < width)
               & (anchors[:, 1] >= 0) & (anchors[:, 1] < height))
    # ArucoPoint is the base sleep renderer's point container, not a raw observation.
    points = [ArucoPoint(int(review.ids[i]), float(anchors[i, 0]), float(anchors[i, 1]), np.nan)
              for i in np.flatnonzero(visible)]
    return points, {int(review.ids[i]): poses[i] for i in np.flatnonzero(visible)}


class InteractionDebugViewer(ArucoSleepViewer):
    def __init__(self, root, *, review: FinishedTrackClip, **kwargs):
        self.review = review
        self.inverse_h = np.linalg.inv(kwargs["panorama_h"])
        self.return_frame = review.manifest["return_frame"]
        self.distance_var = tk.StringVar(root, value=str(DEFAULT_MICRO_DISTANCE_MM))
        self.show_sleep_var = tk.BooleanVar(root, value=False)
        self.review_status = tk.StringVar(root)
        self.distance_error = tk.StringVar(root)
        self.distance_job = None
        self.syncing_distance = False
        self.selected_pair = None
        self.pair_distances = []
        self.distance_frame_key = None
        super().__init__(root, **kwargs)
        self.distance_var.trace_add("write", self.queue_distance_update)

    def _build_ui(self):
        super()._build_ui()
        self.root.title(f"Live distance review - {self.video_path.name}")
        toolbar = self.root.grid_slaves(row=0, column=0)[0]
        for widget in toolbar.winfo_children():
            if isinstance(widget, ttk.Checkbutton):
                label = {"ArUco IDs": "Track IDs", "SLEAP": "Track poses"}.get(widget.cget("text"))
                if label:
                    widget.configure(text=label)
                if widget.cget("text") == "Metrics":
                    self.metrics_button = widget
                    widget.configure(state="disabled")
        ttk.Checkbutton(toolbar, text="Sleep/wake", variable=self.show_sleep_var,
                        command=self.toggle_sleep_labels).pack(side="left", padx=(8, 0))
        panel = self.root.grid_slaves(row=1, column=1)[0]
        for child in panel.winfo_children():
            child.destroy()
        controls = ttk.Frame(panel)
        controls.pack(fill="x", pady=6)
        ttk.Label(controls, text="Skeleton distance (mm)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(controls, textvariable=self.distance_var, from_=0, to=10, increment=.01,
                    width=8).grid(row=0, column=1, sticky="e")
        controls.columnconfigure(0, weight=1)
        self.distance_slider = ttk.Scale(panel, from_=0, to=2, command=self.distance_from_slider)
        self.distance_slider.set(DEFAULT_MICRO_DISTANCE_MM)
        self.distance_slider.pack(fill="x", pady=(0, 4))
        ttk.Label(panel, textvariable=self.distance_error, foreground="#ad2020", wraplength=305).pack(fill="x")
        actions = ttk.Frame(panel)
        actions.pack(fill="x", pady=4)
        ttk.Button(actions, text="Colony crossing", command=lambda: self.goto_frame(self.return_frame, pause=True)).pack(side="left")
        ttk.Button(actions, text="Export frame", command=self.export_review).pack(side="right")
        ttk.Label(panel, textvariable=self.review_status, wraplength=305).pack(fill="x", pady=6)

        table_frame = ttk.Frame(panel)
        table_frame.pack(fill="both", expand=True)
        self.pair_table = ttk.Treeview(table_frame, columns=("pair", "mm", "state"), show="headings", height=12)
        for name, label, width in (("pair", "Pair", 83), ("mm", "Distance mm", 102), ("state", "State", 70)):
            self.pair_table.heading(name, text=label)
            self.pair_table.column(name, width=width, stretch=True, anchor="center")
        self.pair_table.tag_configure("hit", background="#cff3ed")
        self.pair_table.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.pair_table.yview)
        scrollbar.pack(side="right", fill="y")
        self.pair_table.configure(yscrollcommand=scrollbar.set)
        self.pair_table.bind("<<TreeviewSelect>>", self.select_pair)

        classifier = ttk.Frame(panel)
        classifier.pack(fill="x", pady=(8, 0))
        ttk.Label(classifier, text="Sleep classifier").grid(row=0, column=0, sticky="w")
        for row, (label, key) in enumerate((("Body (mm/s)", "body_threshold_mm_s"),
                                          ("Antenna (mm/s)", "antenna_threshold_mm_s"),
                                          ("History (s)", "window_seconds"),
                                          ("Quiet fraction", "quiet_fraction")), start=1):
            ttk.Label(classifier, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(classifier, textvariable=self.param_vars[key], width=7).grid(row=row, column=1, sticky="e")
        classifier.columnconfigure(0, weight=1)
        ttk.Button(classifier, text="Apply sleep", command=self.apply_parameters).grid(row=5, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(panel, textvariable=self.parameter_status_var, wraplength=305).pack(fill="x")
        self.slider.configure(from_=self.review.start, to=self.review.stop - 1)

    def toggle_sleep_labels(self):
        self.metrics_button.configure(state="normal" if self.show_sleep_var.get() else "disabled")
        self.render()

    def distance_from_slider(self, raw):
        if not self.syncing_distance:
            self.distance_var.set(f"{float(raw):.3f}")

    def queue_distance_update(self, *_args):
        if self.distance_job is not None:
            self.root.after_cancel(self.distance_job)
        self.distance_job = self.root.after(75, self.apply_contact_parameters)

    def apply_contact_parameters(self):
        self.distance_job = None
        try:
            threshold = validate_distance(self.distance_var.get())
        except ValueError:
            pass  # Render clears hits and shows the validation error, not stale results.
        else:
            self.syncing_distance = True
            try:
                self.distance_slider.configure(to=max(2, threshold))
                self.distance_slider.set(threshold)
            finally:
                self.syncing_distance = False
        self.render()

    def goto_frame(self, frame, *, pause):
        super().goto_frame(int(np.clip(frame, self.review.start, self.review.stop-1)), pause=pause)

    def select_pair(self, _event=None):
        selection = self.pair_table.selection()
        pair = tuple(map(int, selection[0].split(":"))) if selection else None
        if pair != self.selected_pair:
            self.selected_pair = pair
            self.render()

    def export_review(self):
        try:
            threshold = validate_distance(self.distance_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid distance", str(exc), parent=self.root)
            return
        self.render()
        if self.current_annotated_rgb is None:
            return
        settings = dict(version=1, geometry="all_skeleton_segments_and_nodes", frame=self.current_frame,
                        distance_mm=threshold, mm_per_pixel=self.review.manifest["mm_per_pixel"],
                        side=self.review.manifest["side"], video=str(self.video_path),
                        pose_cache=str(self.review.root), sources=self.review.manifest.get("sources", []))
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
        root = self.review.root / "reviews" / f"distance_frame_{self.current_frame}_{key}"
        root.mkdir(parents=True, exist_ok=True)
        with (root / "pair_distances.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "ant_a", "ant_b", "distance_mm", "hit", "a_x_px", "a_y_px", "b_x_px", "b_y_px"])
            for row in self.pair_distances:
                writer.writerow([self.current_frame, row.ant_a, row.ant_b,
                                 row.distance_mm if np.isfinite(row.distance_mm) else "", row.is_hit(threshold),
                                 *(row.point_a or ("", "")), *(row.point_b or ("", ""))])
        (root / "settings.json").write_text(json.dumps(settings, indent=2)+"\n")
        self.parameter_status_var.set(f"Saved: {root.name}")
        print(f"Distance frame exported: {root}", flush=True)

    def pixel(self, xy):
        point = self.inverse_h @ np.r_[xy, 1]
        if not np.isfinite(point).all() or abs(point[2]) < 1e-12:
            return None
        return tuple(np.round(point[:2] / point[2]).astype(int))

    def render(self):
        self._label_boxes = []
        try:
            image = self.video.read_frame(self.current_frame).copy()
            self.scale_bar = self.scale_bar_layout(image)
            self._label_boxes.append(self.scale_bar["bounds"])
            spec = self.labels.spec_for_frame(self.current_frame)
            points, poses = finished_track_frame(self.review, self.current_frame, self.inverse_h,
                                                width=image.shape[1], height=image.shape[0])
            pose_count = self.draw_finished_poses(image, poses) if self.show_sleap_var.get() else 0
            counts = self.draw_aruco_and_sleep(image, points, spec)
            self._draw_text(image, f"Frame {self.current_frame}  chunk {spec.chunk:03d}",
                            (12, 34), (255, 255, 255), scale=0.85)
            self.draw_scale_bar(image)
            self.current_annotated_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.display_current_image()
            sleep_status = (f" | sleep {counts['sleep']} | wake {counts['wake']} | unknown {counts['unknown']}"
                            if self.show_sleep_var.get() else "")
            self.status_var.set(
                f"Frame {self.current_frame:,} | chunk {spec.chunk:03d} local {spec.to_local(self.current_frame):,} | "
                f"Finished tracks {len(points)} | poses {pose_count}{sleep_status}"
            )
        except Exception as exc:
            self.current_annotated_rgb = None
            self.canvas.delete("all")
            self.canvas_image = None
            self.status_var.set(f"Frame {self.current_frame}: {exc}")

    def scale_bar_layout(self, image):
        h, w = image.shape[:2]
        s = self.overlay_scale
        margin = min(round(40*s), w//12, h//12)
        points = scale_bar_endpoints(self.panorama_h, origin=(margin, h-margin), length_mm=1,
                                     mm_per_pixel=self.review.manifest["mm_per_pixel"])
        (tw, th), baseline = cv2.getTextSize("1 mm", cv2.FONT_HERSHEY_SIMPLEX, .6*s, max(1, round(2*s)))
        label_origin = (round((points[0, 0]+points[1, 0]-tw)/2), round(points[0, 1]-14*s))
        pad = max(3, round(4*s))
        bounds = (min(points[0, 0], label_origin[0])-pad, label_origin[1]-th-pad,
                  max(points[1, 0], label_origin[0]+tw)+pad, points[0, 1]+round(6*s)+pad)
        return dict(points=points, label_origin=label_origin, bounds=bounds)

    def draw_scale_bar(self, image):
        a, b = np.round(self.scale_bar["points"]).astype(int)
        s = self.overlay_scale
        tick = np.array([0, round(6*s)])
        for color, thickness in (((0, 0, 0), max(3, round(6*s))), ((255, 255, 255), max(1, round(3*s)))):
            cv2.line(image, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
            for p in (a, b):
                cv2.line(image, tuple(p-tick), tuple(p+tick), color, thickness, cv2.LINE_AA)
        super()._draw_text(image, "1 mm", self.scale_bar["label_origin"], (255, 255, 255), scale=.6)

    def draw_finished_poses(self, image, poses):
        count = 0
        h, w = image.shape[:2]
        margin = round(20*self.overlay_scale)
        for track_id, xy in poses.items():
            color = color_for_id(track_id)
            line_color = tuple(max(45, round(channel*0.65)) for channel in color)
            nodes = {i: tuple(np.round(point).astype(int)) for i, point in enumerate(xy)
                     if np.isfinite(point).all() and -margin <= point[0] < w+margin
                     and -margin <= point[1] < h+margin}
            if not nodes:
                continue
            count += 1
            for a, b in SLEAP_SKELETON_EDGES:
                if a in nodes and b in nodes:
                    cv2.line(image, nodes[a], nodes[b], line_color, max(2, round(2*self.overlay_scale)), cv2.LINE_AA)
            for node, point in nodes.items():
                cv2.circle(image, point, round((5 if node == 0 else 3)*self.overlay_scale), color, -1, cv2.LINE_AA)
        return count

    def _draw_text(self, image, text, point, color, *, scale=0.65, thickness=2):
        outline = max(1, round(thickness*self.overlay_scale)) + max(2, round(2*self.overlay_scale))
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                 scale*self.overlay_scale, outline)
        pad = max(3, round(3*self.overlay_scale))
        image_h, image_w = image.shape[:2]
        x, y = point
        step = height + baseline + 2*pad
        offsets = [(dx, dy*step) for dx in (0, -width-pad, width+pad) for dy in range(-12, 13)]
        offsets.sort(key=lambda offset: offset[0]**2 + offset[1]**2)
        candidates = []
        for dx, dy in offsets:
            px = int(np.clip(x+dx, pad, max(pad, image_w-width-pad)))
            py = int(np.clip(y+dy, height+pad, max(height+pad, image_h-baseline-pad)))
            box = (px-pad, py-height-pad, px+width+pad, py+baseline+pad)
            overlap = sum(max(0, min(box[2], b[2])-max(box[0], b[0])) *
                          max(0, min(box[3], b[3])-max(box[1], b[1])) for b in self._label_boxes)
            candidates.append((overlap, (px-x)**2+(py-y)**2, px, py, box))
            if overlap == 0:
                break
        _, _, px, py, box = min(candidates)
        self._label_boxes.append(box)
        if ("HIT" in text or "selected" in text) and (px-x)**2+(py-y)**2 > step**2:
            endpoint = (int(np.clip(x, box[0], box[2])), int(np.clip(y, box[1], box[3])))
            cv2.line(image, point, endpoint, color, max(1, round(self.overlay_scale)), cv2.LINE_AA)
        super()._draw_text(image, text, (px, py), color, scale=scale, thickness=thickness)

    def draw_track_labels(self, image, points, spec):
        if self.show_sleep_var.get():
            return super().draw_aruco_and_sleep(image, points, spec)
        if self.show_aruco_var.get():
            s = self.overlay_scale
            for point in points:
                center = (round(point.x), round(point.y))
                color = color_for_id(point.track_id)
                cv2.circle(image, center, round(11*s), color, max(2, round(3*s)), cv2.LINE_AA)
                self._draw_text(image, f"T{point.track_id}",
                                (round(center[0]+13*s), round(center[1]-11*s)), color, scale=.72)
        return dict(sleep=0, wake=0, unknown=0)

    def draw_aruco_and_sleep(self, image, points, spec):
        counts = self.draw_track_labels(image, points, spec)
        visible_ids = tuple(sorted(p.track_id for p in points))
        key = (self.current_frame, visible_ids)
        if key != self.distance_frame_key:
            local = self.current_frame - self.review.start
            poses = {int(ant): self.review.xy[local, i] for i, ant in enumerate(self.review.ids) if ant in visible_ids}
            self.pair_distances = skeleton_pair_distances(poses, mm_per_pixel=self.review.manifest["mm_per_pixel"],
                                                        edges=SLEAP_SKELETON_EDGES)
            self.distance_frame_key = key
        try:
            threshold = validate_distance(self.distance_var.get())
            self.distance_error.set("")
        except ValueError as exc:
            threshold = None
            self.distance_error.set(str(exc))
        keep = {f"{row.ant_a}:{row.ant_b}" for row in self.pair_distances}
        for iid in self.pair_table.get_children():
            if iid not in keep:
                self.pair_table.delete(iid)
        hits = 0
        unknown = 0
        for index, row in enumerate(self.pair_distances):
            known = np.isfinite(row.distance_mm)
            hit = threshold is not None and row.is_hit(threshold)
            hits += int(hit)
            unknown += int(not known)
            iid = f"{row.ant_a}:{row.ant_b}"
            state = "HIT" if hit else "no pose" if not known else "-"
            values = (f"{row.ant_a}-{row.ant_b}", f"{row.distance_mm:.4f}" if known else "-", state)
            tags = ("hit",) if hit else ()
            if self.pair_table.exists(iid):
                self.pair_table.item(iid, values=values, tags=tags)
                self.pair_table.move(iid, "", index)
            else:
                self.pair_table.insert("", index, iid=iid, values=values, tags=tags)
            selected = (row.ant_a, row.ant_b) == self.selected_pair
            if not known or not (hit or selected):
                continue
            a, b = self.pixel(row.point_a), self.pixel(row.point_b)
            if a is None or b is None:
                continue
            color = (255, 220, 20) if hit else (0, 180, 255)
            cv2.line(image, a, b, color, max(2, round(3*self.overlay_scale)), cv2.LINE_AA)
            for point in (a, b):
                cv2.circle(image, point, round(6*self.overlay_scale), color, -1, cv2.LINE_AA)
            midpoint = tuple((np.array(a)+b)//2)
            self._draw_text(image, f"{row.ant_a}-{row.ant_b} {'HIT' if hit else 'selected'} {row.distance_mm:.3f}mm",
                            (midpoint[0], midpoint[1]+round(16*self.overlay_scale)), color, scale=0.5)
        threshold_text = f"Distance <= {threshold:g} mm" if threshold is not None else "Invalid distance"
        self.review_status.set(f"{threshold_text}\nVisible ants: {len(points)}\n"
                               f"Current hits: {hits} / {len(self.pair_distances)} pairs\nPairs without pose: {unknown}")
        return counts

    def close(self):
        if self.distance_job is not None:
            self.root.after_cancel(self.distance_job)
        super().close()


class OffsetVideoReader(OpenCvVideoReader):
    """An exactly trimmed local clip whose labels still use global frame numbers."""
    def __init__(self, path, start_frame):
        super().__init__(path)
        self.start_frame = start_frame

    def read_frame(self, frame):
        local = frame - self.start_frame
        if local < 0 or (self.frame_count() and local >= self.frame_count()):
            raise ValueError("Frame outside local video clip")
        return super().read_frame(local)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--block-dir", type=Path)
    parser.add_argument("--review-cache", type=Path)
    parser.add_argument("--video-clip", type=Path, help="Optional frame-exact local clip for fast random access")
    parser.add_argument("--clip-start-frame", type=int, help="Global frame corresponding to local clip frame zero")
    parser.add_argument("--distance-mm", type=float, default=DEFAULT_MICRO_DISTANCE_MM)
    parser.add_argument("--target", type=int, default=56)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--return-frame", type=int, default=677082)
    parser.add_argument("--start-frame", type=int, default=677286)
    parser.add_argument("--pre-seconds", type=float, default=40)
    parser.add_argument("--post-seconds", type=float, default=300)
    parser.add_argument("--playback-fps", type=float, default=10)
    args = parser.parse_args()
    try:
        validate_distance(args.distance_mm)
    except ValueError as exc:
        parser.error(str(exc))
    if args.video_clip is not None and args.clip_start_frame is None:
        parser.error("--video-clip requires --clip-start-frame")
    block = args.block_dir or args.video.parent
    root_path = args.review_cache or build_review_clip(
        block, side=args.side, target=args.target, return_frame=args.return_frame,
        start_frame=max(0, args.return_frame-round(args.pre_seconds*24)),
        stop_frame=args.return_frame+round(args.post_seconds*24),
    )
    review = FinishedTrackClip(root_path)
    print(f"Finished pose cache: {root_path}", flush=True)
    h_path = infer_hmats_path(block)
    if h_path is None:
        raise FileNotFoundError("Camera-to-panorama homography required for contact endpoints")
    with np.load(h_path) as h:
        panorama_h = h["H"][parse_camera_index(args.video)]
    root = tk.Tk()
    viewer = InteractionDebugViewer(root, review=review, video_path=args.video,
                           specs=resolve_chunk_specs(args.video, block / "data"),
                           motion_store=SleepMotionStore(block / "stitched" / "sleep_motion"),
                           params=SleepParameters(), start_frame=args.start_frame,
                           playback_fps=args.playback_fps, side=review.manifest["side"],
                           panorama_h=panorama_h, x_split_px=infer_x_split(block))
    viewer.frame_count = min(viewer.frame_count, review.stop)
    viewer.distance_var.set(str(args.distance_mm))
    if args.video_clip is not None:
        reader = OffsetVideoReader(args.video_clip, args.clip_start_frame)
        if args.clip_start_frame > review.start or (reader.frame_count() and args.clip_start_frame + reader.frame_count() < review.stop):
            reader.close()
            raise ValueError("Local video clip does not cover the contact review interval")
        viewer.video.close()
        viewer.video = reader
    root.mainloop()


if __name__ == "__main__":
    main()
