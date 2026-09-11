"""Bounded real-video smoke test of lossless ArUco export, mapping and tracking.

Creates a NEW output directory; never modifies production detections/tracks.
Frames in test files are rebased to zero; summary.json records the source span.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from detection_pipeline.scripts import run_aruco_mp as detector
from detection_pipeline.scripts import aruco_output
from tracking.colony import map_combine as mapper
from tracking.colony.panorama_io import load_aruco_pkl, load_sleap_pkl
from tracking.core.tracking_utils import get_complete_tracks


def fingerprint(path):
    info = path.stat()
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=3)
    parser.add_argument("--start-frame", type=int, default=600)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--snapshot-frame", type=int, default=637)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--x-threshold", type=float, default=2500.)
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1 or args.start_frame < 0:
        parser.error("Require count/workers >= 1 and start-frame >= 0")
    if not args.start_frame <= args.snapshot_frame < args.start_frame + args.count <= 43200:
        parser.error("Snapshot and replay range must be inside chunk000")
    root, out = args.dataset, args.output
    video, = root.glob(f"cam{args.camera:02d}_*.mkv")
    raw, = (root / "data").glob(f"cam{args.camera:02d}_*_000_aruco_tracks.h5")
    sleap_path, = (root / "data").glob(f"cam{args.camera:02d}_*_000_sleap_data.h5")
    state = json.loads((root / "data/PIPELINE_STATE.json").read_text())
    if state["detection"].get("aruco_params"):
        raise ValueError("Recorded custom detector parameters need an explicit matching replay")
    dictionary = Path(state["detection"]["aruco_dict"])
    if not dictionary.is_file():
        dictionary = Path(str(root).split("/bucket/")[0] + str(dictionary))
    metadata = json.loads((root / "panorama_from_hmats_metadata.json").read_text())
    hmats = mapper.load_homographies(metadata["homographies"])
    H = hmats[args.camera-1]
    _, dict_size = detector.load_custom_aruco_dict(str(dictionary))
    source_before = [fingerprint(p) for p in (video, raw, sleap_path)]
    out.mkdir(parents=True, exist_ok=False)
    raw_out, panorama, tracks_out = [out / name for name in ("raw", "panorama", "tracks")]
    for directory in (raw_out, panorama, tracks_out):
        directory.mkdir()

    # Invoke the actual production worker on a bounded original-video range;
    # no transcoding, rescaling, or detection in the parent before cold fork.
    edges = np.linspace(args.start_frame, args.start_frame + args.count,
                        min(args.workers, args.count)+1, dtype=int)
    tasks = [(str(video), ("custom", str(dictionary)), int(a), int(b-a), False)
             for a, b in zip(edges[:-1], edges[1:])]
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        result = list(pool.map(detector._worker, tasks))
    assert [r[1] for r in result] == edges[1:].tolist(), "Worker failed to reach requested end frame"
    rows = [(f-args.start_frame, tag, x, y) for detections, _ in result for f, tag, x, y in detections]
    dense, confidence, expected = detector.pack_detections(rows, args.count, dict_size)
    name = raw.name.removesuffix("_aruco_tracks.h5")
    detector.save_aruco_outputs(raw_out, name, dense, confidence, expected, "both")
    saved_raw = raw_out / raw.name
    loaded, nframes = mapper._load_aruco_input_to_df_and_num_frames(saved_raw)
    pd.testing.assert_frame_equal(loaded, expected, check_dtype=False)
    assert nframes == args.count

    with h5py.File(raw) as f:
        original = f["aruco_tracks"][args.start_frame:args.start_frame+args.count]
    same_slots = np.all(original == dense, axis=2)
    presence_matches = np.array_equal(np.any(original != 0, axis=2), confidence > 0)
    coordinate_difference = float(np.linalg.norm(original-dense, axis=2).max())

    mapper.set_x_threshold(args.x_threshold)
    mapper.process_aruco_chunks(hmats, raw_out, panorama, "patched",
                               min_instance_frame_frac=0.)
    # Same decoded detections with the old last-write rule: isolates storage
    # changes from detector-version or parameter differences.
    legacy = expected.drop_duplicates(["Frame", "Instance"], keep="last").copy()
    legacy[["X", "Y"]] = mapper.apply_homography(legacy[["X", "Y"]].to_numpy(), H)
    legacy["Cam"] = args.camera-1
    mapper.split_and_write_with_num_frames_flat(legacy, panorama, "legacy_chunk000_aruco_panorama", args.count)

    with h5py.File(sleap_path) as f:
        source = f["sleap_data"]
        frames = source.fields("Frame")[:]
        indices = np.flatnonzero((frames >= args.start_frame) & (frames < args.start_frame+args.count))
        sleap = pd.DataFrame.from_records(source[indices])
    sleap["Frame"] -= args.start_frame
    sleap[["X", "Y"]] = mapper.apply_homography(sleap[["X", "Y"]].to_numpy(), H)
    sleap["Cam"] = args.camera-1
    mapper.split_and_write_flat(sleap, panorama, "patched_chunk000_sleap_panorama")

    per_frame = []
    totals = {}
    for side in ("left", "right"):
        skeleton_path, = panorama.glob(f"patched*sleap*_{side}*.pkl")
        skeletons = load_sleap_pkl(skeleton_path)
        totals[side] = {}
        for version in ("legacy", "patched"):
            tag_path, = panorama.glob(f"{version}*aruco*_{side}*.pkl")
            tags, count = load_aruco_pkl(tag_path)
            destination = tracks_out / f"{version}_{side}.parquet"
            get_complete_tracks(destination, tags, skeletons, num_frames=count, stream_output=True)
            tracks = pd.read_parquet(destination)
            id75 = tracks[tracks.TrackID == 75].drop_duplicates("Frame")
            tagged75 = tags[tags.Instance == 75]
            assert ((tagged75.X < args.x_threshold) == (side == "left")).all()
            tracked_frames = set(id75.Frame)
            detected_frames = set(tagged75.Frame)
            totals[side][version] = {"tag75_detected_frames": len(detected_frames),
                                     "tag75_tracked_frames": len(tracked_frames),
                                     "all_tag_detections": len(tags)}
            for frame in range(count):
                per_frame.append({"source_frame": frame+args.start_frame, "side": side,
                                  "version": version, "id75_detected": frame in detected_frames,
                                  "id75_tracked": frame in tracked_frames})
    frame_table = pd.DataFrame(per_frame)
    frame_table.to_csv(out / "id75_by_frame.csv", index=False)
    duplicate_rows = expected[expected.duplicated(["Frame", "Instance"], keep=False)].copy()
    duplicate_rows["source_frame"] = duplicate_rows.Frame + args.start_frame
    duplicate_rows.to_csv(out / "duplicate_detections.csv", index=False)
    snapshot = loaded[(loaded.Frame == args.snapshot_frame-args.start_frame) & (loaded.Instance == 75)]
    assert len(snapshot) == 2, "Known duplicate ID75 did not survive replay/export/load"
    snapshot_xy = mapper.apply_homography(snapshot[["X", "Y"]].to_numpy(), H)
    assert sorted(snapshot_xy[:, 0] < args.x_threshold) == [False, True]
    for side in ("left", "right"):
        row = frame_table.query("source_frame == @args.snapshot_frame and side == @side and version == 'patched'")
        assert row.id75_detected.all() and row.id75_tracked.all(), f"ID75 missing on {side} in snapshot"

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.snapshot_frame)
    ok, image = cap.read()
    cap.release()
    assert ok
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"height_ratios": [1.4, 1]})
    for (_, row), xy in zip(snapshot.iterrows(), snapshot_xy):
        side = "left" if xy[0] < args.x_threshold else "right"
        column = 0 if side == "left" else 1
        ax = axes[0, column]
        x, y = float(row.X), float(row.Y)
        x0, y0 = max(0, int(x)-180), max(0, int(y)-180)
        crop = image[y0:int(y)+180, x0:int(x)+180]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.scatter([x-x0], [y-y0], s=500, facecolors="none", edgecolors="#00ff88", linewidths=2)
        old = original[args.snapshot_frame-args.start_frame, 75]
        retained = np.linalg.norm(old-[x, y]) < 1.
        ax.set_title(f"{side.capitalize()} colony: ID75\nPreviously {'saved' if retained else 'overwritten'}; now saved and tracked")
        ax.set_axis_off()
    for column, side in enumerate(("left", "right")):
        ax = axes[1, column]
        for version, color, offset in [("legacy", "#c35c32", 0.), ("patched", "#2478b4", .04)]:
            table = frame_table.query("side == @side and version == @version")
            ax.step(table.source_frame / fps, table.id75_detected.astype(float)+offset,
                    where="mid", color=color, label=f"{version}: tag saved")
        ax.set(ylim=(-.1, 1.2), yticks=[0, 1], yticklabels=["No", "Yes"],
               xlabel="Seconds from recording start", title=f"{side.capitalize()} colony: ID75 detection retained")
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Camera {args.camera:02d}: duplicate-ID export fix\n"
                 f"Original frames {args.start_frame}–{args.start_frame+args.count-1}; crops at frame {args.snapshot_frame}")
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=160, facecolor="white")
    plt.close(fig)
    assert source_before == [fingerprint(p) for p in (video, raw, sleap_path)], "A source file changed during validation"
    summary = {"status": "passed", "opencv_version": cv2.__version__, "production_opencv_version": "4.9.0",
               "camera": args.camera, "source_start_frame": args.start_frame, "count": args.count,
               "fps": fps, "snapshot_frame": args.snapshot_frame, "dictionary": str(dictionary),
               "homographies": metadata["homographies"], "x_threshold": args.x_threshold,
               "sources": source_before, "lossless_detection_count": len(expected),
               "legacy_detection_count": len(legacy), "additional_detections": len(expected)-len(legacy),
               "frames_with_duplicate_ids": int(duplicate_rows.Frame.nunique()),
               "duplicated_ids": sorted(int(x) for x in duplicate_rows.Instance.unique()),
               "legacy_replay_slots_different_from_saved": int((~same_slots).sum()),
               "legacy_replay_detection_presence_matches_saved": presence_matches,
               "legacy_replay_max_coordinate_difference_px": coordinate_difference,
               "id75_results": totals,
               "scope": "One camera, short selected interval; not full-window prevalence or multicamera tracking benchmark.",
               "code_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(detector.__file__), Path(aruco_output.__file__),
                                         Path(mapper.__file__), REPO / "tracking/core/tracking_utils.py",
                                         Path(__file__)]}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
