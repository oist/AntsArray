"""Native-camera image overlays for reviewing a manual-cup tracking pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from camera_cal.panorama_from_hmats import _extract_frame_from_video_ffmpeg
from tracking.colony.map_combine import apply_homography


def make_snapshots(output, manifest_path, block):
    output, block = Path(output), Path(block)
    run = json.loads((output / "run.json").read_text())
    m = json.loads(Path(manifest_path).read_text())
    track = pd.read_parquet(output / "pilot_frames_mm.parquet")
    nodes = pd.read_parquet(output / "nodes_mm.parquet")
    tags = pd.read_parquet(output / "aruco_mm.parquet")
    poses = pd.read_parquet(output / "poses_mm.parquet")
    qualifications = pd.read_csv(output / "cup_qualification.csv")
    cups = [c for c in run["cups"] if c["cup_id"] in set(track.CupID)]
    fig, axes = plt.subplots(len(cups), 3, figsize=(18, 5 * len(cups)), squeeze=False, layout="constrained")
    cache = output / "snapshots"
    cache.mkdir(exist_ok=True)
    records = []
    for i, cup in enumerate(cups):
        t = track.loc[track.CupID.eq(cup["cup_id"])]
        for j, source in enumerate(["aruco_sleap", "sleap_bridge", "flag_evidence"]):
            ax = axes[i, j]
            if source == "flag_evidence":
                q = qualifications.loc[qualifications.CupID.eq(cup["cup_id"])].iloc[0]
                if not q.FlaggedMultiAnt:
                    ax.set_axis_off()
                    continue
                frame = int(q.StopFrame)
                observed = poses.loc[poses.Frame.eq(frame) & poses.CupID.eq(cup["cup_id"])]
                a = tags.loc[tags.Frame.eq(frame) & tags.CupID.eq(cup["cup_id"])]
                foreign = a.loc[a.Instance.ne(int(q.ArucoID))]
                possible = observed.loc[observed.Cam.isin(foreign.Cam)] if len(foreign) else observed
                if possible.empty:
                    possible = observed
                camera = int(possible.sort_values(["NNodes", "MedianScore"], ascending=False).iloc[0].Cam)
                key = -1
                chosen = t.loc[t.Frame.eq(frame)].iloc[0]
            else:
                candidate = t.loc[t.PositionSource.eq(source)]
                if candidate.empty:
                    ax.text(.5, .5, f"{cup['name']}: no {source} frame", ha="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                # Sample a match at 30 s and the middle of the longest bridge.
                if source == "sleap_bridge":
                    from tracking.single_ant.manual_cup_report import intervals
                    windows = intervals(t.PositionSource.eq(source))
                    lo, hi = max(windows, key=lambda w: w[1] - w[0])
                    fr = int(t.iloc[(lo + hi) // 2].Frame)
                    chosen = t.loc[t.Frame.eq(fr)].iloc[0]
                else:
                    chosen = candidate.iloc[np.argmin(abs(candidate.Frame.to_numpy() - run["start_frame"] - 30 * run["fps"]))]
                frame, camera, key = int(chosen.Frame), int(chosen.SleapCam), int(chosen.PoseKey)
            f = next(f for f in m["files"] if f["camera_id"] == camera and
                     f["frame_start"] <= frame < f["frame_start"] + f["frames"])
            stem = re.sub(r"_\d{3}_sleap_data\.h5$", "", Path(f["sleap"]).name)
            video = block / (stem + ".mkv")
            saved = cache / f"{cup['name']}_cam{camera + 1:02d}_frame{frame:06d}.png"
            if saved.exists():
                image = plt.imread(saved)
            else:
                print(f"SNAPSHOT {cup['name']} frame {frame} camera {camera + 1}", flush=True)
                image = _extract_frame_from_video_ffmpeg(video, frame_index=frame, fps=run["fps"])
                import cv2
                if not cv2.imwrite(str(saved), image):
                    raise IOError(saved)
            inverse = np.linalg.inv(m["homographies"][camera])

            def project(mm):
                return apply_homography(np.asarray(mm).reshape(-1, 2) * m["board_px_per_mm"], inverse)

            circle = np.asarray(cup["center_mm"]) + cup["radius_mm"] * np.c_[
                np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100))]
            boundary = project(circle)
            ax.imshow(image, cmap="gray", vmin=0 if image.max() > 1 else None, vmax=255 if image.max() > 1 else None)
            ax.plot(*boundary.T, color="#e8cc21", lw=.8, label="manual cup")
            if source == "flag_evidence":
                keys = poses.loc[poses.CupID.eq(cup["cup_id"]) & poses.Frame.eq(frame) & poses.Cam.eq(camera), "PoseKey"]
                p = nodes.loc[nodes.PoseKey.isin(keys)].sort_values(["PoseKey", "Bodypoint"])
            else:
                p = nodes.loc[nodes.PoseKey.eq(key)].sort_values("Bodypoint")
            xy = project(p[["X", "Y"]].to_numpy())
            color = "#00c5ff" if source == "aruco_sleap" else "#ff9424"
            ax.scatter(*xy.T, s=25, edgecolors="black", linewidths=.35, color=color,
                       label="observed posture (not tracked)" if source == "flag_evidence" else "selected posture")
            for bp, point in zip(p.Bodypoint, xy):
                ax.annotate(str(bp), point, xytext=(3, 3), textcoords="offset points", color=color, fontsize=7)
            a = tags.loc[tags.Frame.eq(frame) & tags.CupID.eq(cup["cup_id"]) & tags.Cam.eq(camera)]
            if len(a):
                tagxy = project(a[["X", "Y"]].to_numpy())
                ax.scatter(*tagxy.T, marker="s", s=80, facecolors="none", edgecolors="#21fb4f", lw=1.5, label="raw ArUco")
                for raw, point in zip(a.Instance, tagxy):
                    ax.annotate(f"ID {raw}", point, xytext=(8, -12), textcoords="offset points", color="#21fb4f", fontsize=8)
            alternatives = poses.loc[poses.Frame.eq(frame) & poses.CupID.eq(cup["cup_id"])
                                     & poses.Cam.eq(camera) & poses.PoseKey.ne(key)]
            if len(alternatives) and source != "flag_evidence":
                ax.scatter(*project(alternatives[["X", "Y"]].to_numpy()).T, marker="x", color="red", s=50, label="other pose anchors")
            pad = .06 * np.ptp(boundary, axis=0)
            ax.set(xlim=(max(0, boundary[:, 0].min() - pad[0]), min(image.shape[1], boundary[:, 0].max() + pad[0])),
                   ylim=(min(image.shape[0], boundary[:, 1].max() + pad[1]), max(0, boundary[:, 1].min() - pad[1])),
                   title=f"{cup['name']} | ID {int(chosen.ArucoID)} | frame {frame} ({frame/run['fps']:.2f}s)\n"
                         + ["Nearest ArUco–SLEAP match", "Natural tag gap: observed SLEAP posture",
                            "STOP event: persistent extra ID + skeleton"][j])
            ax.set_axis_off()
            ax.legend(fontsize=6, loc="lower left")
            records.append(dict(cup=cup["name"], frame=frame, camera=camera, video=str(video),
                                source=source, pose_key=key, image=str(saved)))
    fig.suptitle("Native-camera frames — selected posture and raw tag detections\n"
                 "Yellow outline = manual cup; numbers = actual SLEAP bodypoints", fontsize=14)
    fig.savefig(output / "plots" / "native_camera_overlays.png", dpi=160)
    fig.savefig(output / "native_camera_overlays.pdf")
    plt.close(fig)
    from tracking.single_ant.manual_cup_tracking import write_json
    write_json(output / "snapshot_frames.json", records)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--block", type=Path, required=True)
    args = p.parse_args()
    make_snapshots(args.output, args.manifest, args.block)
