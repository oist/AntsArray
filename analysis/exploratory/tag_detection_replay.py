"""Replay a few raw frames without modifying detection or tracking outputs.

Checks the dense (frame, marker-ID) export's inability to retain two copies
of the same marker ID. Local OpenCV version is recorded explicitly.
"""
from concurrent.futures import ThreadPoolExecutor
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from detection_pipeline.scripts.run_aruco_mp import load_custom_aruco_dict, _build_detector
from analysis.exploratory.tag_detection_spatial import transform


def replay_camera(args):
    root, camera, frame, dictionary, H = args
    video = next(root.glob(f"cam{camera:02d}_*.mkv"))
    raw = next((root / "data").glob(f"cam{camera:02d}_*_000_aruco_tracks.h5"))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, im = cap.read()
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.release()
    if not ok or abs(pos-frame-1) > 1:
        raise ValueError(f"Cannot decode requested frame {frame}: {video}")
    d, _ = load_custom_aruco_dict(str(dictionary))
    corners, ids, _ = _build_detector(d).detectMarkers(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    ids = ids.ravel() if ids is not None else np.array([], int)
    centers = np.array([c[0].mean(axis=0) for c in corners]).reshape(-1, 2)
    with h5py.File(raw) as f:
        saved = f["aruco_tracks"][frame]
    valid = np.isfinite(saved).all(axis=1) & (saved != 0).any(axis=1)
    rows, crops = [], []
    for marker, xy in zip(ids, centers):
        sx, sy = saved[marker]
        duplicate = np.count_nonzero(ids == marker) > 1
        distance = np.linalg.norm(xy-[sx, sy]) if valid[marker] else np.inf
        px, py = transform(xy[None], H)[0]
        rows.append(dict(camera=camera, frame=frame, marker_id=int(marker), native_x=float(xy[0]), native_y=float(xy[1]),
                         panorama_x=float(px), panorama_y=float(py), duplicate_id=duplicate,
                         present_in_saved=bool(distance < 1), saved_distance_px=float(distance)))
        if duplicate:
            x, y = np.rint(xy).astype(int)
            crop = im[max(0,y-220):min(im.shape[0],y+220), max(0,x-220):min(im.shape[1],x+220)]
            crops.append((camera, int(marker), distance < 1, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    summary = dict(camera=camera, frame=frame, replay_markers=len(ids), saved_markers=int(valid.sum()),
                   replay_unique_ids=len(np.unique(ids)), duplicate_ids=int(sum(np.count_nonzero(ids == i)>1 for i in np.unique(ids))),
                   duplicate_instances_lost=sum(r["duplicate_id"] and not r["present_in_saved"] for r in rows),
                   matching_saved_instances=sum(r["present_in_saved"] for r in rows))
    print(summary, flush=True)
    return rows, summary, crops


def main():
    root = Path(sys.argv[1])
    cameras = [int(x) for x in sys.argv[2:]] or [3, 8, 13, 18, 23]
    out = root / "analysis_outputs" / "tag_detection_spatial"
    state = json.loads((root / "data" / "PIPELINE_STATE.json").read_text())
    if state["detection"].get("aruco_params"):
        raise ValueError("Explicit detector parameters require a matching replay implementation")
    dictionary = Path(str(root).split('/bucket/')[0] + state["detection"]["aruco_dict"])
    meta = json.loads((root / "panorama_from_hmats_metadata.json").read_text())
    hmats = np.load(meta["homographies"])["H"]
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(replay_camera, [(root, c, 637, dictionary, hmats[c-1]) for c in cameras]))
    pd.DataFrame([r for result in results for r in result[0]]).to_csv(out / "replayed_tag_instances.csv", index=False)
    pd.DataFrame([result[1] for result in results]).to_csv(out / "duplicate_id_replay_summary.csv", index=False)
    (out / "replay_settings.json").write_text(json.dumps(dict(opencv_version=cv2.__version__,
        production_module="opencv/4.9.0", parameters="run_aruco_mp._build_detector", dictionary=str(dictionary),
        frame=637, cameras=cameras, note="Selected snapshot, not an estimate of full-window overwrite frequency."), indent=2))
    crops = [item for result in results for item in result[2]][:16]
    if crops:
        fig, axes = plt.subplots(int(np.ceil(len(crops)/4)), 4, figsize=(13, 3.8*int(np.ceil(len(crops)/4))), squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, (camera, marker, retained, im) in zip(axes.flat, crops):
            ax.imshow(im)
            ax.set_title(f"Camera {camera:02d} • ID {marker}\n" + ("Saved" if retained else "Decoded in replay, absent from saved output"), fontsize=9)
        fig.suptitle("Duplicate IDs: two visible tags, only one output slot per camera/frame/ID\nSame dictionary and detector parameters; replay OpenCV version recorded in replay_settings.json")
        fig.tight_layout(rect=(0, 0, 1, .92), h_pad=2)
        fig.savefig(out / "05_duplicate_id_overwrite.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
