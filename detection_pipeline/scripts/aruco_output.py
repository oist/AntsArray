"""Lossless ArUco output shared by the serial and multiprocessing detectors."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
import pandas as pd


SCHEMA_VERSION = 2


def pack_detections(dets, num_frames, dictionary_size):
    """Return legacy dense summaries AND every raw detection as a separate row.

    Instance is the tag ID, not a row identifier. Never deduplicate these rows
    by Frame/Instance or Frame/Instance/Cam before spatial splitting/tracking.
    """
    df = pd.DataFrame(dets, columns=["Frame", "Instance", "X", "Y"]).astype(
        {"Frame": "int32", "Instance": "int32", "X": "float32", "Y": "float32"}
    )
    df["Confidence"] = np.ones(len(df), dtype=np.float32)
    tracks = np.zeros((num_frames, dictionary_size, 2), dtype=np.float32)
    confidences = np.zeros((num_frames, dictionary_size), dtype=np.float32)
    # Explicit last-write behavior for backwards compatibility ONLY.
    for frame, marker, x, y in df[["Frame", "Instance", "X", "Y"]].itertuples(index=False, name=None):
        tracks[frame, marker] = (x, y)
        confidences[frame, marker] = 1.0
    return tracks, confidences, df


@contextmanager
def _atomic_output(path):
    """Publish only completed files; don't leave a truncated previous output."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        yield temporary
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_aruco_outputs(out_dir, name, tracks, confidences, detections, output_format="h5"):
    """Write native lossless records and optional pandas HDF/CSV sidecars.

    The native records are authoritative, independent of pandas/PyTables.
    Explicit frame count includes trailing frames without detections. Legacy
    dense datasets retain their original names and shapes for compatibility.
    """
    if output_format not in {"csv", "h5", "both"}:
        raise ValueError(f"Unsupported output format: {output_format}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_frames = int(tracks.shape[0])
    duplicate_count = int(detections.duplicated(["Frame", "Instance"]).sum())
    raw_h5 = out_dir / f"{name}_aruco_tracks.h5"
    with _atomic_output(raw_h5) as temporary:
        with h5py.File(temporary, "w") as h:
            h.attrs["aruco_detection_schema_version"] = SCHEMA_VERSION
            h.attrs["num_frames"] = num_frames
            h.attrs["dense_duplicate_instances_dropped"] = duplicate_count
            h.create_dataset("aruco_detections", data=detections.to_records(index=False),
                             compression="gzip", shuffle=True, chunks=True)
            dense = h.create_dataset("aruco_tracks", data=tracks, compression="gzip", shuffle=True, chunks=True)
            dense.attrs["semantics"] = "Legacy last-detection-per-frame/ID summary; use aruco_detections."
            h.create_dataset("aruco_confidences", data=confidences, compression="gzip", shuffle=True, chunks=True)
    print(f"[INFO] Saved {len(detections)} lossless detections to: {raw_h5}", flush=True)
    if duplicate_count:
        print(f"[WARN] Legacy dense summary omits {duplicate_count} repeated-ID detections; "
              "all are preserved in aruco_detections.", flush=True)

    if output_format in ("csv", "both"):
        with _atomic_output(out_dir / f"{name}_aruco_detections.csv") as temporary:
            detections.to_csv(temporary, index=False)
    if output_format in ("h5", "both"):
        try:
            import tables  # noqa: F401
        except ImportError:
            print("[WARN] 'tables' not found; lossless detections remain available in the raw H5.", flush=True)
        else:
            with _atomic_output(out_dir / f"{name}_aruco_detections.h5") as temporary:
                with pd.HDFStore(temporary, mode="w", complevel=4, complib="zlib") as store:
                    # Empty table-format writes omit the key; fixed format keeps it.
                    store.put("detections", detections, format="fixed" if detections.empty else "table")
                    attrs = store.get_storer("detections").attrs
                    attrs.num_frames = num_frames
                    attrs.aruco_detection_schema_version = SCHEMA_VERSION
