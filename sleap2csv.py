#!/usr/bin/env python3
"""
slp2csv.py
==========

Convert a SLEAP .slp prediction file to a flattened CSV with columns:
  Frame, Instance, Bodypoint, X, Y, Score_node

Writes: <output_folder>/<stem>_sleap_data.csv
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, Iterable, Iterator

import h5py
import numpy as np
import pandas as pd


# ─────────────────────────── progress ───────────────────────────
def _progress(it: Iterable, total: int | None = None, desc: str = "") -> Iterator:
    """
    Wrap an iterable with a progress bar if possible.
    - Uses tqdm if available.
    - Disables live bar when not attached to a TTY (e.g., Slurm logs) to avoid spam.
    - Falls back to the raw iterator.
    """
    try:
        from tqdm.auto import tqdm  # type: ignore
        # If not a TTY, use a minimal, non-dynamic bar; or disable via env.
        is_tty = sys.stderr.isatty()
        return tqdm(
            it,
            total=total,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            miniters=1,
            file=sys.stderr,
            disable=not is_tty,
        )
    except Exception:
        # No tqdm or it failed: just return the iterator.
        return iter(it)


# ─────────────────────────── helpers ────────────────────────────
def _parse_json(raw_bytes, name):
    if raw_bytes is None:
        return None
    if isinstance(raw_bytes, np.ndarray):
        if raw_bytes.size == 0:
            return None
        raw_bytes = b"".join(raw_bytes.flat)
    if isinstance(raw_bytes, bytes):
        txt = raw_bytes.decode("utf-8").strip()
    else:
        txt = str(raw_bytes).strip()
    if not txt:
        return None
    if name == "tracks_json":
        import re
        cleaned = re.sub(r'[\[\]"]', "", txt)
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        return [int(p) if p.isdigit() else p for p in parts]
    if name in ("suggestions_json", "videos_json"):
        return [json.loads(s) for s in txt.split()]
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return txt


def _h5_to_nested_dict(h5obj: h5py.Group | h5py.File) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, item in h5obj.items():
        if isinstance(item, h5py.Dataset):
            data = item[()]
            if key.endswith("_json") and isinstance(data, np.ndarray) and data.dtype.kind == "S":
                data = b"".join(data.flat)
            if key.endswith("_json"):
                data = _parse_json(data, key)
            out[key] = data
        else:
            out[key] = _h5_to_nested_dict(item)
    return out


# ─────────────────────── core conversion ────────────────────────
def import_slp(slp_path: pathlib.Path) -> Dict[str, Any]:
    if not slp_path.is_file():
        raise FileNotFoundError(f"{slp_path} is not a file")
    with h5py.File(slp_path, "r") as f:
        attr_json_raw = f.attrs.get("tracksjson")
        attr = json.loads(attr_json_raw) if attr_json_raw is not None else {}
        tracks = _h5_to_nested_dict(f)

    dset = {
        "dir": str(slp_path.parent),
        "name": slp_path.stem,
        "ext": slp_path.suffix,
        "Attr": attr,
        "tracks": tracks,
    }
    dset["nFrame"] = len(tracks["frames"]["frame_idx"])
    dset["nAnimals"] = len(tracks["instances"]["instance_id"])
    dset["nNodes"] = len(attr.get("nodes", []))

    print(
        f" imported h5 file: {slp_path.name}"
        f"\n   # of instances: {dset['nAnimals']} ({dset['nFrame']} frames)",
        file=sys.stderr,
    )
    return dset


def flatten_data(dset: Dict[str, Any]) -> pd.DataFrame:
    tracks = dset["tracks"]

    frame_idx = tracks["frames"]["frame_idx"]
    instances = tracks["instances"]
    pred = tracks["pred_points"]

    n_instances = len(instances["frame_id"])
    num_rows = len(pred["x"])

    n_nodes = dset["nNodes"] if dset["nNodes"] > 0 else num_rows // n_instances

    frame_ids = frame_idx[instances["frame_id"]]
    point_starts = instances["point_id_start"]

    instance_indices = np.empty(num_rows, dtype=np.int32)
    frame_indices = np.empty(num_rows, dtype=np.int32)

    # Progress over instances
    for i in _progress(range(n_instances), total=n_instances, desc="Instances"):
        start = point_starts[i]
        end = point_starts[i + 1] if i < n_instances - 1 else num_rows
        instance_indices[start:end] = i
        frame_indices[start:end] = frame_ids[i]

    bodypoint_indices = np.tile(np.arange(n_nodes, dtype=np.int32), n_instances)

    df = pd.DataFrame(
        {
            "Frame": frame_indices,
            "Instance": instance_indices,
            "Bodypoint": bodypoint_indices,
            "X": pred["x"],
            "Y": pred["y"],
            "Score_node": pred["score"],
        }
    )

    df[["X", "Y", "Score_node"]] = (
        df[["X", "Y", "Score_node"]].astype(np.float32).round(1)
    )
    return df


def slp2csv(filename: str | pathlib.Path, output_folder: str | pathlib.Path) -> pathlib.Path:
    filename = pathlib.Path(filename)
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dset = import_slp(filename)
    print(" flattening data …", file=sys.stderr)
    flat = flatten_data(dset)

    csv_path = output_folder / f"{dset['name']}_sleap_data.csv"
    flat.to_csv(csv_path, index=False, float_format="%.1f")
    print(f" saved: {csv_path}", file=sys.stderr)
    return csv_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert SLEAP .slp → flat CSV")
    p.add_argument("slp_file", type=pathlib.Path, help="Input .slp file")
    p.add_argument("output_folder", type=pathlib.Path, help="Output folder for the CSV")
    args = p.parse_args()
    slp2csv(args.slp_file, args.output_folder)
