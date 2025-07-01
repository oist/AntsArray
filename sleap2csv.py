#!/usr/bin/env python3
"""
slp2csv.py
==========

Convert a SLEAP .slp prediction file to a flattened CSV with columns

    Frame, Instance, Bodypoint, X, Y, Score_node

The logic mirrors the original MATLAB implementation.

-------------
Dependencies
-------------
h5py   >= 3.8
numpy  >= 1.21
pandas >= 1.5

Install (conda or pip):

    conda install h5py pandas numpy
        –– or ––
    pip install h5py pandas numpy

-------------
Usage example
-------------
$ python slp2csv.py /path/to/file.slp  /path/to/output_folder
# → writes /path/to/output_folder/file.csv
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd


# ─────────────────────────── helpers ────────────────────────────
def _parse_json(raw_bytes, name):
    """
    Decode *_json datasets robustly, handling:
      - None
      - empty numpy arrays
      - non-empty numpy array of byte-strings
      - plain bytes
    """
    # 1) Nothing there?
    if raw_bytes is None:
        return None
    if isinstance(raw_bytes, np.ndarray):
        # if array has no elements, bail
        if raw_bytes.size == 0:
            return None
        # join an array of byte-strings into one bytes blob
        raw_bytes = b"".join(raw_bytes.flat)

    # 2) Now raw_bytes should be actual bytes or str
    if isinstance(raw_bytes, bytes):
        txt = raw_bytes.decode("utf-8").strip()
    else:
        txt = str(raw_bytes).strip()

    if not txt:
        return None

    # 3) Type-specific parsing
    if name == "tracks_json":
        # strip brackets and quotes, split on commas
        import re
        cleaned = re.sub(r'[\[\]"]', "", txt)
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        # convert purely numeric tokens
        return [int(p) if p.isdigit() else p for p in parts]

    if name in ("suggestions_json", "videos_json"):
        # assume whitespace-separated JSON strings
        return [json.loads(s) for s in txt.split()]

    # fallback: try a straight JSON decode
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return txt


def _h5_to_nested_dict(h5obj: h5py.Group | h5py.File) -> Dict[str, Any]:
    """
    Recursively copy every group/dataset into a nested plain-Python dict.
    """
    out: Dict[str, Any] = {}
    for key, item in h5obj.items():
        if isinstance(item, h5py.Dataset):  # leaf
            data = item[()]
            if key.endswith("_json") and isinstance(data, np.ndarray) and data.dtype.kind == "S":
                # convert byte-string array → single bytes object
                data = b"".join(data.flat)
            if key.endswith("_json"):
                data = _parse_json(data, key)
            out[key] = data
        else:  # subgroup
            out[key] = _h5_to_nested_dict(item)
    return out


# ─────────────────────── core conversion ────────────────────────
def import_slp(slp_path: pathlib.Path) -> Dict[str, Any]:
    """
    Read an .slp file into a regular Python structure.
    """
    if not slp_path.is_file():
        raise FileNotFoundError(f"{slp_path} is not a file")
   # import pdb; pdb.set_trace()
    with h5py.File(slp_path, "r") as f:
        # root attributes (metadata)
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

    # derived counts (mirrors MATLAB logic)
    dset["nFrame"] = len(tracks["frames"]["frame_idx"])
    dset["nAnimals"] = len(tracks["instances"]["instance_id"])
    dset["nNodes"] = len(attr.get("nodes", []))

    print(
        f" imported h5 file: {slp_path.name}"
        f"\n   # of instances: {dset['nAnimals']} ({dset['nFrame']} frames)"
    )
    return dset


def flatten_data(dset: Dict[str, Any]) -> pd.DataFrame:
    """
    Re-implement MATLAB `flattenData` one-to-one.
    """
    tracks = dset["tracks"]

    frame_idx = tracks["frames"]["frame_idx"]            # shape (nFrame,)
    instances = tracks["instances"]
    pred = tracks["pred_points"]

    n_instances = len(instances["frame_id"])
    num_rows = len(pred["x"])

    # How many body-points per instance?
    n_nodes = (
        dset["nNodes"]
        if dset["nNodes"] > 0
        else num_rows // n_instances
    )

    # MATLAB adds 1 here to move into 1-based indexing; we keep 0-based
    frame_ids = frame_idx[instances["frame_id"]]

    point_starts = instances["point_id_start"]         # 0-based
    # Prepare output arrays
    instance_indices = np.empty(num_rows, dtype=np.int32)
    frame_indices = np.empty(num_rows, dtype=np.int32)

    # Fill instance/frame columns
    for i in range(n_instances):
        start = point_starts[i]
        end = point_starts[i + 1] if i < n_instances - 1 else num_rows
        instance_indices[start:end] = i               # 0-based
        frame_indices[start:end] = frame_ids[i]       # 0-based

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
    return df


def slp2csv(filename: str | pathlib.Path, output_folder: str | pathlib.Path) -> pathlib.Path:
    """
    End-to-end convenience wrapper: read, flatten, write CSV.
    """
    filename = pathlib.Path(filename)
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dset = import_slp(filename)
    flat = flatten_data(dset)

    csv_path = output_folder / f"{dset['name']}.csv"
    flat.to_csv(csv_path, index=False)
    print(f" saved: {csv_path}")
    return csv_path

def slp2h5(filename: str | pathlib.Path, output_folder: str | pathlib.Path) -> pathlib.Path:
    """
    Read, flatten, and save the result as a flat HDF5 (one dataset per column).
    """
    filename = pathlib.Path(filename)
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dset = import_slp(filename)
    flat = flatten_data(dset)

    h5_path = output_folder / f"{dset['name']}.h5"
    with h5py.File(h5_path, "w") as f:
        # Save each column as its own dataset
        for col in flat.columns:
            # cast pandas Series to numpy array
            arr = flat[col].to_numpy()
            f.create_dataset(col, data=arr, compression="gzip")
    print(f" saved: {h5_path}")
    return h5_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Convert SLEAP .slp → flat HDF5")
    p.add_argument("slp_file", type=pathlib.Path, help="Input .slp file")
    p.add_argument("output_folder", type=pathlib.Path, help="Where to place the .h5")
    args = p.parse_args()
    slp2h5(args.slp_file, args.output_folder)
