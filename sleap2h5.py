#!/usr/bin/env python3
"""
sleap2h5.py
===========

Convert a SLEAP .slp prediction file to a flattened HDF5 dataset with columns:
  Frame, Instance, Bodypoint, X, Y, Score_node

Writes: <output_folder>/<stem>_sleap_data.h5
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import h5py
import numpy as np
import pandas as pd

from sleap2csv import flatten_data, import_slp


def _to_structured_array(df: pd.DataFrame) -> np.ndarray:
    records = df.to_records(index=False)
    for name in records.dtype.names:
        if records.dtype[name].kind == "O":
            raise TypeError(f"Column {name} has object dtype; cannot store in HDF5 cleanly")
    return records


def slp2h5(filename: str | pathlib.Path, output_folder: str | pathlib.Path) -> pathlib.Path:
    filename = pathlib.Path(filename)
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dset = import_slp(filename)
    print(" flattening data …", file=sys.stderr)
    flat = flatten_data(dset)

    h5_path = output_folder / f"{dset['name']}_sleap_data.h5"
    records = _to_structured_array(flat)

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["source_file"] = str(filename)
        h5.attrs["frame_count"] = int(dset["nFrame"])
        h5.attrs["instance_count"] = int(dset["nAnimals"])
        h5.attrs["node_count"] = int(dset["nNodes"])
        h5.create_dataset(
            "sleap_data",
            data=records,
            compression="gzip",
            shuffle=True,
            chunks=True,
        )

    print(f" saved: {h5_path}", file=sys.stderr)
    return h5_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert SLEAP .slp → flat HDF5")
    p.add_argument("slp_file", type=pathlib.Path, help="Input .slp file")
    p.add_argument("output_folder", type=pathlib.Path, help="Output folder for the HDF5 file")
    args = p.parse_args()
    slp2h5(args.slp_file, args.output_folder)
