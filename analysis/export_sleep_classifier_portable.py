#!/usr/bin/env python3
"""Export a trained sklearn sleep classifier to a numpy-only inference file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analysis.compute_track_sleep_predictions import portable_model_path


def tree_proba(tree) -> np.ndarray:
    value = np.asarray(tree.value, dtype=np.float32)
    if value.ndim == 3:
        value = value[:, 0, :]
    row_sum = value.sum(axis=1, keepdims=True)
    return np.divide(value, row_sum, out=np.zeros_like(value), where=row_sum > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Input sleep_random_forest.joblib.")
    parser.add_argument("--out", type=Path, default=None, help="Output .npz. Default: *_portable.npz beside model.")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Expected a {'model': ..., 'metadata': ...} joblib bundle")

    model = bundle["model"]
    metadata = dict(bundle.get("metadata", {}))
    imputer = model.named_steps["imputer"]
    forest = model.named_steps["forest"]
    out = args.out or portable_model_path(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "n_trees": np.asarray(len(forest.estimators_), dtype=np.int32),
        "classes": np.asarray(forest.classes_),
        "imputer_statistics": np.asarray(imputer.statistics_, dtype=np.float32),
    }
    for index, estimator in enumerate(forest.estimators_):
        tree = estimator.tree_
        arrays[f"tree_{index}_children_left"] = np.asarray(tree.children_left, dtype=np.int32)
        arrays[f"tree_{index}_children_right"] = np.asarray(tree.children_right, dtype=np.int32)
        arrays[f"tree_{index}_feature"] = np.asarray(tree.feature, dtype=np.int32)
        arrays[f"tree_{index}_threshold"] = np.asarray(tree.threshold, dtype=np.float32)
        arrays[f"tree_{index}_proba"] = tree_proba(tree)

    np.savez_compressed(out, **arrays)
    print(f"Wrote portable sleep classifier: {out}")


if __name__ == "__main__":
    main()
