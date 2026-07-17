#!/usr/bin/env python3
"""Apply a trained sleep/wake classifier to one stitched track parquet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import joblib
except ModuleNotFoundError:
    joblib = None

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analysis.sleep_classifier_features import extract_track_features, load_speed_for_track


VALUE_TO_LABEL = {0: "wake", 1: "sleep"}
DEFAULT_MODEL_PATHS = (
    Path(
        "/bucket/ReiterU/Ants/basler/20260515/block02/stitched/sleep_crop_videos/"
        "random_100_600s_min75pct_480px_20260622_161247/"
        "sleep_wake_speed_only_classifier/sleep_random_forest.joblib"
    ),
    Path(
        "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/sleep_crop_videos/"
        "random_100_600s_min75pct_480px_20260622_161247/"
        "sleep_wake_speed_only_classifier/sleep_random_forest.joblib"
    ),
)
PORTABLE_MODEL_SUFFIX = "_portable.npz"


def parse_seconds_list(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("At least one window size is required")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seconds list: {value!r}") from exc


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        return Path(model_path)
    env_model = os.environ.get("SLEEP_MODEL", "").strip()
    if env_model:
        return Path(env_model)
    for candidate in DEFAULT_MODEL_PATHS:
        if candidate.exists():
            return candidate
    return DEFAULT_MODEL_PATHS[0]


def infer_speed_root(track: Path, speed_root: Path | None) -> Path | None:
    if speed_root is not None:
        return Path(speed_root)
    env_speed_root = os.environ.get("SLEEP_SPEED_ROOT", "").strip()
    if env_speed_root:
        return Path(env_speed_root)

    per_track_dir = Path(os.environ.get("PER_TRACK_DIR", track.parent))
    if per_track_dir.name == "per_track":
        return per_track_dir.parent / "speed_vectors"
    if track.parent.name == "per_track":
        return track.parent.parent / "speed_vectors"
    return None


def portable_model_path(path: Path) -> Path:
    if path.suffix.lower() == ".npz":
        return path
    return path.with_name(f"{path.stem}{PORTABLE_MODEL_SUFFIX}")


class PortableRandomForest:
    """Small numpy inference runtime for exported sklearn RF classifiers."""

    def __init__(
        self,
        *,
        classes: np.ndarray,
        imputer_statistics: np.ndarray,
        trees: list[dict[str, np.ndarray]],
    ) -> None:
        self.classes_ = np.asarray(classes)
        self.imputer_statistics_ = np.asarray(imputer_statistics, dtype=np.float32)
        self.trees = trees

    def _prepare_x(self, x) -> np.ndarray:
        if hasattr(x, "to_numpy"):
            arr = x.to_numpy(dtype=np.float32, copy=True)
        else:
            arr = np.asarray(x, dtype=np.float32).copy()
        bad = ~np.isfinite(arr)
        if bad.any():
            fill = self.imputer_statistics_.astype(np.float32, copy=True)
            fill[~np.isfinite(fill)] = 0.0
            arr[bad] = np.take(fill, np.nonzero(bad)[1])
        return arr

    def predict_proba(self, x) -> np.ndarray:
        arr = self._prepare_x(x)
        n_rows = arr.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_rows, n_classes), dtype=np.float32)
        if n_rows == 0:
            return proba

        for tree in self.trees:
            node = np.zeros(n_rows, dtype=np.int32)
            children_left = tree["children_left"]
            children_right = tree["children_right"]
            feature = tree["feature"]
            threshold = tree["threshold"]
            while True:
                left = children_left[node]
                active = left >= 0
                if not active.any():
                    break
                active_idx = np.flatnonzero(active)
                active_node = node[active_idx]
                active_feature = feature[active_node]
                values = arr[active_idx, active_feature]
                go_left = values <= threshold[active_node]
                node[active_idx] = np.where(
                    go_left,
                    children_left[active_node],
                    children_right[active_node],
                )
            proba += tree["proba"][node]
        proba /= max(len(self.trees), 1)
        return proba

    def predict(self, x) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]


def load_portable_model(path: Path) -> tuple[PortableRandomForest, dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"].item()))
    n_trees = int(data["n_trees"])
    trees = []
    for index in range(n_trees):
        trees.append(
            {
                "children_left": data[f"tree_{index}_children_left"].astype(np.int32, copy=False),
                "children_right": data[f"tree_{index}_children_right"].astype(np.int32, copy=False),
                "feature": data[f"tree_{index}_feature"].astype(np.int32, copy=False),
                "threshold": data[f"tree_{index}_threshold"].astype(np.float32, copy=False),
                "proba": data[f"tree_{index}_proba"].astype(np.float32, copy=False),
            }
        )
    model = PortableRandomForest(
        classes=data["classes"],
        imputer_statistics=data["imputer_statistics"],
        trees=trees,
    )
    return model, metadata


def load_model_bundle(path: Path) -> tuple[object, dict[str, object]]:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return load_portable_model(path)
    if joblib is None:
        portable_path = portable_model_path(path)
        if portable_path.exists():
            return load_portable_model(portable_path)
        raise ModuleNotFoundError(
            "joblib is not installed and no portable model sidecar was found: "
            f"{portable_path}"
        )
    try:
        bundle = joblib.load(path)
    except ModuleNotFoundError:
        portable_path = portable_model_path(path)
        if portable_path.exists():
            return load_portable_model(portable_path)
        raise
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], dict(bundle.get("metadata", {}))
    return bundle, {}


def final_estimator(model: object) -> object:
    steps = getattr(model, "steps", None)
    if steps:
        return steps[-1][1]
    return model


def class_indices(model: object) -> dict[int, int]:
    estimator = final_estimator(model)
    classes = list(getattr(estimator, "classes_", []))
    return {int(value): index for index, value in enumerate(classes)}


def feature_columns_from_model(model: object, metadata: dict[str, object]) -> list[str]:
    feature_cols = metadata.get("feature_columns")
    if feature_cols:
        return [str(col) for col in feature_cols]
    feature_cols = getattr(model, "feature_names_in_", None)
    if feature_cols is not None:
        return [str(col) for col in feature_cols]
    raise ValueError(
        "Model bundle does not include feature_columns metadata. "
        "Use a model trained by analysis/sleep_classifier.py."
    )


def feature_mode_from_metadata(metadata: dict[str, object], requested: str) -> str:
    if requested != "auto":
        return requested
    feature_set = str(metadata.get("feature_set", "all"))
    if feature_set == "speed_only":
        return "speed_only"
    if feature_set == "posture_motion":
        return "posture_motion"
    return "all"


def prediction_valid_mask(features: pd.DataFrame) -> np.ndarray:
    if "speed_mm_s" in features.columns:
        return pd.to_numeric(features["speed_mm_s"], errors="coerce").notna().to_numpy()
    if "posture_bp_speed_valid_frac" in features.columns:
        return (pd.to_numeric(features["posture_bp_speed_valid_frac"], errors="coerce") > 0).to_numpy()
    return np.ones(len(features), dtype=bool)


def prediction_table(
    features: pd.DataFrame,
    *,
    model: object,
    feature_cols: list[str],
    strict_features: bool,
) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in features.columns]
    if missing and strict_features:
        raise ValueError(f"Track features are missing trained columns: {missing}")
    for col in missing:
        features[col] = np.nan

    features = features.loc[prediction_valid_mask(features)].reset_index(drop=True)
    if features.empty:
        return pd.DataFrame()

    x = features[feature_cols]
    proba = model.predict_proba(x)
    indices = class_indices(model)
    classes = np.asarray(final_estimator(model).classes_)
    pred = classes[np.argmax(proba, axis=1)].astype(np.int8)

    wake_probability = np.zeros(len(features), dtype=np.float64)
    sleep_probability = np.zeros(len(features), dtype=np.float64)
    if 0 in indices:
        wake_probability = proba[:, indices[0]]
    if 1 in indices:
        sleep_probability = proba[:, indices[1]]

    keep_cols = [col for col in ["Frame", "track_name", "track_id", "side"] if col in features.columns]
    out = features[keep_cols].copy()
    out["wake_probability"] = wake_probability.astype(np.float32)
    out["sleep_probability"] = sleep_probability.astype(np.float32)
    out["predicted_label_value"] = pred
    out["predicted_label"] = [VALUE_TO_LABEL.get(int(value), str(int(value))) for value in pred]
    return out.sort_values("Frame", kind="mergesort").reset_index(drop=True)


def write_dense_vectors(predictions: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    frames = pd.to_numeric(predictions["Frame"], errors="coerce").to_numpy(np.float64)
    valid_frame = np.isfinite(frames)
    if not valid_frame.any():
        raise ValueError("No finite prediction frames")

    frame_i = frames[valid_frame].round().astype(np.int64)
    frame_min = int(frame_i.min())
    frame_max = int(frame_i.max())
    n_frames = frame_max - frame_min + 1
    idx = frame_i - frame_min

    sleep_probability = np.full(n_frames, np.nan, dtype=np.float32)
    wake_probability = np.full(n_frames, np.nan, dtype=np.float32)
    predicted_sleep = np.full(n_frames, -1, dtype=np.int8)

    sleep_values = predictions.loc[valid_frame, "sleep_probability"].to_numpy(np.float32)
    wake_values = predictions.loc[valid_frame, "wake_probability"].to_numpy(np.float32)
    pred_values = predictions.loc[valid_frame, "predicted_label_value"].to_numpy(np.int8)
    sleep_probability[idx] = sleep_values
    wake_probability[idx] = wake_values
    predicted_sleep[idx] = pred_values

    sleep_path = out_dir / "sleep_probability_f4.npy"
    wake_path = out_dir / "wake_probability_f4.npy"
    pred_path = out_dir / "predicted_sleep_i1.npy"
    np.save(sleep_path, sleep_probability)
    np.save(wake_path, wake_probability)
    np.save(pred_path, predicted_sleep)

    observed = predicted_sleep >= 0
    sleep = predicted_sleep == 1
    wake = predicted_sleep == 0
    return {
        "sleep_probability_path": str(sleep_path),
        "wake_probability_path": str(wake_path),
        "predicted_sleep_path": str(pred_path),
        "prediction_values": {"missing_frame": -1, "wake": 0, "sleep": 1},
        "frame_min": frame_min,
        "frame_max": frame_max,
        "n_frames": int(n_frames),
        "n_predicted_frames": int(observed.sum()),
        "n_sleep_frames": int(sleep.sum()),
        "n_wake_frames": int(wake.sum()),
        "sleep_fraction_predicted_frames": float(sleep.sum() / observed.sum()) if observed.any() else None,
        "mean_sleep_probability": (
            float(np.nanmean(sleep_probability[observed])) if observed.any() else None
        ),
    }


def empty_prediction_table(track: Path) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Frame": pd.Series(dtype="int64"),
            "track_name": pd.Series(dtype="object"),
            "track_id": pd.Series(dtype="float64"),
            "side": pd.Series(dtype="object"),
            "wake_probability": pd.Series(dtype="float32"),
            "sleep_probability": pd.Series(dtype="float32"),
            "predicted_label_value": pd.Series(dtype="int8"),
            "predicted_label": pd.Series(dtype="object"),
        }
    ).assign(track_name=track.name)


def write_empty_outputs(
    *,
    track: Path,
    out_dir: Path,
    args: argparse.Namespace,
    model_metadata: dict[str, object],
    feature_mode: str,
    feature_cols: list[str],
    fps: float,
    mm_per_px: float,
    reason: str,
) -> None:
    predictions_path = out_dir / "sleep_predictions.parquet"
    bouts_path = out_dir / "sleep_bouts.parquet"
    empty_prediction_table(track).to_parquet(predictions_path, index=False)
    pd.DataFrame(
        {
            "frame_start": pd.Series(dtype="int64"),
            "frame_end": pd.Series(dtype="int64"),
            "predicted_label_value": pd.Series(dtype="int8"),
            "predicted_label": pd.Series(dtype="object"),
            "n_frames": pd.Series(dtype="int64"),
            "mean_sleep_probability": pd.Series(dtype="float32"),
            "median_sleep_probability": pd.Series(dtype="float32"),
        }
    ).to_parquet(bouts_path, index=False)

    speed, speed_meta = load_speed_for_track(args.speed_root, track) if args.speed_root is not None else (None, None)
    if speed is not None and speed_meta is not None:
        frame_min = int(speed_meta.get("frame_min", 0))
        n_frames = int(len(speed))
        frame_max = frame_min + n_frames - 1
    else:
        frame_min = None
        frame_max = None
        n_frames = 0

    sleep_path = out_dir / "sleep_probability_f4.npy"
    wake_path = out_dir / "wake_probability_f4.npy"
    pred_path = out_dir / "predicted_sleep_i1.npy"
    np.save(sleep_path, np.full(n_frames, np.nan, dtype=np.float32))
    np.save(wake_path, np.full(n_frames, np.nan, dtype=np.float32))
    np.save(pred_path, np.full(n_frames, -1, dtype=np.int8))

    metadata_out = {
        "status": "no_predictions",
        "reason": reason,
        "track_path": str(track),
        "track_name": track.name,
        "model_path": str(args.model),
        "model_feature_set": model_metadata.get("feature_set"),
        "feature_mode": feature_mode,
        "n_model_features": int(len(feature_cols)),
        "prediction_table_path": str(predictions_path),
        "bouts_path": str(bouts_path),
        "features_path": None,
        "speed_root": str(args.speed_root) if args.speed_root is not None else None,
        "fps": fps,
        "mm_per_px": mm_per_px,
        "speed_windows_seconds": [float(value) for value in args.speed_windows_seconds],
        "dtype_probability": "float32",
        "dtype_prediction": "int8",
        "sleep_probability_path": str(sleep_path),
        "wake_probability_path": str(wake_path),
        "predicted_sleep_path": str(pred_path),
        "prediction_values": {"missing_frame": -1, "wake": 0, "sleep": 1},
        "frame_min": frame_min,
        "frame_max": frame_max,
        "n_frames": n_frames,
        "n_predicted_frames": 0,
        "n_sleep_frames": 0,
        "n_wake_frames": 0,
        "sleep_fraction_predicted_frames": None,
        "mean_sleep_probability": None,
    }
    (out_dir / "sleep_prediction_metadata.json").write_text(json.dumps(metadata_out, indent=2) + "\n")
    print(f"Wrote empty sleep prediction outputs for {track.name}: {reason}")


def build_bouts(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    data = predictions.sort_values("Frame", kind="mergesort").reset_index(drop=True).copy()
    frames = pd.to_numeric(data["Frame"], errors="coerce").round().astype(np.int64)
    pred = pd.to_numeric(data["predicted_label_value"], errors="coerce").astype(np.int8)
    breaks = (frames.ne(frames.shift() + 1)) | (pred.ne(pred.shift()))
    data["_bout_id"] = breaks.cumsum()
    grouped = (
        data.groupby("_bout_id", sort=True, as_index=False)
        .agg(
            frame_start=("Frame", "first"),
            frame_end=("Frame", "last"),
            predicted_label_value=("predicted_label_value", "first"),
            predicted_label=("predicted_label", "first"),
            n_frames=("Frame", "size"),
            mean_sleep_probability=("sleep_probability", "mean"),
            median_sleep_probability=("sleep_probability", "median"),
        )
        .drop(columns=["_bout_id"])
    )
    return grouped.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=None, help="Input per-track parquet. Defaults to $TRACK_PATH.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to $TASK_OUTPUT_DIR.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Trained sleep_random_forest.joblib. Defaults to $SLEEP_MODEL, then the checked-in production model path.",
    )
    parser.add_argument(
        "--speed_root",
        type=Path,
        default=None,
        help="speed_vectors output folder. Defaults to $SLEEP_SPEED_ROOT, then sibling stitched/speed_vectors.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Defaults to model metadata, then 24.")
    parser.add_argument("--mm_per_px", type=float, default=None, help="Defaults to model metadata, then 0.016.")
    parser.add_argument(
        "--feature_mode",
        choices=("auto", "all", "posture_motion", "speed_only"),
        default="auto",
        help="Feature extractor to use. Default auto uses the model metadata feature_set.",
    )
    parser.add_argument(
        "--speed_windows_seconds",
        type=parse_seconds_list,
        default=(1.0, 5.0, 30.0),
        help="Comma-separated rolling windows used by feature extraction. Default: 1,5,30.",
    )
    parser.add_argument("--write_features", action="store_true", help="Also write the full feature table parquet.")
    parser.add_argument("--strict_features", action="store_true", help="Fail instead of filling missing features with NaN.")
    args = parser.parse_args()

    track = args.track or Path(os.environ["TRACK_PATH"])
    out_dir = args.out or Path(os.environ["TASK_OUTPUT_DIR"])
    args.model = resolve_model_path(args.model)
    args.speed_root = infer_speed_root(track, args.speed_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, metadata = load_model_bundle(args.model)
    feature_cols = feature_columns_from_model(model, metadata)
    fps = float(args.fps if args.fps is not None else metadata.get("fps", 24.0))
    mm_per_px = float(args.mm_per_px if args.mm_per_px is not None else metadata.get("mm_per_px", 0.016))
    feature_mode = feature_mode_from_metadata(metadata, str(args.feature_mode))

    features = extract_track_features(
        track,
        speed_root=args.speed_root,
        fps=fps,
        mm_per_px=mm_per_px,
        speed_windows_seconds=tuple(args.speed_windows_seconds),
        feature_mode=feature_mode,
    )
    if features.empty:
        write_empty_outputs(
            track=track,
            out_dir=out_dir,
            args=args,
            model_metadata=metadata,
            feature_mode=feature_mode,
            feature_cols=feature_cols,
            fps=fps,
            mm_per_px=mm_per_px,
            reason="no_feature_rows",
        )
        return

    predictions = prediction_table(
        features,
        model=model,
        feature_cols=feature_cols,
        strict_features=bool(args.strict_features),
    )
    if predictions.empty:
        write_empty_outputs(
            track=track,
            out_dir=out_dir,
            args=args,
            model_metadata=metadata,
            feature_mode=feature_mode,
            feature_cols=feature_cols,
            fps=fps,
            mm_per_px=mm_per_px,
            reason="no_valid_prediction_rows",
        )
        return
    predictions_path = out_dir / "sleep_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    bouts = build_bouts(predictions)
    bouts_path = out_dir / "sleep_bouts.parquet"
    bouts.to_parquet(bouts_path, index=False)

    dense_metadata = write_dense_vectors(predictions, out_dir)
    features_path = None
    if args.write_features:
        features_path = out_dir / "sleep_features.parquet"
        features.to_parquet(features_path, index=False)

    metadata_out = {
        "track_path": str(track),
        "track_name": track.name,
        "track_id": int(predictions["track_id"].dropna().iloc[0]) if "track_id" in predictions and predictions["track_id"].notna().any() else None,
        "side": str(predictions["side"].dropna().iloc[0]) if "side" in predictions and predictions["side"].notna().any() else None,
        "model_path": str(args.model),
        "model_feature_set": metadata.get("feature_set"),
        "feature_mode": feature_mode,
        "n_model_features": int(len(feature_cols)),
        "prediction_table_path": str(predictions_path),
        "bouts_path": str(bouts_path),
        "features_path": str(features_path) if features_path is not None else None,
        "speed_root": str(args.speed_root) if args.speed_root is not None else None,
        "fps": fps,
        "mm_per_px": mm_per_px,
        "speed_windows_seconds": [float(value) for value in args.speed_windows_seconds],
        "dtype_probability": "float32",
        "dtype_prediction": "int8",
        **dense_metadata,
    }
    (out_dir / "sleep_prediction_metadata.json").write_text(json.dumps(metadata_out, indent=2) + "\n")
    print(
        f"Wrote {predictions_path} ({len(predictions):,} frames; "
        f"sleep fraction={metadata_out['sleep_fraction_predicted_frames']})"
    )


if __name__ == "__main__":
    main()
