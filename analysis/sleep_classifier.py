#!/usr/bin/env python3
"""Train or apply a supervised sleep/wake classifier from labeled ant frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analysis.sleep_classifier_features import (
    default_feature_columns,
    extract_track_features,
    side_from_name,
    track_id_from_name,
)


LABEL_TO_VALUE = {"wake": 0, "sleep": 1}
VALUE_TO_LABEL = {0: "wake", 1: "sleep"}


def read_labels(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        labels = pd.read_csv(path)
    else:
        labels = pd.read_parquet(path)
    required = {"frame_start", "frame_end", "label"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"{path} missing label columns: {sorted(missing)}")
    labels = labels.copy()
    labels["label"] = labels["label"].astype(str).str.lower()
    labels = labels[labels["label"].isin(LABEL_TO_VALUE)].copy()
    labels["frame_start"] = pd.to_numeric(labels["frame_start"], errors="coerce")
    labels["frame_end"] = pd.to_numeric(labels["frame_end"], errors="coerce")
    labels = labels.dropna(subset=["frame_start", "frame_end"])
    starts = np.minimum(labels["frame_start"].to_numpy(np.int64), labels["frame_end"].to_numpy(np.int64))
    ends = np.maximum(labels["frame_start"].to_numpy(np.int64), labels["frame_end"].to_numpy(np.int64))
    labels["frame_start"] = starts
    labels["frame_end"] = ends
    labels["label_value"] = labels["label"].map(LABEL_TO_VALUE).astype(np.int8)
    labels["label_file"] = str(path)
    return labels.reset_index(drop=True)


def load_label_files(paths: list[Path]) -> pd.DataFrame:
    tables = [read_labels(path) for path in paths]
    if not tables:
        raise ValueError("No label files provided")
    labels = pd.concat(tables, ignore_index=True)
    if labels.empty:
        raise ValueError("No sleep/wake intervals found in label files")
    return labels


def find_track_path(row: pd.Series, per_track_dir: Path | None) -> Path:
    if "track_path" in row and pd.notna(row["track_path"]):
        path = Path(str(row["track_path"]))
        if path.exists():
            return path
    if "track_name" in row and pd.notna(row["track_name"]) and per_track_dir is not None:
        path = Path(per_track_dir) / str(row["track_name"])
        if path.exists():
            return path
    if "track_id" in row and per_track_dir is not None and pd.notna(row["track_id"]):
        track_id = int(row["track_id"])
        side = str(row.get("side", ""))
        matches = sorted(Path(per_track_dir).glob(f"TrackID_{track_id:04d}_*.parquet"))
        if side in {"left", "right"}:
            matches = [path for path in matches if path.stem.endswith(f"_{side}")]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not locate track parquet for label row: {row.to_dict()}")


def label_features_for_track(
    track_path: Path,
    labels: pd.DataFrame,
    *,
    speed_root: Path | None,
    fps: float,
    mm_per_px: float,
    context_seconds: float,
) -> pd.DataFrame:
    frame_min = int(labels["frame_start"].min() - round(float(context_seconds) * float(fps)))
    frame_max = int(labels["frame_end"].max() + round(float(context_seconds) * float(fps)))
    features = extract_track_features(
        track_path,
        speed_root=speed_root,
        fps=fps,
        mm_per_px=mm_per_px,
        frame_min=frame_min,
        frame_max=frame_max,
    )
    if features.empty:
        return features

    labeled_rows = []
    labels = labels.sort_values(["frame_start", "frame_end"], kind="mergesort").reset_index(drop=True)
    for order, label_row in labels.iterrows():
        mask = (
            (features["Frame"] >= int(label_row["frame_start"]))
            & (features["Frame"] <= int(label_row["frame_end"]))
        )
        if not mask.any():
            continue
        subset = features.loc[mask].copy()
        subset["label"] = str(label_row["label"])
        subset["label_value"] = int(label_row["label_value"])
        subset["_label_order"] = int(order)
        labeled_rows.append(subset)
    if not labeled_rows:
        return pd.DataFrame()
    out = pd.concat(labeled_rows, ignore_index=True)
    out = (
        out.sort_values(["Frame", "_label_order"], kind="mergesort")
        .drop_duplicates(["track_name", "Frame"], keep="last")
        .drop(columns=["_label_order"])
        .reset_index(drop=True)
    )
    return out


def build_training_table(
    labels: pd.DataFrame,
    *,
    per_track_dir: Path | None,
    speed_root: Path | None,
    fps: float,
    mm_per_px: float,
    context_seconds: float,
    max_rows_per_class: int | None,
    random_state: int,
) -> pd.DataFrame:
    labels = labels.copy()
    track_paths = []
    for _, row in labels.iterrows():
        track_paths.append(find_track_path(row, per_track_dir))
    labels["resolved_track_path"] = [str(path) for path in track_paths]

    tables = []
    for track_path_text, track_labels in labels.groupby("resolved_track_path", sort=True):
        track_path = Path(track_path_text)
        print(f"features: {track_path.name} ({len(track_labels)} intervals)")
        table = label_features_for_track(
            track_path,
            track_labels,
            speed_root=speed_root,
            fps=fps,
            mm_per_px=mm_per_px,
            context_seconds=context_seconds,
        )
        if not table.empty:
            tables.append(table)
    if not tables:
        raise ValueError("No labeled feature rows were produced")

    training = pd.concat(tables, ignore_index=True)
    if max_rows_per_class is not None:
        sampled = []
        for label_value, group in training.groupby("label_value", sort=True):
            if len(group) > int(max_rows_per_class):
                group = group.sample(n=int(max_rows_per_class), random_state=int(random_state))
            sampled.append(group)
        training = pd.concat(sampled, ignore_index=True).sort_values(["track_name", "Frame"], kind="mergesort")
    return training.reset_index(drop=True)


def split_train_test(training: pd.DataFrame, *, test_size: float, random_state: int):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    y = training["label_value"].to_numpy()
    groups = training["track_name"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2 and len(np.unique(y)) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
        train_idx, test_idx = next(splitter.split(training, y, groups))
    elif len(np.unique(y)) >= 2 and len(training) >= 10:
        train_idx, test_idx = train_test_split(
            np.arange(len(training)),
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=y,
        )
    else:
        train_idx = np.arange(len(training))
        test_idx = np.asarray([], dtype=np.int64)
    return train_idx, test_idx


def train_classifier(args: argparse.Namespace) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline

    labels = load_label_files(args.labels)
    training = build_training_table(
        labels,
        per_track_dir=args.per_track_dir,
        speed_root=args.speed_root,
        fps=float(args.fps),
        mm_per_px=float(args.mm_per_px),
        context_seconds=float(args.context_seconds),
        max_rows_per_class=args.max_rows_per_class,
        random_state=int(args.random_state),
    )
    feature_cols = default_feature_columns(training)
    if not feature_cols:
        raise ValueError("No numeric feature columns available")

    train_idx, test_idx = split_train_test(training, test_size=float(args.test_size), random_state=int(args.random_state))
    x_train = training.iloc[train_idx][feature_cols]
    y_train = training.iloc[train_idx]["label_value"].to_numpy()

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=int(args.n_estimators),
                    max_depth=args.max_depth,
                    min_samples_leaf=int(args.min_samples_leaf),
                    class_weight="balanced",
                    random_state=int(args.random_state),
                    n_jobs=int(args.n_jobs),
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    args.out.mkdir(parents=True, exist_ok=True)
    training_path = args.out / "training_features.parquet"
    training.to_parquet(training_path, index=False)

    forest = model.named_steps["forest"]
    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False, kind="mergesort")
    importances.to_csv(args.out / "feature_importance.csv", index=False)

    report_lines = []
    if len(test_idx) > 0:
        x_test = training.iloc[test_idx][feature_cols]
        y_test = training.iloc[test_idx]["label_value"].to_numpy()
        pred = model.predict(x_test)
        report_lines.append("Held-out report:\n")
        report_lines.append(classification_report(y_test, pred, target_names=["wake", "sleep"]))
        report_lines.append("\nConfusion matrix rows=true cols=pred [wake, sleep]:\n")
        report_lines.append(str(confusion_matrix(y_test, pred, labels=[0, 1])))
    else:
        pred = model.predict(training[feature_cols])
        report_lines.append("Training-set report only; not enough tracks/classes for held-out split:\n")
        report_lines.append(classification_report(training["label_value"], pred, target_names=["wake", "sleep"]))

    metadata = {
        "feature_columns": feature_cols,
        "label_map": LABEL_TO_VALUE,
        "value_to_label": VALUE_TO_LABEL,
        "fps": float(args.fps),
        "mm_per_px": float(args.mm_per_px),
        "n_rows": int(len(training)),
        "n_sleep": int((training["label_value"] == 1).sum()),
        "n_wake": int((training["label_value"] == 0).sum()),
        "n_tracks": int(training["track_name"].nunique()),
        "training_features": str(training_path),
    }
    bundle = {"model": model, "metadata": metadata}
    joblib.dump(bundle, args.out / "sleep_random_forest.joblib")
    (args.out / "training_report.txt").write_text("\n".join(report_lines) + "\n")
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("\n".join(report_lines))
    print(f"wrote model: {args.out / 'sleep_random_forest.joblib'}")
    print(f"wrote feature importances: {args.out / 'feature_importance.csv'}")


def predict_one_track(
    *,
    model_bundle: dict,
    track_path: Path,
    speed_root: Path | None,
    out_path: Path,
    fps: float,
    mm_per_px: float,
) -> None:
    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    feature_cols = list(metadata["feature_columns"])
    features = extract_track_features(track_path, speed_root=speed_root, fps=fps, mm_per_px=mm_per_px)
    if features.empty:
        return
    missing = [col for col in feature_cols if col not in features.columns]
    if missing:
        raise ValueError(f"{track_path.name} missing trained feature columns: {missing}")
    proba = model.predict_proba(features[feature_cols])
    class_order = list(model.named_steps["forest"].classes_)
    sleep_idx = class_order.index(1)
    wake_idx = class_order.index(0)
    pred = model.predict(features[feature_cols])
    out = features[["Frame", "track_name", "track_id", "side"]].copy()
    out["sleep_probability"] = proba[:, sleep_idx]
    out["wake_probability"] = proba[:, wake_idx]
    out["predicted_label"] = [VALUE_TO_LABEL[int(value)] for value in pred]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(out):,} rows)")


def predict_classifier(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model)
    track_paths = []
    if args.track is not None:
        track_paths.append(args.track)
    if args.per_track_dir is not None:
        track_paths.extend(sorted(args.per_track_dir.glob(args.track_glob)))
    if not track_paths:
        raise ValueError("Provide --track or --per_track_dir")
    args.out.mkdir(parents=True, exist_ok=True)
    for track_path in track_paths:
        out_path = args.out / f"{track_path.stem}_sleep_predictions.parquet"
        predict_one_track(
            model_bundle=bundle,
            track_path=track_path,
            speed_root=args.speed_root,
            out_path=out_path,
            fps=float(args.fps),
            mm_per_px=float(args.mm_per_px),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train random-forest sleep/wake classifier.")
    train.add_argument("--labels", type=Path, nargs="+", required=True, help="Sleep label parquet/csv files.")
    train.add_argument("--per_track_dir", type=Path, default=None, help="Folder with TrackID_*.parquet files.")
    train.add_argument("--speed_root", type=Path, default=None, help="speed_vectors output folder.")
    train.add_argument("--out", type=Path, required=True, help="Output model/report folder.")
    train.add_argument("--fps", type=float, default=24.0)
    train.add_argument("--mm_per_px", type=float, default=0.016)
    train.add_argument("--context_seconds", type=float, default=30.0)
    train.add_argument("--max_rows_per_class", type=int, default=250000)
    train.add_argument("--test_size", type=float, default=0.25)
    train.add_argument("--n_estimators", type=int, default=200)
    train.add_argument("--max_depth", type=int, default=10)
    train.add_argument("--min_samples_leaf", type=int, default=50)
    train.add_argument("--n_jobs", type=int, default=-1)
    train.add_argument("--random_state", type=int, default=0)
    train.set_defaults(func=train_classifier)

    predict = sub.add_parser("predict", help="Apply trained classifier to one track or a folder.")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--track", type=Path, default=None)
    predict.add_argument("--per_track_dir", type=Path, default=None)
    predict.add_argument("--track_glob", default="TrackID_*.parquet")
    predict.add_argument("--speed_root", type=Path, default=None)
    predict.add_argument("--out", type=Path, required=True)
    predict.add_argument("--fps", type=float, default=24.0)
    predict.add_argument("--mm_per_px", type=float, default=0.016)
    predict.set_defaults(func=predict_classifier)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
