#!/usr/bin/env python3
"""Train or apply a supervised sleep/wake classifier from labeled ant frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
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
from analysis.compute_track_sleep_predictions import (
    build_bouts,
    empty_prediction_table,
    write_dense_vectors,
)


LABEL_TO_VALUE = {"wake": 0, "sleep": 1}
VALUE_TO_LABEL = {0: "wake", 1: "sleep"}
CROP_NAME_RE = re.compile(r"_frame(?P<frame_start>\d+)_(?P<side>left|right)_track(?P<track_id>\d+)_", re.IGNORECASE)


def parse_crop_name(path: Path | str) -> dict[str, int | str] | None:
    match = CROP_NAME_RE.search(Path(path).stem)
    if match is None:
        return None
    return {
        "crop_frame_start": int(match.group("frame_start")),
        "side": match.group("side").lower(),
        "track_id": int(match.group("track_id")),
    }


def collapse_frame_labels_to_intervals(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.sort_values("frame_start", kind="mergesort").reset_index(drop=True)
    if labels.empty:
        return labels
    break_group = (
        (labels["label"].ne(labels["label"].shift()))
        | (labels["frame_start"].ne(labels["frame_start"].shift() + 1))
        | (labels["track_id"].ne(labels["track_id"].shift()))
        | (labels["side"].ne(labels["side"].shift()))
        | (labels["label_file"].ne(labels["label_file"].shift()))
    )
    labels["_interval_group"] = break_group.cumsum()
    grouped = (
        labels.groupby("_interval_group", sort=True, as_index=False)
        .agg(
            frame_start=("frame_start", "first"),
            frame_end=("frame_start", "last"),
            label=("label", "first"),
            label_value=("label_value", "first"),
            label_file=("label_file", "first"),
            video_path=("video_path", "first"),
            local_frame_start=("local_frame", "first"),
            local_frame_end=("local_frame", "last"),
            crop_frame_start=("crop_frame_start", "first"),
            track_id=("track_id", "first"),
            side=("side", "first"),
        )
        .drop(columns=["_interval_group"])
    )
    return grouped.reset_index(drop=True)


def read_labels(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        labels = pd.read_csv(path)
    else:
        labels = pd.read_parquet(path)
    labels = labels.copy()
    labels["label"] = labels["label"].astype(str).str.lower()
    labels = labels[labels["label"].isin(LABEL_TO_VALUE)].copy()
    if labels.empty:
        return labels

    if {"Frame", "label"}.issubset(labels.columns):
        labels["local_frame"] = pd.to_numeric(labels["Frame"], errors="coerce")
        labels = labels.dropna(subset=["local_frame"]).copy()
        labels["local_frame"] = labels["local_frame"].round().astype(np.int64)
        video_path = None
        if "video_path" in labels.columns and labels["video_path"].notna().any():
            video_path = Path(str(labels["video_path"].dropna().iloc[0]))
        metadata_path = path.with_name(path.name.replace("_labels.parquet", "_metadata.json"))
        if video_path is None and metadata_path.exists():
            try:
                video_path = Path(str(json.loads(metadata_path.read_text()).get("video_path", "")))
            except Exception:
                video_path = None
        crop_info = parse_crop_name(video_path or path)
        if crop_info is None:
            raise ValueError(
                f"{path} looks like a frame-level GUI label table, but track/frame info "
                "could not be parsed from its video filename."
            )
        labels["frame_start"] = labels["local_frame"] + int(crop_info["crop_frame_start"])
        labels["label_value"] = labels["label"].map(LABEL_TO_VALUE).astype(np.int8)
        labels["label_file"] = str(path)
        labels["video_path"] = str(video_path) if video_path is not None else ""
        labels["crop_frame_start"] = int(crop_info["crop_frame_start"])
        labels["track_id"] = int(crop_info["track_id"])
        labels["side"] = str(crop_info["side"])
        return collapse_frame_labels_to_intervals(labels)

    required = {"frame_start", "frame_end", "label"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"{path} missing label columns: {sorted(missing)}")
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


def expand_label_paths(paths: list[Path] | None, labels_dir: Path | None, label_glob: str) -> list[Path]:
    expanded: list[Path] = []
    if labels_dir is not None:
        expanded.extend(sorted(Path(labels_dir).glob(label_glob)))
    for path in paths or []:
        path = Path(path)
        if path.is_dir():
            expanded.extend(sorted(path.glob(label_glob)))
        else:
            expanded.append(path)
    deduped = []
    seen = set()
    for path in expanded:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def load_label_files(paths: list[Path] | None, *, labels_dir: Path | None = None, label_glob: str = "*_labels.parquet") -> pd.DataFrame:
    expanded = expand_label_paths(paths, labels_dir, label_glob)
    tables = [read_labels(path) for path in expanded]
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
    feature_mode: str,
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
        feature_mode=feature_mode,
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
        for col in ["label_file", "video_path", "crop_frame_start", "track_id", "side"]:
            if col in label_row and pd.notna(label_row[col]):
                subset[col] = label_row[col]
        if "crop_frame_start" in subset.columns:
            subset["local_frame"] = subset["Frame"].astype(np.int64) - int(label_row["crop_frame_start"])
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
    feature_mode: str,
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
            feature_mode=feature_mode,
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
        best_split = None
        for offset in range(100):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=float(test_size),
                random_state=int(random_state) + offset,
            )
            train_idx, test_idx = next(splitter.split(training, y, groups))
            best_split = (train_idx, test_idx)
            if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2:
                break
        else:
            train_idx, test_idx = best_split
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


def add_training_predictions(
    training: pd.DataFrame,
    model,
    feature_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    out = training.copy()
    proba = model.predict_proba(out[feature_cols])
    class_order = list(model.named_steps["forest"].classes_)
    sleep_idx = class_order.index(1)
    wake_idx = class_order.index(0)
    pred = model.predict(out[feature_cols])
    out["wake_probability"] = proba[:, wake_idx]
    out["sleep_probability"] = proba[:, sleep_idx]
    out["predicted_label_value"] = pred.astype(np.int8)
    out["predicted_label"] = [VALUE_TO_LABEL[int(value)] for value in pred]
    out["split"] = "train"
    if len(test_idx) > 0:
        out.loc[out.index[np.asarray(test_idx, dtype=np.int64)], "split"] = "test"
    if "local_frame" not in out.columns and "crop_frame_start" in out.columns:
        out["local_frame"] = out["Frame"].astype(np.int64) - pd.to_numeric(out["crop_frame_start"], errors="coerce")
    return out


def _plot_confusion(predictions: pd.DataFrame, out_path: Path, *, split: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    data = predictions[predictions["split"] == split].copy()
    if data.empty:
        data = predictions.copy()
        split = "all"
    cm = confusion_matrix(data["label_value"], data["predicted_label_value"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["wake", "sleep"])
    ax.set_yticks([0, 1], labels=["wake", "sleep"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Sleep classifier confusion matrix ({split})")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{cm[row, col]:,}", ha="center", va="center", color="black")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_probability_curves(predictions: pd.DataFrame, out_path: Path, *, split: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    data = predictions[predictions["split"] == split].copy()
    if data.empty or data["label_value"].nunique() < 2:
        data = predictions.copy()
        split = "all"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    if data["label_value"].nunique() >= 2:
        y = data["label_value"].to_numpy(np.int8)
        score = data["sleep_probability"].to_numpy(np.float64)
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        roc_auc = roc_auc_score(y, score)
        ap = average_precision_score(y, score)
        axes[0].plot(fpr, tpr, color="tab:blue", lw=2, label=f"AUC {roc_auc:.3f}")
        axes[0].plot([0, 1], [0, 1], color="0.7", lw=1, linestyle="--")
        axes[1].plot(recall, precision, color="tab:orange", lw=2, label=f"AP {ap:.3f}")
        for ax in axes:
            ax.legend(loc="lower right")
    axes[0].set_title(f"ROC ({split})")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].set_title(f"Precision-recall ({split})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_probability_histogram(predictions: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    bins = np.linspace(0, 1, 41)
    for label, color in [("wake", "tab:blue"), ("sleep", "tab:orange")]:
        values = predictions.loc[predictions["label"] == label, "sleep_probability"].dropna().to_numpy(np.float64)
        if len(values):
            ax.hist(values, bins=bins, alpha=0.55, density=True, label=f"{label} n={len(values):,}", color=color)
    ax.set_xlabel("Predicted P(sleep)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted sleep probability by ground truth label")
    ax.legend()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_feature_importance(importances: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importances.head(30).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.27 * len(top))), constrained_layout=True)
    ax.barh(top["feature"], top["importance"], color="0.25")
    ax.set_xlabel("Random forest importance")
    ax.set_title("Top classifier features")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_motion_feature_distributions(predictions: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidate_cols = [
        "posture_bp_speed_median_mean_5s",
        "posture_bp_speed_q90_mean_5s",
        "posture_bp_speed_mean_mean_5s",
        "posture_bp_speed_max_mean_5s",
        "speed_mean_5s",
        "posture_bp_speed_median_mm_s",
    ]
    cols = [col for col in candidate_cols if col in predictions.columns]
    if not cols:
        return
    fig, axes = plt.subplots(len(cols), 1, figsize=(8.5, max(3.0, 2.2 * len(cols))), constrained_layout=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        groups = [
            predictions.loc[predictions["label"] == "wake", col].dropna().to_numpy(np.float64),
            predictions.loc[predictions["label"] == "sleep", col].dropna().to_numpy(np.float64),
        ]
        groups = [np.log1p(np.clip(values, 0, None)) for values in groups]
        ax.boxplot(groups, tick_labels=["wake", "sleep"], showfliers=False)
        ax.set_ylabel("log1p mm/s")
        ax.set_title(col)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_timeline_recapitulation(predictions: pd.DataFrame, out_path: Path, *, max_panels: int = 8) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group_col = "label_file" if "label_file" in predictions.columns else "track_name"
    groups = list(predictions.groupby(group_col, sort=True))
    groups = groups[: int(max_panels)]
    if not groups:
        return
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, max(3.0, 2.0 * len(groups))), sharex=False, constrained_layout=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, (name, group) in zip(axes, groups):
        group = group.sort_values("Frame", kind="mergesort")
        x_col = "local_frame" if "local_frame" in group.columns else "Frame"
        x = pd.to_numeric(group[x_col], errors="coerce").to_numpy(np.float64)
        if np.nanmax(x) > 1000:
            x = x / 24.0
            xlabel = "crop time (s)" if x_col == "local_frame" else "recording time (s)"
        else:
            xlabel = x_col
        y_true = group["label_value"].to_numpy(np.float64)
        prob = group["sleep_probability"].to_numpy(np.float64)
        step = max(1, len(group) // 2500)
        ax.plot(x[::step], prob[::step], color="black", lw=1.1, label="P(sleep)")
        ax.fill_between(
            x[::step],
            0,
            1,
            where=y_true[::step] > 0.5,
            color="tab:orange",
            alpha=0.22,
            step="mid",
            label="true sleep",
        )
        ax.fill_between(
            x[::step],
            0,
            1,
            where=y_true[::step] < 0.5,
            color="tab:blue",
            alpha=0.12,
            step="mid",
            label="true wake",
        )
        title = Path(str(name)).name.replace("_labels.parquet", "")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("P(sleep)")
        ax.set_xlabel(xlabel)
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary_plots(
    predictions: pd.DataFrame,
    importances: pd.DataFrame,
    out_dir: Path,
    *,
    eval_split: str,
) -> None:
    out_dir = Path(out_dir)
    _plot_confusion(predictions, out_dir / "summary_confusion_matrix.png", split=eval_split)
    _plot_probability_curves(predictions, out_dir / "summary_roc_pr.png", split=eval_split)
    _plot_probability_histogram(predictions, out_dir / "summary_sleep_probability_histogram.png")
    _plot_feature_importance(importances, out_dir / "summary_feature_importance_top30.png")
    _plot_motion_feature_distributions(predictions, out_dir / "summary_motion_features_by_label.png")
    _plot_timeline_recapitulation(predictions, out_dir / "summary_label_recapitulation_timeline.png")


def select_feature_columns(training: pd.DataFrame, feature_set: str) -> list[str]:
    feature_cols = default_feature_columns(training)
    if feature_set == "posture_motion":
        feature_cols = [
            col
            for col in feature_cols
            if col.startswith("posture_bp_speed_") or (col.startswith("bp") and col.endswith("_speed_mm_s"))
        ]
    elif feature_set == "speed_only":
        feature_cols = [col for col in feature_cols if col.startswith("speed_")]
    return feature_cols


def feature_mode_for_feature_set(feature_set: str) -> str:
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


def train_classifier(args: argparse.Namespace) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline

    labels = load_label_files(args.labels, labels_dir=args.labels_dir, label_glob=args.label_glob)
    training = build_training_table(
        labels,
        per_track_dir=args.per_track_dir,
        speed_root=args.speed_root,
        fps=float(args.fps),
        mm_per_px=float(args.mm_per_px),
        context_seconds=float(args.context_seconds),
        max_rows_per_class=args.max_rows_per_class,
        random_state=int(args.random_state),
        feature_mode=feature_mode_for_feature_set(args.feature_set),
    )
    feature_cols = select_feature_columns(training, args.feature_set)
    if not feature_cols:
        raise ValueError(f"No feature columns available for feature_set={args.feature_set!r}")

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
        report_lines.append(
            classification_report(
                y_test,
                pred,
                labels=[0, 1],
                target_names=["wake", "sleep"],
                zero_division=0,
            )
        )
        report_lines.append("\nConfusion matrix rows=true cols=pred [wake, sleep]:\n")
        report_lines.append(str(confusion_matrix(y_test, pred, labels=[0, 1])))
    else:
        pred = model.predict(training[feature_cols])
        report_lines.append("Training-set report only; not enough tracks/classes for held-out split:\n")
        report_lines.append(
            classification_report(
                training["label_value"],
                pred,
                labels=[0, 1],
                target_names=["wake", "sleep"],
                zero_division=0,
            )
        )

    predictions = add_training_predictions(training, model, feature_cols, train_idx, test_idx)
    predictions_path = args.out / "training_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    write_summary_plots(
        predictions,
        importances,
        args.out,
        eval_split="test" if len(test_idx) > 0 else "train",
    )

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
        "training_predictions": str(predictions_path),
        "feature_set": str(args.feature_set),
    }
    bundle = {"model": model, "metadata": metadata}
    joblib.dump(bundle, args.out / "sleep_random_forest.joblib")
    (args.out / "training_report.txt").write_text("\n".join(report_lines) + "\n")
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("\n".join(report_lines))
    print(f"wrote model: {args.out / 'sleep_random_forest.joblib'}")
    print(f"wrote feature importances: {args.out / 'feature_importance.csv'}")
    print(f"wrote summary plots: {args.out}")


def _classifier_classes(model) -> list[int]:
    estimator = model.named_steps["forest"] if hasattr(model, "named_steps") and "forest" in model.named_steps else model
    return [int(value) for value in estimator.classes_]


def predict_track_table(
    *,
    model_bundle: dict,
    track_path: Path,
    speed_root: Path | None,
    fps: float,
    mm_per_px: float,
) -> pd.DataFrame:
    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    feature_cols = list(metadata["feature_columns"])
    feature_mode = feature_mode_for_feature_set(str(metadata.get("feature_set", "all")))
    features = extract_track_features(
        track_path,
        speed_root=speed_root,
        fps=fps,
        mm_per_px=mm_per_px,
        feature_mode=feature_mode,
    )
    if features.empty:
        return empty_prediction_table(track_path)
    features = features.loc[prediction_valid_mask(features)].reset_index(drop=True)
    if features.empty:
        return empty_prediction_table(track_path)
    missing = [col for col in feature_cols if col not in features.columns]
    if missing:
        raise ValueError(f"{track_path.name} missing trained feature columns: {missing}")
    proba = model.predict_proba(features[feature_cols])
    class_order = _classifier_classes(model)
    sleep_idx = class_order.index(1)
    wake_idx = class_order.index(0)
    pred = model.predict(features[feature_cols])
    out = features[["Frame", "track_name", "track_id", "side"]].copy()
    out["sleep_probability"] = proba[:, sleep_idx]
    out["wake_probability"] = proba[:, wake_idx]
    out["predicted_label_value"] = pred.astype(np.int8)
    out["predicted_label"] = [VALUE_TO_LABEL[int(value)] for value in pred]
    return out.sort_values("Frame", kind="mergesort").reset_index(drop=True)


def predict_one_track(
    *,
    model_bundle: dict,
    track_path: Path,
    speed_root: Path | None,
    out_path: Path,
    fps: float,
    mm_per_px: float,
) -> None:
    out = predict_track_table(
        model_bundle=model_bundle,
        track_path=track_path,
        speed_root=speed_root,
        fps=fps,
        mm_per_px=mm_per_px,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(out):,} rows)")


def _first_prediction_value(predictions: pd.DataFrame, column: str) -> object | None:
    if column not in predictions or not predictions[column].notna().any():
        return None
    return predictions[column].dropna().iloc[0]


def _empty_dense_vectors(out_dir: Path) -> dict[str, object]:
    sleep_path = out_dir / "sleep_probability_f4.npy"
    wake_path = out_dir / "wake_probability_f4.npy"
    pred_path = out_dir / "predicted_sleep_i1.npy"
    np.save(sleep_path, np.full(0, np.nan, dtype=np.float32))
    np.save(wake_path, np.full(0, np.nan, dtype=np.float32))
    np.save(pred_path, np.full(0, -1, dtype=np.int8))
    return {
        "sleep_probability_path": str(sleep_path),
        "wake_probability_path": str(wake_path),
        "predicted_sleep_path": str(pred_path),
        "prediction_values": {"missing_frame": -1, "wake": 0, "sleep": 1},
        "frame_min": None,
        "frame_max": None,
        "n_frames": 0,
        "n_predicted_frames": 0,
        "n_sleep_frames": 0,
        "n_wake_frames": 0,
        "sleep_fraction_predicted_frames": None,
        "mean_sleep_probability": None,
    }


def write_prediction_bundle(
    *,
    predictions: pd.DataFrame,
    track_path: Path,
    model_bundle: dict,
    model_path: Path,
    speed_root: Path | None,
    out_dir: Path,
    fps: float,
    mm_per_px: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "sleep_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    bouts = build_bouts(predictions)
    bouts_path = out_dir / "sleep_bouts.parquet"
    bouts.to_parquet(bouts_path, index=False)

    dense_metadata = _empty_dense_vectors(out_dir) if predictions.empty else write_dense_vectors(predictions, out_dir)
    metadata = dict(model_bundle.get("metadata", {}))
    feature_cols = list(metadata.get("feature_columns", []))
    feature_set = str(metadata.get("feature_set", "all"))

    track_id = _first_prediction_value(predictions, "track_id")
    side = _first_prediction_value(predictions, "side")
    metadata_out = {
        "status": "ok" if not predictions.empty else "no_predictions",
        "track_path": str(track_path),
        "track_name": track_path.name,
        "track_id": int(track_id) if track_id is not None else track_id_from_name(track_path),
        "side": str(side) if side is not None else side_from_name(track_path),
        "model_path": str(model_path),
        "model_feature_set": metadata.get("feature_set"),
        "feature_mode": feature_mode_for_feature_set(feature_set),
        "n_model_features": int(len(feature_cols)),
        "prediction_table_path": str(predictions_path),
        "bouts_path": str(bouts_path),
        "features_path": None,
        "speed_root": str(speed_root) if speed_root is not None else None,
        "fps": float(fps),
        "mm_per_px": float(mm_per_px),
        "dtype_probability": "float32",
        "dtype_prediction": "int8",
        **dense_metadata,
    }
    (out_dir / "sleep_prediction_metadata.json").write_text(json.dumps(metadata_out, indent=2) + "\n")
    print(f"wrote {predictions_path} ({len(predictions):,} rows)")


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
        if args.flat_outputs:
            out_path = args.out / f"{track_path.stem}_sleep_predictions.parquet"
            predict_one_track(
                model_bundle=bundle,
                track_path=track_path,
                speed_root=args.speed_root,
                out_path=out_path,
                fps=float(args.fps),
                mm_per_px=float(args.mm_per_px),
            )
            continue

        predictions = predict_track_table(
            model_bundle=bundle,
            track_path=track_path,
            speed_root=args.speed_root,
            fps=float(args.fps),
            mm_per_px=float(args.mm_per_px),
        )
        write_prediction_bundle(
            predictions=predictions,
            track_path=track_path,
            model_bundle=bundle,
            model_path=args.model,
            speed_root=args.speed_root,
            out_dir=args.out / "per_track" / track_path.stem,
            fps=float(args.fps),
            mm_per_px=float(args.mm_per_px),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train random-forest sleep/wake classifier.")
    train.add_argument("--labels", type=Path, nargs="+", default=None, help="Sleep label parquet/csv files or directories.")
    train.add_argument("--labels_dir", type=Path, default=None, help="Directory containing GUI *_labels.parquet files.")
    train.add_argument("--label_glob", default="*_labels.parquet", help="Glob used for label directories.")
    train.add_argument("--per_track_dir", type=Path, default=None, help="Folder with TrackID_*.parquet files.")
    train.add_argument("--speed_root", type=Path, default=None, help="speed_vectors output folder.")
    train.add_argument("--out", type=Path, required=True, help="Output model/report folder.")
    train.add_argument("--fps", type=float, default=24.0)
    train.add_argument("--mm_per_px", type=float, default=0.016)
    train.add_argument("--context_seconds", type=float, default=30.0)
    train.add_argument("--max_rows_per_class", type=int, default=250000)
    train.add_argument(
        "--feature_set",
        choices=("all", "posture_motion", "speed_only"),
        default="all",
        help="Use all numeric features, posture bodypoint motion, or compact precomputed speed-vector features.",
    )
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
    predict.add_argument(
        "--flat_outputs",
        action="store_true",
        help="Write the legacy flat *_sleep_predictions.parquet files instead of pipeline-style per-track bundles.",
    )
    predict.set_defaults(func=predict_classifier)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
