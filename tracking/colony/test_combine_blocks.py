"""Cross-block time, identity, cache, and publication invariants."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.sleep_motion_utils import compute_sleep_motion_cache, load_sleep_motion_cache
from tracking.stitch_tracks import write_parquet_with_num_frames, concatenate_shifted_parquets
from tracking.colony.combine_blocks import build_plan, worker, finalize, publish
from tracking.colony.block_caches import CACHE_METADATA


def add_track(block, stamp, side="left", track_id=1, frames=(0, 1, 3), speeds=(1., np.nan, 2., 3.)):
    name = f"TrackID_{track_id:04d}_all_{stamp[-6:]}_{side}.parquet"
    path = block / "stitched/per_track" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(dict(Frame=frames, TrackID=[track_id] * len(frames), Bodypoint=[0] * len(frames),
                             X=np.arange(len(frames), dtype=float), Y=np.zeros(len(frames)),
                             TrackX=np.arange(len(frames), dtype=float), TrackY=np.zeros(len(frames)),
                             source_file=[f"{stamp}_chunk000_{side}.parquet"] * len(frames)))
    write_parquet_with_num_frames(rows, path, num_frames=4, engine="pyarrow", compression="zstd")
    chunks = block / "tracks"
    chunks.mkdir(exist_ok=True)
    (chunks / f"{stamp}_chunk000_{side}.parquet").touch()
    cache = block / "stitched/speed_vectors/per_track" / path.stem
    cache.mkdir(parents=True)
    np.save(cache / "speed_mm_s.npy", np.asarray(speeds, dtype=np.float32))
    (cache / "speed_metadata.json").write_text(json.dumps(dict(track_name=name, track_id=track_id,
        fps=4., mm_per_px=.016, frame_min=0, frame_max=3, n_frames=4,
        n_observed_frames=len(frames), n_valid_speed_frames=3,
        bodypoint_filter=0, max_interp_gap_frames=5, smooth_sigma_frames=2., max_speed_mm_s=5.,
        x_col="TrackX", y_col="TrackY")))
    compute_sleep_motion_cache(path, block / "stitched/sleep_motion/per_track" / path.stem, fps=4)
    return path


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "20260515"
    # Deliberately different from the parent date, and crossing midnight.
    add_track(root / "block02", "20260518_235959")
    add_track(root / "block03", "20260519_000001", speeds=(4., np.nan, 5., 6.))
    return root


def prepare_run(root, tmp_path):
    plan = build_plan(root)
    run = tmp_path / "run"
    run.mkdir()
    destination = root / "continous_stitched"
    lock = root / ".continous_stitched.lock"
    lock.mkdir()
    (lock / "run").write_text(str(run))
    plan.update(run=str(run), stage=str(run / "result"), output=str(destination), overwrite=False, lock=str(lock))
    manifest = run / "plan.json"
    manifest.write_text(json.dumps(plan))
    return plan, manifest


def test_elapsed_midnight_and_same_tag_opposite_sides(dataset):
    add_track(dataset / "block02", "20260518_235959", side="right")
    plan = build_plan(dataset)
    group = plan["groups"][0]
    assert group["start_datetime"] == "2026-05-18T23:59:59"
    assert group["num_frames"] == 12
    assert group["blocks"][1]["frame_offset"] == 8
    assert group["blocks"][1]["gap_before_frames"] == 4
    assert {(a["side"], a["track_id"]) for a in group["ants"]} == {("left", 1), ("right", 1)}


def test_missing_block_splits_runs_and_singletons_are_reported(dataset):
    add_track(dataset / "block05", "20260520_000000")
    add_track(dataset / "block06", "20260520_000002")
    add_track(dataset / "block08", "20260521_000000")
    (dataset / "block04").mkdir()
    plan = build_plan(dataset)
    assert [g["relative_output"] for g in plan["groups"]] == ["block02_block03", "block05_block06"]
    assert {s["block"] for s in plan["skipped_blocks"]} == {"block04", "block08"}


def test_overlapping_recordings_rejected(dataset):
    path = dataset / "block03/block_combination_timing.json"
    path.write_text(json.dumps(dict(start_datetime="2026-05-18T23:59:59.5", fps=4)))
    with pytest.raises(ValueError, match="Overlapping"):
        build_plan(dataset)


def test_incompatible_cache_parameters_rejected(dataset):
    path = next((dataset / "block03/stitched/speed_vectors").rglob("speed_metadata.json"))
    meta = json.loads(path.read_text())
    meta["smooth_sigma_frames"] = 7
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="smooth_sigma_frames"):
        build_plan(dataset)


def test_no_frame_rate_guess(dataset):
    for path in dataset.rglob("*metadata.json"):
        meta = json.loads(path.read_text())
        meta.pop("fps", None)
        path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="frame rates"):
        build_plan(dataset)
    assert build_plan(dataset, fps=4)["groups"][0]["fps"] == 4


def test_tracking_only_blocks_read_fps_from_camera_diagnostics(dataset):
    import shutil
    for block in [dataset / "block02", dataset / "block03"]:
        shutil.rmtree(block / "stitched/speed_vectors")
        shutil.rmtree(block / "stitched/sleep_motion")
        (block / "cam01.avi.diag.json").write_text(json.dumps({"context": {"fps": 4}}))
    plan = build_plan(dataset)
    assert plan["groups"][0]["fps"] == 4
    assert all(len(b["fps_sources"]) == 1 for b in plan["groups"][0]["blocks"])


def test_duplicate_ant_identity_rejected(dataset):
    source = next((dataset / "block02/stitched/per_track").glob("*.parquet"))
    target = source.with_name(source.name.replace("0001", "00001"))
    target.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="duplicate ant"):
        build_plan(dataset)


def test_old_timestamp_track_is_reported_and_not_double_counted(dataset):
    source = next((dataset / "block02/stitched/per_track").glob("*.parquet"))
    target = source.with_name(source.name.replace("235959", "235958"))
    target.write_bytes(source.read_bytes())
    plan = build_plan(dataset)
    assert len(plan["groups"][0]["ants"]) == 1
    assert plan["groups"][0]["blocks"][0]["ignored_tracks"][0]["path"] == str(target)


def test_streamed_parquet_promotes_schema_and_preserves_columns(tmp_path):
    paths = []
    for i, dtype in enumerate([pa.int64(), pa.float64()]):
        path = tmp_path / f"{i}.parquet"
        table = pa.table({"Frame": pa.array([0, 1]), "TrackID": pa.array([4, 4]),
                          "SleapCam": pa.array([1, 2], type=dtype), "extra": pa.array(["a", "b"])})
        pq.write_table(table, path)
        paths.append((path, i * 4, f"block0{i}"))
    out = tmp_path / "out.parquet"
    report = concatenate_shifted_parquets(paths, out, expected_track_id=4, keep_block_frame=True, batch_size=1)
    table = pd.read_parquet(out)
    assert report["rows"] == 4
    assert table.Frame.tolist() == [0, 1, 4, 5]
    assert table.block_frame.tolist() == [0, 1, 0, 1]
    assert table.source_block.tolist() == ["block00", "block00", "block01", "block01"]
    assert table.extra.tolist() == ["a", "b", "a", "b"]
    with pytest.raises(ValueError, match="TrackID"):
        concatenate_shifted_parquets(paths, tmp_path / "bad.parquet", expected_track_id=9)
    assert not (tmp_path / "bad.parquet").exists()


def test_worker_cache_alignment_missing_ant_and_publication(dataset, tmp_path):
    add_track(dataset / "block03", "20260519_000001", side="right", track_id=7)
    plan, manifest = prepare_run(dataset, tmp_path)
    group = plan["groups"][0]
    run = Path(plan["run"])
    with pytest.raises(ValueError, match="Final validation"):
        publish(manifest, wait=False)
    for ant in group["ants"]:
        out = run / group["name"] / "tasks/per_track" / Path(ant["name"]).stem
        worker(manifest, group["name"], Path(ant["name"]), out)
        (out / "_SUCCESS").touch()
    finalize(manifest)
    publish(manifest)
    output = Path(plan["output"])
    left = group["ants"][0]
    rows = pd.read_parquet(output / "per_track" / left["name"])
    assert rows.Frame.tolist() == [0, 1, 3, 8, 9, 11]
    speed = np.load(output / "speed_vectors/per_track" / Path(left["name"]).stem / "speed_mm_s.npy")
    np.testing.assert_equal(speed, [1, np.nan, 2, 3, np.nan, np.nan, np.nan, np.nan, 4, np.nan, 5, 6])
    motion = load_sleep_motion_cache(output / "sleep_motion/per_track" / Path(left["name"]).stem)
    assert motion.frames.tolist() == [0, 1, 3, 8, 9, 11]
    assert motion.metadata["n_frames"] == 12
    right = group["ants"][1]
    right_speed = np.load(output / "speed_vectors/per_track" / Path(right["name"]).stem / "speed_mm_s.npy")
    assert np.isnan(right_speed[:8]).all()
    assert (output / "source_blocks/block02").resolve() == dataset / "block02"
    assert not Path(plan["lock"]).exists()
    assert (run / "PUBLISHED.json").is_file()


def test_changed_inputs_stop_worker(dataset, tmp_path):
    plan, manifest = prepare_run(dataset, tmp_path)
    ant = plan["groups"][0]["ants"][0]
    path = Path(ant["sources"][0]["caches"]["speed_vectors"]["metadata"]["path"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Input changed"):
        worker(manifest, plan["groups"][0]["name"], Path(ant["name"]), tmp_path / "task")


def test_sleep_tables_shift_without_merging_bouts(dataset, tmp_path):
    for block in [dataset / "block02", dataset / "block03"]:
        track = next((block / "stitched/per_track").glob("*.parquet"))
        out = block / "stitched/sleep_motion_labels/per_track" / track.stem
        out.mkdir(parents=True)
        for name, values in [("sleep_state_i1.npy", np.array([-1, 0, 1, 1], np.int8)),
                             ("quiet_fraction_f2.npy", np.array([np.nan, 0, 1, 1], np.float16)),
                             ("valid_fraction_f2.npy", np.array([0, 1, 1, 1], np.float16))]:
            np.save(out / name, values)
        pd.DataFrame(dict(frame_start=[2], frame_end=[3], n_frames=[2], duration_seconds=[.5])).to_parquet(out / "sleep_bouts.parquet")
        (out / CACHE_METADATA["sleep_motion_labels"]).write_text(json.dumps(dict(
            fps=4, frame_min=0, frame_max=3, classifier_parameters={}, classifier_type="test",
            summary=dict(track_name=track.name, n_cached_frames=3))))
    plan, manifest = prepare_run(dataset, tmp_path)
    ant = plan["groups"][0]["ants"][0]
    worker(manifest, plan["groups"][0]["name"], Path(ant["name"]), tmp_path / "task")
    out = Path(plan["stage"]) / "sleep_motion_labels/per_track" / Path(ant["name"]).stem
    bouts = pd.read_parquet(out / "sleep_bouts.parquet")
    assert bouts.frame_start.tolist() == [2, 10]
    assert bouts.frame_end.tolist() == [3, 11]
    state = np.load(out / "sleep_state_i1.npy")
    assert state.tolist() == [-1, 0, 1, 1, -1, -1, -1, -1, -1, 0, 1, 1]
    meta = json.loads((out / CACHE_METADATA["sleep_motion_labels"]).read_text())
    assert meta["summary"]["n_sleep_frames"] == 4
    assert meta["summary"]["n_unknown_frames"] == 6


def test_rf_vectors_with_nonzero_origin_and_empty_segment(dataset, tmp_path):
    for i, block in enumerate([dataset / "block02", dataset / "block03"]):
        track = next((block / "stitched/per_track").glob("*.parquet"))
        out = block / "stitched/sleep_predictions/per_track" / track.stem
        out.mkdir(parents=True)
        for name, values in [("predicted_sleep_i1.npy", np.array([0, -1, 1] if i == 0 else [], np.int8)),
                             ("sleep_probability_f4.npy", np.array([.1, np.nan, .9] if i == 0 else [], np.float32)),
                             ("wake_probability_f4.npy", np.array([.9, np.nan, .1] if i == 0 else [], np.float32))]:
            np.save(out / name, values)
        pd.DataFrame(dict(Frame=[1, 3] if i == 0 else [], track_name=[track.name] * (2 if i == 0 else 0))).to_parquet(out / "sleep_predictions.parquet")
        pd.DataFrame(dict(frame_start=[3] if i == 0 else [], frame_end=[3] if i == 0 else [])).to_parquet(out / "sleep_bouts.parquet")
        (out / CACHE_METADATA["sleep_predictions"]).write_text(json.dumps(dict(
            track_name=track.name, fps=4, frame_min=1 if i == 0 else None, frame_max=3 if i == 0 else None,
            model_path="test_model", feature_mode="speed_only")))
    plan, manifest = prepare_run(dataset, tmp_path)
    ant = plan["groups"][0]["ants"][0]
    worker(manifest, plan["groups"][0]["name"], Path(ant["name"]), tmp_path / "task")
    out = Path(plan["stage"]) / "sleep_predictions/per_track" / Path(ant["name"]).stem
    assert np.load(out / "predicted_sleep_i1.npy").tolist() == [-1, 0, -1, 1] + [-1] * 8
    table = pd.read_parquet(out / "sleep_predictions.parquet")
    assert table.Frame.tolist() == [1, 3]
    assert table.track_name.tolist() == [ant["name"], ant["name"]]


def test_geometry_caches_rebuilt_from_shared_annotations(dataset, tmp_path):
    import csv
    for block in [dataset / "block02", dataset / "block03"]:
        rows = []
        for name, label, xmin, xmax in [("left arena", "arena", -10, 5), ("right arena", "arena", 6, 20),
                                        ("left colony", "colony_left", -2, 3), ("right colony", "colony_right", 8, 12)]:
            rows.append(dict(name=name, semantic_label=label, shape="rectangle", tracking_x_min_px=xmin,
                             tracking_x_max_px=xmax, tracking_y_min_px=-10, tracking_y_max_px=10, mm_per_pixel=.016))
        with (block / "panorama_regions.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plan, manifest = prepare_run(dataset, tmp_path)
    group = plan["groups"][0]
    root = Path(plan["stage"])
    root.mkdir()
    (root / "grid_bounds_from_panorama_regions.json").write_text(json.dumps(group["geometry"]["bounds"]))
    ant = group["ants"][0]
    worker(manifest, group["name"], Path(ant["name"]), tmp_path / "task")
    out = root / "colony_presence_vectors/per_track" / Path(ant["name"]).stem
    assert np.load(out / "colony_presence_i1.npy").tolist() == [1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1]
    grid = root / "grid_occupancy_histograms/per_track" / Path(ant["name"]).stem
    meta = json.loads((grid / "grid_occupancy_metadata.json").read_text())
    assert meta["x_split_px"] == 5.5
    assert meta["n_detected_frames"] == 6
    assert meta["combination_method"] == "recomputed_from_combined_track_and_panorama_regions"


def test_retry_refuses_active_workers(dataset, tmp_path, monkeypatch):
    from tracking.colony import combine_blocks as cb
    plan, manifest = prepare_run(dataset, tmp_path)
    run = Path(plan["run"])
    jobs = run / plan["groups"][0]["name"] / "tasks/jobs"
    jobs.mkdir(parents=True)
    (jobs / "block_combine_job_ids.tsv").write_text("0\t123\tinput\toutput\n")
    (run / "submission.json").write_text(json.dumps(dict(groups=[dict(completion_job="124")], finalizer="125")))
    monkeypatch.setattr(cb.subprocess, "check_output", lambda *args, **kwargs: "123|RUNNING\n124|PENDING\n125|PENDING\n")
    with pytest.raises(RuntimeError, match="still active"):
        cb.retry(manifest)


def test_retry_submits_only_unfinished_ants(dataset, tmp_path, monkeypatch):
    from tracking.colony import combine_blocks as cb
    add_track(dataset / "block03", "20260519_000001", side="right", track_id=7)
    plan, manifest = prepare_run(dataset, tmp_path)
    run = Path(plan["run"])
    jobs = run / plan["groups"][0]["name"] / "tasks/jobs"
    jobs.mkdir(parents=True)
    rows = []
    for i, ant in enumerate(plan["groups"][0]["ants"]):
        task = jobs.parent / "per_track" / Path(ant["name"]).stem
        task.mkdir(parents=True)
        if i == 0:
            (task / "_SUCCESS").touch()
        rows.append(f"{i}\tinput\t{ant['name']}\t{Path(ant['name']).stem}\t{ant['track_id']}\t{task}\n")
    (jobs / "block_combine_worklist.tsv").write_text("".join(rows))
    (run / "submission.json").write_text(json.dumps(dict(groups=[dict(completion_job="124")], finalizer="125")))
    commands = []
    def command(args, **kwargs):
        commands.append(args)
        return "" if args[0] == "squeue" else "999\n"
    monkeypatch.setattr(cb.subprocess, "check_output", command)
    monkeypatch.setattr(cb, "start_publisher", lambda *args: None)
    cb.retry(manifest)
    assert len(commands) == 3
    assert commands[1][-1].endswith("block_combine_task1.sbatch")
    assert commands[2][-2] == "--dependency=afterok:999"
