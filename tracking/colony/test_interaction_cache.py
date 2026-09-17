import json
import pandas as pd

from tracking.colony import interaction_one_chunk as worker
from tracking.colony import interaction_batch as batch


def test_distance_and_parameter_aware_rebuild(tmp_path):
    source = tmp_path / "test_chunk000_right.parquet"
    rows = []
    for frame, distance in enumerate([0.05, 0.2]):
        for ant, node, x in [(1, 4, 0), (2, 0, distance)]:
            rows.append(dict(Frame=frame, TrackID=ant, Bodypoint=node, X=x, Y=0, TrackX=x, TrackY=0))
    pd.DataFrame(rows).to_parquet(source)
    output = tmp_path / "outputs" / source.name
    kwargs = dict(chunk_file=source, output_path=output, mm_per_px=1, interaction_radius_mm=8,
                  micro_interaction_distance_mm=1, antenna_bodypoints=(4, 5, 6, 7, 8, 9),
                  frame_start=0, max_frames=None, frame_step=1, frame_batch_size=1,
                  progress_every_frames=0, skip_existing=True)
    worker.process_chunk(**kwargs)
    assert pd.read_parquet(output).Frame.tolist() == [0, 1]
    kwargs["micro_interaction_distance_mm"] = 0.1
    worker.process_chunk(**kwargs)
    assert pd.read_parquet(output).Frame.tolist() == [0]
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert metadata["parameters"]["micro_interaction_distance_mm"] == 0.1
    assert metadata["parameters"]["geometry"] == "all_skeleton_segments_and_nodes"
    assert metadata["parameters"]["directed"] is False
    assert pd.read_parquet(output).columns.tolist() == ["Frame", "ant_a", "ant_b", "distance_mm"]
    assert metadata["n_pair_detections"] == 1
    before = output.stat().st_mtime_ns
    worker.process_chunk(**kwargs)
    assert output.stat().st_mtime_ns == before
    matching = metadata["parameters"]
    assert not batch.discover_jobs(tmp_path, output.parent, sides=["right"], skip_existing=True,
                                   chunks=None, parameters=matching)
    stale = {**matching, "micro_interaction_distance_mm": 1.0}
    assert len(batch.discover_jobs(tmp_path, output.parent, sides=["right"], skip_existing=True,
                                   chunks=None, parameters=stale)) == 1
    assert not list(output.parent.glob("*.partial.parquet"))


def test_default_is_point_one():
    assert worker.DEFAULT_MICRO_DISTANCE_MM == 0.1
