import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from scripts import publish_interaction_run as publisher


@pytest.fixture(params=["n_directed_detections", "n_pair_detections"])
def run(tmp_path, request):
    source = tmp_path / "flash"
    source.mkdir()
    path = source / "chunk000_right.parquet"
    pd.DataFrame({"Frame": [0]}).to_parquet(path)
    parameters = {"micro_interaction_distance_mm": 0.1}
    path.with_suffix(".metadata.json").write_text(json.dumps(dict(
        parameters=parameters, output_size=path.stat().st_size, **{request.param: 1},
    )))
    manifest = dict(run_id="test", output_dir=str(source), bucket_output_dir=str(tmp_path / "bucket"),
                    expected_files=[path.name], parameters=parameters)
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    (source / "interactions_complete.ok").touch()
    return tmp_path, manifest


def test_publishes_only_after_transfer(run, monkeypatch):
    root, manifest = run
    destination = Path(manifest["bucket_output_dir"])

    def transfer(command, *, check):
        assert command[0] == "rsync" and check
        assert not (destination / "transfer_complete.ok").exists()
        shutil.copytree(command[-2], command[-1], dirs_exist_ok=True)

    monkeypatch.setattr(publisher.subprocess, "run", transfer)
    publisher.publish(root)
    assert json.loads((destination / "transfer_complete.ok").read_text())["n_chunks"] == 1
    assert pd.read_parquet(destination / manifest["expected_files"][0]).Frame.tolist() == [0]


def test_refuses_wrong_distance(run):
    root, manifest = run
    path = Path(manifest["output_dir"]) / "chunk000_right.metadata.json"
    metadata = json.loads(path.read_text())
    metadata["parameters"]["micro_interaction_distance_mm"] = 1.0
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Parameter mismatch"):
        publisher.publish(root)
    assert not Path(manifest["bucket_output_dir"]).exists()


def test_failed_transfer_never_marks_ready(run, monkeypatch):
    root, manifest = run

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(23, command)

    monkeypatch.setattr(publisher.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        publisher.publish(root)
    assert not (Path(manifest["bucket_output_dir"]) / "transfer_complete.ok").exists()


def test_unfinished_workers_never_publish(run):
    root, manifest = run
    (Path(manifest["output_dir"]) / "interactions_complete.ok").unlink()
    with pytest.raises(TimeoutError):
        publisher.publish(root, timeout_hours=0)
    assert not Path(manifest["bucket_output_dir"]).exists()
