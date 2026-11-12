import json
import pytest

from tracking.single_ant.manual_cup_tracking import write_json
from tracking.single_ant.run import annotation_panorama, submit, _watch


def test_tracking_batches_stay_within_array_bounds(tmp_path, monkeypatch):
    from tracking.single_ant.run import work
    manifest = tmp_path / "manifest.json"
    write_json(manifest, dict(chunks=[{}, {}], cups=[{}, {}, {}]))
    plan = dict(run=str(tmp_path), code=str(tmp_path / "code"), python="python", track_task_batch_size=4, diagnose_mem="64G",
                blocks=[dict(name="block01", manifest=str(manifest), destination="published", chunks=2, cups=3)],
                combined="combined")
    path = tmp_path / "plan.json"; write_json(path, plan)
    commands = []
    def submit_job(args, **kwargs):
        commands.append(args)
        return str(100+len(commands))
    monkeypatch.setattr("tracking.single_ant.run.subprocess.check_output", submit_job)
    submit(path)
    assert "--array=0-1" in commands[2]
    assert "_worker track" in (tmp_path / "jobs/block01_track.sbatch").read_text()
    assert "#SBATCH --mem=64G" in (tmp_path / "jobs/block01_diagnose.sbatch").read_text()
    slots = []
    monkeypatch.setattr("tracking.single_ant.run.tracking.track_task", lambda m, i: slots.append(i))
    work("track", plan, index=0)
    work("track", plan, index=1)
    assert slots == list(range(6))


def test_alternate_annotation_panorama_is_used_even_when_block_has_an_old_panorama(tmp_path):
    (tmp_path / "panorama_from_hmats.png").touch()
    chosen = tmp_path / "corrected"; chosen.mkdir()
    meta = chosen / "panorama_from_hmats_metadata.json"
    # Absolute metadata paths may have been written through another machine's mount.
    write_json(meta, dict(panorama_image="/other/mount/corrected/panorama_from_hmats.png"))
    image = chosen / "panorama_from_hmats.png"; image.touch()
    assert annotation_panorama(meta) == image


def test_resume_retries_failed_stage_and_rebuilds_its_dependencies(tmp_path, monkeypatch):
    plan = dict(run=str(tmp_path), code=str(tmp_path / "code"), python="python",
                exclude_nodes="unavailable_node", blocks=[dict(name="block01", manifest=str(tmp_path / "manifest.json"),
                             destination="published", chunks=2, cups=1)], combined="combined")
    path = tmp_path / "plan.json"
    write_json(path, plan)
    previous, states, commands, cancelled = {}, {}, [], []

    def check_output(args, **kwargs):
        if args[0] == "sacct":
            assert "--format=JobID,State" in args  # Raw IDs lose the array parent/task relationship.
            return "\n".join(f"{jid}|{state}|" for jid, state in states.items())
        assert args[0] == "sbatch"
        commands.append(args)
        return str(100 + len(commands))

    monkeypatch.setattr("tracking.single_ant.run.subprocess.check_output", check_output)
    monkeypatch.setattr("tracking.single_ant.run.subprocess.run", lambda args, **kwargs: cancelled.append(args))
    submit(path)
    assert len(commands) == 6
    assert "#SBATCH --exclude=unavailable_node" in (tmp_path / "jobs/block01_track.sbatch").read_text()
    previous = json.loads((tmp_path / "jobs.json").read_text())
    states.update({jid: "COMPLETED" for jid in previous.values()})
    states[previous["block01_track"]+"_1"] = "FAILED"
    for name in ["block01_stitch", "block01_diagnose", "block01_report"]:
        states[previous[name]] = "PENDING"
    submit(path)
    current = json.loads((tmp_path / "jobs.json").read_text())
    assert current["block01_map"] == previous["block01_map"]
    assert current["identities"] == previous["identities"]
    assert len(commands) == 10 and len(cancelled) == 3
    assert "--array=1" in commands[6]  # Only the failed slot is retried after the array finishes.
    assert not any(a.startswith("--dependency=") for a in commands[6])  # Completed jobs may be purged from the controller.
    attempts = json.loads((tmp_path / "job_attempts.json").read_text())
    assert attempts["block01_track"] == [previous["block01_track"], current["block01_track"]]
    assert "--dependency=afterok:" + current["block01_track"] in commands[7]
    states.update({jid: "COMPLETED" for jid in current.values()})
    submit(path)
    assert len(commands) == 10  # A completed run is never submitted twice.


def test_single_new_block_with_prior_blocks_waits_for_combination(tmp_path, monkeypatch):
    out = tmp_path / "block01"; out.mkdir()
    manifest = out / "manifest.json"
    write_json(manifest, dict(output=str(out)))
    write_json(out / "FULL_RUN_COMPLETE.json", {})
    write_json(tmp_path / "jobs.json", dict(report="123"))
    plan = dict(run=str(tmp_path), blocks=[dict(manifest=str(manifest), destination="published")],
                prior_manifests=["earlier_block.json"])
    published = []
    monkeypatch.setattr("tracking.single_ant.full_cup_report.publish", lambda *args: published.append(args))
    monkeypatch.setattr("tracking.single_ant.run.subprocess.check_output", lambda *args, **kwargs: "123|COMPLETED|0:0|")
    class WaitingForCombination(Exception):
        pass
    def sleep(seconds):
        raise WaitingForCombination
    monkeypatch.setattr("tracking.single_ant.run.time.sleep", sleep)
    with pytest.raises(WaitingForCombination):
        _watch(plan)
    assert len(published) == 1 and not (tmp_path / "PUBLISHED.json").exists()


def test_stitch_only_submits_no_tracking_jobs(tmp_path, monkeypatch):
    sources = []
    for i in range(2):
        p = tmp_path / f"source{i}.json"
        write_json(p, dict(cups=[{}, {}]))
        sources.append(str(p))
    plan = dict(run=str(tmp_path), code=str(tmp_path / "code"), python="python",
                blocks=[], prior_manifests=sources, combined="published")
    path = tmp_path / "plan.json"; write_json(path, plan)
    commands = []
    def submit_job(args, **kwargs):
        commands.append(args)
        return str(100+len(commands))
    monkeypatch.setattr("tracking.single_ant.run.subprocess.check_output", submit_job)
    submit(path)
    jobs = json.loads((tmp_path / "jobs.json").read_text())
    assert set(jobs) == {"combine_plan", "combine_ants", "combined_report"}
    assert "--array=0-3" in commands[1]
