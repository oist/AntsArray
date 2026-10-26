"""Prerequisite validation must neither hang on failed dependencies nor skip jobs."""
import subprocess

import pytest

from tracking.single_ant import submit


def test_failed_upstream_does_not_wait_for_queued_dependents(monkeypatch):
    monkeypatch.setattr(submit, "command", lambda *args: "10|FAILED|\n11|PENDING|")
    monkeypatch.setattr(submit, "is_live", lambda *args: True)
    monkeypatch.setattr(submit.time, "sleep", lambda *args: pytest.fail("Waited on failed dependency"))
    with pytest.raises(RuntimeError, match="10.*FAILED"):
        submit.wait_for_prerequisites("10,11", "saion")


def test_waits_for_every_job_and_delayed_accounting(monkeypatch):
    reports = iter(["10|COMPLETED|", "", "10|COMPLETED|\n11|COMPLETED|\n11_0|COMPLETED|"])
    waits = []
    monkeypatch.setattr(submit, "command", lambda *args: next(reports))
    monkeypatch.setattr(submit, "is_live", lambda *args: False)
    monkeypatch.setattr(submit.time, "sleep", waits.append)
    submit.wait_for_prerequisites("10,11", "saion")
    assert waits == [60, 60]


def test_missing_accounting_never_counts_as_success(monkeypatch):
    monkeypatch.setattr(submit, "command", lambda *args: "10|COMPLETED|")
    monkeypatch.setattr(submit, "is_live", lambda *args: False)
    monkeypatch.setattr(submit.time, "sleep", lambda *args: None)
    with pytest.raises(RuntimeError, match="could not be verified"):
        submit.wait_for_prerequisites("10,11", "saion")


def test_cancelled_array_task_fails_prerequisite(monkeypatch):
    monkeypatch.setattr(submit, "command", lambda *args: "10|RUNNING|\n10_0|CANCELLED by 123|")
    monkeypatch.setattr(submit.time, "sleep", lambda *args: pytest.fail("Waited on cancelled job"))
    with pytest.raises(RuntimeError, match="CANCELLED"):
        submit.wait_for_prerequisites("10", "saion")


def test_completed_array_elements_without_parent_records(monkeypatch):
    monkeypatch.setattr(submit, "command", lambda *args: "10_0|COMPLETED|\n11_0|COMPLETED|")
    monkeypatch.setattr(submit.time, "sleep", lambda *args: pytest.fail("Completed jobs were not recognized"))
    submit.wait_for_prerequisites("10,11", "saion")


def test_scheduler_outage_can_recover_after_more_than_five_failures(monkeypatch):
    failed = subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Unable to contact slurm controller (connect failure)")
    results = iter([failed] * 7 + [subprocess.CompletedProcess([], 0, "123\n", "")])
    waits = []
    monkeypatch.setattr(submit.subprocess, "run", lambda *args, **kwargs: next(results))
    monkeypatch.setattr(submit.time, "sleep", waits.append)
    assert submit.command(["squeue", "-h"]) == "123"
    assert waits == [60] * 7


def test_scheduler_timeout_is_retried(monkeypatch):
    attempts = []
    def run(*args, **kwargs):
        attempts.append(kwargs["timeout"])
        if len(attempts) == 1:
            raise subprocess.TimeoutExpired(args[0], 90)
        return subprocess.CompletedProcess([], 0, "123|COMPLETED|", "")
    monkeypatch.setattr(submit.subprocess, "run", run)
    monkeypatch.setattr(submit.time, "sleep", lambda *args: None)
    assert submit.command(["sacct", "-j", "123"]) == "123|COMPLETED|"
    assert attempts == [90, 90]


def test_permanent_scheduler_errors_still_fail(monkeypatch):
    monkeypatch.setattr(submit.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess([], 1, "", "Invalid user"))
    monkeypatch.setattr(submit.time, "sleep", lambda *args: None)
    with pytest.raises(RuntimeError, match="Invalid user"):
        submit.command(["squeue", "-u", "missing"])
