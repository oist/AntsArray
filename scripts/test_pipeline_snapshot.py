from pathlib import Path
import pytest
from scripts.pipeline_snapshot import snapshot, verify


def test_queued_code_survives_source_edits_and_records_modifications(tmp_path):
    repo = tmp_path / 'repo'; (repo / 'analysis').mkdir(parents=True)
    source = repo / 'analysis/report.py'; source.write_text('version = 1\n')
    frozen = snapshot(repo, tmp_path / 'jobs')
    source.write_text('version = 2\n')
    assert (frozen / 'analysis/report.py').read_text() == 'version = 1\n'
    verify(frozen)
    copied = frozen / 'analysis/report.py'; copied.chmod(0o644); copied.write_text('version = 3\n')
    with pytest.raises(RuntimeError, match='snapshot was modified'):
        verify(frozen)


def test_snapshot_refuses_source_changed_while_copying(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'; (repo / 'analysis').mkdir(parents=True)
    source = repo / 'analysis/report.py'; source.write_text('version = 1\n')
    write = Path.write_bytes
    def concurrent_edit(path, data):
        result = write(path, data)
        if path.name == 'report.py':
            source.write_text('version = 2\n')
        return result
    monkeypatch.setattr(Path, 'write_bytes', concurrent_edit)
    with pytest.raises(RuntimeError, match='changed while taking'):
        snapshot(repo, tmp_path / 'jobs')
    assert not list((tmp_path / 'jobs').iterdir())
