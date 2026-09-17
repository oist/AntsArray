from pathlib import Path
import os

import pytest

from camera_cal.region_paths import panorama_regions_path


@pytest.mark.parametrize("suffix", [".json", ".csv"])
def test_blocks_read_shared_annotations_but_save_local_copies(tmp_path, suffix):
    dataset = tmp_path / "20260810"
    dataset.mkdir()
    shared = dataset / f"panorama_regions{suffix}"
    shared.write_text("shared")
    for name in ("block01", "block02", "block02-w000-031", "block02-w149-197"):
        block = dataset / name
        block.mkdir()
        assert panorama_regions_path(block, suffix) == shared
        local = block / shared.name
        assert panorama_regions_path(block, suffix, for_write=True) == local
        local.write_text("new block annotations")
        assert panorama_regions_path(block, suffix) == local
    assert panorama_regions_path(dataset, suffix) == shared


def test_existing_local_annotations_remain_local(tmp_path):
    block = tmp_path / "20260723" / "block02"
    block.mkdir(parents=True)
    legacy = block / "panorama_regions.json"
    legacy.write_text("legacy")
    assert panorama_regions_path(block, ".json") == legacy
    assert panorama_regions_path(block, ".json", for_write=True) == legacy


def test_missing_returns_expected_local_path_without_borrowing_other_blocks(tmp_path):
    block = tmp_path / "20260810" / "block02"
    sibling = block.with_name("block01")
    sibling.mkdir(parents=True)
    (sibling / "panorama_regions.csv").touch()
    (tmp_path / "panorama_regions.csv").touch()
    assert panorama_regions_path(block) == block / "panorama_regions.csv"


def test_non_date_parent_preserves_existing_layout():
    block = Path("/project/experiment/block02")
    assert panorama_regions_path(block, for_write=True) == block / "panorama_regions.csv"


def test_unsupported_format_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=".csv or .json"):
        panorama_regions_path(tmp_path, ".png")


def annotation(root, relative, *, mtime=100):
    path = root / relative / "panorama_regions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    os.utime(path, (mtime, mtime))
    return path


def test_tracking_fallback_ranks_recording_date_before_file_mtime(tmp_path):
    block = tmp_path / "20260515/block03"
    block.mkdir(parents=True)
    annotation(tmp_path, "20260501/block02", mtime=999)
    expected = annotation(tmp_path, "20260514/block01", mtime=100)
    annotation(tmp_path, "20260516/block01", mtime=2000)
    annotation(tmp_path, "20260515/block01", mtime=2000)
    assert panorama_regions_path(block, search_earlier_dates=True) == expected
    # The GUI and analysis helpers still opt into historical borrowing explicitly.
    assert panorama_regions_path(block) == block / "panorama_regions.csv"


def test_fallback_skips_empty_dates_and_uses_latest_file_within_date(tmp_path):
    block = tmp_path / "20260515/block03"
    block.mkdir(parents=True)
    (tmp_path / "20260514/block01").mkdir(parents=True)
    annotation(tmp_path, "20260513", mtime=100)
    annotation(tmp_path, "20260513/block01", mtime=200)
    expected = annotation(tmp_path, "20260513/block02", mtime=300)
    assert panorama_regions_path(block, search_earlier_dates=True) == expected


@pytest.mark.parametrize("dated_name", ["20260515", "20260515_stim", "20260515-control"])
def test_current_annotations_and_write_destination_take_priority(tmp_path, dated_name):
    block = tmp_path / dated_name / "block03"
    block.mkdir(parents=True)
    annotation(tmp_path, "20260514/block02", mtime=999)
    shared = annotation(tmp_path, dated_name)
    assert panorama_regions_path(block, search_earlier_dates=True) == shared
    assert panorama_regions_path(block, search_earlier_dates=True, for_write=True) == block / shared.name
    local = annotation(tmp_path, f"{dated_name}/block03")
    assert panorama_regions_path(block, search_earlier_dates=True) == local


def test_historical_lookup_supports_suffixed_dates_and_date_level_recordings(tmp_path):
    recording = tmp_path / "20260515_stim"
    recording.mkdir()
    annotation(tmp_path, "20260513_control", mtime=900)
    annotation(tmp_path, "20260514_a", mtime=100)
    expected = annotation(tmp_path, "20260514_b/block01", mtime=200)
    assert panorama_regions_path(recording, search_earlier_dates=True) == expected


def test_no_eligible_history_never_borrows_future_same_day_or_invalid_dates(tmp_path):
    block = tmp_path / "20260515/block03"
    block.mkdir(parents=True)
    for relative in ["20260516/block01", "20260515/block01", "20260515_other/block01",
                     "20260230/block01", "notes/block01"]:
        annotation(tmp_path, relative)
    assert panorama_regions_path(block, search_earlier_dates=True) == block / "panorama_regions.csv"


def test_undated_recording_does_not_search_unrelated_history(tmp_path):
    block = tmp_path / "experiment/block03"
    block.mkdir(parents=True)
    annotation(tmp_path, "20260514/block01")
    assert panorama_regions_path(block, search_earlier_dates=True) == block / "panorama_regions.csv"
