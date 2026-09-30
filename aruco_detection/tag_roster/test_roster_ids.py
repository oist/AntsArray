"""Tests for roster_ids.py on synthetic map_combine ArUco panorama pickles."""

import sys
from collections import Counter
from pathlib import Path

import cv2
import cv2.aruco as aruco
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import roster_ids as R  # noqa: E402

NPZ = HERE.parent / "custom_dicts" / "custom_4x4_A100_d4_20260410_103938.npz"
NF = 200          # frames per synthetic chunk
X0 = {"left": 100.0, "right": 2600.0}


def write_chunk(pano, chunk, side, seen):
    """seen = {tag id: number of frames it is detected in}; tags sit 20 px apart."""
    rows = [(f, iid, X0[side] + 20 * iid, 500.0, 0)
            for iid, n in seen.items() for f in range(n)]
    df = pd.DataFrame(rows, columns=["Frame", "Instance", "X", "Y", "Cam"])
    pd.to_pickle({"detections": df, "num_frames": NF},
                 pano / f"exp_chunk{chunk:03d}_aruco_panorama_x_{side}2465.pkl")


@pytest.fixture
def pano(tmp_path):
    """4 chunks. Left: 1 present, 2 dropped after chunk 1, 3 a ghost blip, 4 weak. Right: 5 present."""
    d = tmp_path / "pano"
    d.mkdir()
    for c in range(4):
        write_chunk(d, c, "left", {1: 150, 2: 150 if c < 2 else 0, 3: 1 if c == 3 else 0, 4: 10})
        write_chunk(d, c, "right", {5: 150})
    return d


def run(pano, tmp_path, *extra):
    out = tmp_path / "out"
    R.main(["--pano", str(pano), "--out", str(out), "--npz", str(NPZ),
            "--recent", "2", "--copies", "1", *extra])
    status = pd.read_csv(out / "roster_status.csv", keep_default_na=False)
    return out, {(r.side, r.id): r for r in status.itertuples()}


@pytest.mark.parametrize("rates,expected", [
    ([0.8, 0.7, 0.9, 0.6], "present"),
    ([0.8, 0.7, 0.0, 0.01], "dropped"),
    ([0.0, 0.0, 0.0, 0.005], "absent"),
    ([0.05, 0.05, 0.05, 0.05], "uncertain"),
    ([0.0, 0.0, 0.3, 0.05], "present"),     # tagged mid-window: recent mean 0.175
])
def test_classify(rates, expected):
    assert R.classify(rates, recent=2)[0] == expected


def test_status_per_side(pano, tmp_path):
    _, st = run(pano, tmp_path)
    assert st[("left", 1)].status == "present"
    assert st[("left", 2)].status == "dropped"
    assert st[("left", 3)].status == "absent"
    assert st[("left", 4)].status == "uncertain"
    assert st[("left", 0)].status == "absent"          # never detected at all
    assert st[("right", 5)].status == "present"
    assert st[("right", 1)].status == "absent"         # IDs are per colony
    assert len(st) == 200                              # 100 IDs x 2 sides


def test_dropped_id_reports_when_it_was_last_seen(pano, tmp_path):
    _, st = run(pano, tmp_path)
    # last present chunk 1, last frame 149 -> frame 349 of the block = 14.5 s at 24 fps
    assert st[("left", 2)].last_seen.startswith("+00:00:14")


def test_previous_present_now_absent_counts_as_dropped(pano, tmp_path):
    prev = tmp_path / "prev.csv"
    pd.DataFrame([{"side": "left", "id": 7, "status": "present"},
                  {"side": "right", "id": 5, "status": "present"}]).to_csv(prev, index=False)
    _, st = run(pano, tmp_path, "--previous", str(prev))
    assert st[("left", 7)].status == "dropped"
    assert st[("left", 7)].last_seen == "before this window"
    assert st[("right", 5)].status == "present"


def test_two_pickles_for_one_chunk_and_side_stop_the_run(pano, tmp_path):
    """A rerun with another split leaves ..._left2500.pkl next to ..._left2465.pkl: never mix them."""
    (pano / "exp_chunk000_aruco_panorama_x_left2465.pkl").rename(pano / "exp_chunk000_aruco_panorama_x_left2500.pkl")
    write_chunk(pano, 0, "left", {1: 150})
    with pytest.raises(SystemExit, match="chunk 000 left"):
        run(pano, tmp_path)


def test_a_chunk_missing_one_side_stops_the_run(pano, tmp_path):
    """Otherwise every ID of that side reads as 0 in that chunk and would be printed."""
    (pano / "exp_chunk003_aruco_panorama_x_right2465.pkl").unlink()
    with pytest.raises(SystemExit, match="chunk 003"):
        run(pano, tmp_path)


def test_needed_ids_warns_that_unseen_is_not_proof_of_a_lost_tag(pano, tmp_path):
    out, _ = run(pano, tmp_path)
    text = (out / "needed_ids.txt").read_text(encoding="utf-8")
    assert "hidden" in text and "window" in text


def test_sheets_hold_dropped_and_absent_ids_only(pano, tmp_path):
    out, st = run(pano, tmp_path)
    img = cv2.imread(str(out / "retag_left.png"), cv2.IMREAD_GRAYSCALE)
    params = aruco.DetectorParameters()
    params.errorCorrectionRate = 0.0
    dictionary = R.sheet.load_custom_dictionary(NPZ)[0]
    _, ids, _ = aruco.ArucoDetector(dictionary, params).detectMarkers(img)
    printed = Counter(int(i) for i in ids.ravel())
    expected = {i for (side, i), r in st.items() if side == "left" and r.status in ("dropped", "absent")}
    assert printed == Counter(expected)
    assert 1 not in printed and 4 not in printed       # present and uncertain are not printed
    assert (out / "retag_left.svg").exists() and (out / "retag_right.png").exists()
    text = (out / "needed_ids.txt").read_text(encoding="utf-8")
    assert "dropped    2" in text and "uncertain  4" in text
