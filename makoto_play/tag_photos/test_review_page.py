"""Tests for review_page: the data embedded in review.html and the in-browser tag reader
(review_page.js, run with node)."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import photo_decode as P  # noqa: E402
import review_page as RP  # noqa: E402
import synth_scene as S  # noqa: E402
import tag_photos as TP  # noqa: E402

NPZ = P.DEFAULT_NPZ
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
RUNNER = """
const m = require(process.argv[2]);
const p = JSON.parse(require('fs').readFileSync(0, 'utf8'));
let out;
if (p.csv) out = m.toCSV(p.columns, p.rows);
else {
  const img = {w: p.w, h: p.h, data: Uint8Array.from(p.data)};
  out = p.quads.map(q => m.decodeQuad(img, m.orderCorners(q), p.refs));
}
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def dec():
    return P.load_decoder(NPZ)


def js_refs(dec):
    return dec.refs.reshape(len(dec.refs), 4, 16).astype(int).tolist()


def run_js(tmp_path, payload):
    script = tmp_path / "runner.js"
    script.write_text(RUNNER, encoding="utf-8")
    done = subprocess.run([NODE, str(script), str(RP.JS_FILE)], input=json.dumps(payload),
                          capture_output=True, encoding="utf-8", check=True, timeout=120)
    return json.loads(done.stdout)


def decode_js(tmp_path, dec, gray, quads):
    payload = {"w": gray.shape[1], "h": gray.shape[0], "data": gray.ravel().tolist(),
               "quads": [np.asarray(q, float).tolist() for q in quads], "refs": js_refs(dec)}
    return run_js(tmp_path, payload)


def tag_scene(dec, tag_id, mirrored=False):
    """A 400x400 photo of one tag and its true corners (from the scan of the clean photo)."""
    gray = S.scene_with_tags(dec.dictionary, [(tag_id, (200, 200), 60, 25, 0.05, mirrored)], shape=(400, 400))
    quad = next(r.quad for r in P.decode_photo(gray, dec) if r.tag_id == tag_id)
    return gray, quad


def test_html_embeds_rows_photos_and_dictionary(tmp_path, dec):
    rows = [{"photo": "C22_dead/IMG_1.JPEG", "n": 1, "tag_id": 5, "note": "</script><b>x</b>"}]
    pages = [{"photo": "C22_dead/IMG_1.JPEG", "img": "data:image/jpeg;base64,AAAA", "scale": 1.0, "reads": []}]
    out = tmp_path / "review.html"
    RP.write_html(out, "20261020_120000", TP.COLUMNS, dec.refs, pages, rows, ["C22_dead/bad.JPEG: OSError"])
    text = out.read_text(encoding="utf-8")
    blob = re.search(r"<script type='application/json' id='data'>(.*?)</script>", text, re.S).group(1)
    data = json.loads(blob)
    assert data["rows"] == rows and data["photos"] == pages and data["columns"] == TP.COLUMNS
    assert np.array_equal(np.array(data["refs"]).reshape(dec.refs.shape), dec.refs)
    assert "could not be read" in text and "function decodeQuad" in text


def test_embedded_photo_is_capped_and_scaled():
    img, scale = RP.embed_photo(np.zeros((3000, 4000, 3), np.uint8))
    assert img.startswith("data:image/jpeg;base64,") and scale == pytest.approx(RP.EMBED_LONG_SIDE / 4000)


@needs_node
def test_js_reads_a_tag_from_its_corners(tmp_path, dec):
    gray, quad = tag_scene(dec, 61)
    best = decode_js(tmp_path, dec, gray, [quad])[0]["candidates"][0]
    assert (best["id"], best["mirrored"], best["d"]) == (61, False, 0) and best["votes"] >= 20
    assert best["rot"] == P.nearest(P.read_grid(gray, quad).bits, dec.refs)[3]    # same orientation as Python


@needs_node
def test_js_tolerates_clicks_a_few_pixels_off(tmp_path, dec):
    gray, quad = tag_scene(dec, 28)
    rng = np.random.default_rng(3)
    quads = [quad + rng.uniform(-3, 3, quad.shape) for _ in range(5)]
    assert [r["candidates"][0]["id"] for r in decode_js(tmp_path, dec, gray, quads)] == [28] * 5


@needs_node
def test_js_reads_a_tag_with_a_leg_across_it(tmp_path, dec):
    gray, quad = tag_scene(dec, 39)
    cv2.line(gray, (120, 260), (280, 150), 25, 5)
    assert decode_js(tmp_path, dec, gray, [quad])[0]["candidates"][0]["id"] == 39


@needs_node
def test_js_flags_a_tag_seen_from_behind(tmp_path, dec):
    gray, quad = tag_scene(dec, 8, mirrored=True)
    best = decode_js(tmp_path, dec, gray, [quad])[0]["candidates"][0]
    assert (best["id"], best["mirrored"]) == (8, True)


@needs_node
def test_js_page_helpers(tmp_path):
    script = tmp_path / "helpers.js"
    script.write_text("""
const m = require(process.argv[2]);
const manual = [{photo: 'A', n: 'm1'}, {photo: 'A', n: 'm3'}, {photo: 'B', n: 'm7'}];
const sq = [[0, 0], [10, 0], [10, 10], [0, 10]];
process.stdout.write(JSON.stringify({
  next: [m.nextManualN(manual, 'A'), m.nextManualN(manual, 'C')],
  inside: [m.insideQuad([5, 5], sq), m.insideQuad([15, 5], sq)],
  sig: m.rowsSignature([{photo: 'A', n: 1, tag_id: 5, crop: 'c'}]),
}));""", encoding="utf-8")
    out = json.loads(subprocess.run([NODE, str(script), str(RP.JS_FILE)], capture_output=True, encoding="utf-8",
                                    check=True, timeout=60).stdout)
    assert out == {"next": ["m4", "m1"], "inside": [True, False], "sig": "A|1|5|c"}


@needs_node
def test_js_csv_is_read_back_by_apply(tmp_path, dec):
    rows = [{c: "" for c in TP.COLUMNS} | {"photo": "C22_dead/IMG_1.JPEG", "n": "m1", "id_override": "40",
                                           "note": 'comma, "quote"', "status": "manual"}]
    path = tmp_path / "review.csv"
    path.write_text(run_js(tmp_path, {"csv": True, "columns": TP.COLUMNS, "rows": rows}), encoding="utf-8",
                    newline="")
    assert TP.read_review(path) == rows
