#!/usr/bin/env python3
"""review.html for a tag_photos scan: the interactive page of review_page.js with every photo, read
and dictionary code embedded, so it works straight from disk (file://) without a server.

Photos are embedded as data URIs because a page opened from disk may not read the pixels of image
files beside it (the canvas is "tainted"), and the manual tag finder needs those pixels.
"""
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

JS_FILE = Path(__file__).with_name("review_page.js")
EMBED_LONG_SIDE = 2048      # tags stay >= ~40 px; 7 photos make a ~6 MB page
CSS = """
body{font-family:sans-serif;margin:16px;max-width:1200px}
h2{margin-top:36px;border-top:2px solid #ccc;padding-top:12px}
table{border-collapse:collapse;margin:8px 0 12px}td,th{border:1px solid #bbb;padding:3px;text-align:center}
.ok{background:#e3f6e3}.check{background:#fff1d6}.unreadable{background:#fde2e2}.manual{background:#f3e3fb}
canvas.view{max-width:100%;cursor:crosshair;display:block}canvas.zoom{border:1px solid #999;cursor:crosshair}
.toolbar{position:sticky;top:0;background:#fff;padding:6px 0;border-bottom:1px solid #ddd;z-index:2}
.toolbar button{font-size:15px;font-weight:bold}.msg{color:#06c;margin-left:8px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}.card{text-align:center;font-size:13px}
.card canvas{border:1px solid #ccc;display:block;margin:0 auto 4px}.hint{color:#555}
.warn{color:#b00;font-weight:bold}
"""
HELP = (
    "<p>Green = <b>ok</b> (accept is already y), orange = <b>check</b>, red = <b>unreadable</b>. "
    "In each table choose accept <b>y</b> or <b>n</b>, or type the ID you read into id_override. "
    "<b>Missed a tag?</b> Click near it on the photo, then click the four outer corners of its black "
    "border in the zoom view: the page reads the tag from those corners and lists candidate IDs "
    "(each beside the dictionary drawing, turned the way the tag lies). <b>use</b> adds it as a "
    "manual row. When done, click <b>Save review.csv</b>, save it over review.csv in this folder, "
    "and run apply. Until then decisions are kept in this browser. Decide either here or in Excel, "
    "not both: saving from this page replaces review.csv.</p>")


def embed_photo(color):
    """(data URI of the upright photo as JPEG, at most EMBED_LONG_SIDE px; scale original -> embedded)."""
    f = min(1.0, EMBED_LONG_SIDE / max(color.shape[:2]))
    img = color if f == 1.0 else cv2.resize(color, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise OSError("cannot encode the photo for review.html")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"), f


def page_reads(reads):
    """Outline of every read for the page (n = the row's n in review.csv; original pixels)."""
    return [{"n": n, "quad": np.round(np.asarray(r.quad, float), 1).tolist(), "status": r.status,
             "tag_id": r.tag_id} for n, r in enumerate(reads, 1)]


def page_data(stamp, columns, refs, pages, rows):
    refs = np.asarray(refs)
    return {"stamp": stamp, "columns": list(columns), "rows": rows, "photos": pages,
            "refs": refs.reshape(len(refs), refs.shape[1], 16).astype(int).tolist()}


def write_html(path, stamp, columns, refs, pages, rows, failed):
    blob = json.dumps(page_data(stamp, columns, refs, pages, rows), separators=(",", ":"))
    blob = blob.replace("</", "<\\/")                   # a note can never close the script element
    e = html.escape
    failed_html = ("<h2>Photos that could not be read</h2><ul>"
                   + "".join(f"<li>{e(f)}</li>" for f in failed) + "</ul>") if failed else ""
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width, initial-scale=1'>",
             f"<title>Tag photo review {e(stamp)}</title><style>{CSS}</style></head><body>",
             f"<h1>Tag photo review {e(stamp)}</h1>", HELP, failed_html,
             "<div id='app'></div><noscript>This page needs JavaScript.</noscript>",
             f"<script type='application/json' id='data'>{blob}</script>",
             f"<script>{JS_FILE.read_text(encoding='utf-8')}</script></body></html>"]
    Path(path).write_text("\n".join(parts), encoding="utf-8")
