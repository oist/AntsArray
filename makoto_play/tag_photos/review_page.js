/* Interactive review page for tag_photos scan results (inlined into review.html by review_page.py).

   Decide each read (accept y / n, or type the ID into id_override), and find tags the scan missed:
   click near the tag, then click the four outer corners of its black border in the zoom view. The
   tag is read from those corners - perspective warp to an 8x8 grid (white margin ring, black
   border ring, 4x4 bits), median of 9 samples per cell, Otsu threshold - and matched against every
   dictionary code in 4 rotations, also mirrored (a tag seen from behind). The corners are shifted
   and scaled slightly 27 times and every read within 2 bits of a code votes, so a click a few
   pixels off still finds the ID. "Save review.csv" writes all decisions for `apply`.

   The pure functions at the top are exported for the node tests (test_review_page.py). */
"use strict";

const JITTER_SCALES = [0.95, 1, 1.05];
const JITTER_SHIFT = 0.04;               // of the mean side
const VOTE_D = 2;                        // max Hamming distance of a voting read
const VOTE_BORDER = 0.6;                 // min fraction of black border cells of a voting read
const VOTE_MARGIN = 0.6;                 // min fraction of white margin cells (a tag has its white film around it)
const OFFSETS = [0.3, 0.5, 0.7];         // sample points inside a cell
const LOW_VOTES = 5;                     // fewer agreeing reads than this: shown as low confidence
const ZOOMS = [2, 3, 4];
const ZOOM_PX = 480;
const VIEW_PX = 900;

function orderCorners(points) {
  /* Clockwise on screen (y down), as OpenCV orders marker corners; the start corner does not
     matter because every rotation is matched. */
  const cx = points.reduce((s, p) => s + p[0], 0) / points.length;
  const cy = points.reduce((s, p) => s + p[1], 0) / points.length;
  const angle = (p) => Math.atan2(p[1] - cy, p[0] - cx);
  return points.slice().sort((a, b) => angle(a) - angle(b));
}

function homography(q) {
  /* (u, v) in the unit square -> image point; (0,0) q[0], (1,0) q[1], (1,1) q[2], (0,1) q[3]. */
  const [x0, y0] = q[0], [x1, y1] = q[1], [x2, y2] = q[2], [x3, y3] = q[3];
  const dx1 = x1 - x2, dx2 = x3 - x2, dy1 = y1 - y2, dy2 = y3 - y2;
  const sx = x0 - x1 + x2 - x3, sy = y0 - y1 + y2 - y3;
  let g = 0, h = 0;
  if (Math.abs(sx) > 1e-9 || Math.abs(sy) > 1e-9) {
    const den = dx1 * dy2 - dx2 * dy1;
    g = (sx * dy2 - dx2 * sy) / den;
    h = (dx1 * sy - sx * dy1) / den;
  }
  const a = x1 - x0 + g * x1, b = x3 - x0 + h * x3, d = y1 - y0 + g * y1, e = y3 - y0 + h * y3;
  return (u, v) => {
    const w = g * u + h * v + 1;
    return [(a * u + b * v + x0) / w, (d * u + e * v + y0) / w];
  };
}

function pixel(img, x, y) {
  /* Bilinear grey value; img = {w, h, data: Uint8Array}. */
  const xi = Math.min(Math.max(x, 0), img.w - 1.001), yi = Math.min(Math.max(y, 0), img.h - 1.001);
  const x0 = Math.floor(xi), y0 = Math.floor(yi), fx = xi - x0, fy = yi - y0;
  const i = y0 * img.w + x0, p = img.data;
  return p[i] * (1 - fx) * (1 - fy) + p[i + 1] * fx * (1 - fy)
       + p[i + img.w] * (1 - fx) * fy + p[i + img.w + 1] * fx * fy;
}

function otsu(values) {
  const v = values.slice().sort((a, b) => a - b), n = v.length;
  const total = v.reduce((s, x) => s + x, 0);
  let best = -1, thr = (v[0] + v[n - 1]) / 2, sum0 = 0;
  for (let k = 1; k < n; k++) {
    sum0 += v[k - 1];
    const w0 = k / n, m0 = sum0 / k, m1 = (total - sum0) / (n - k);
    const between = w0 * (1 - w0) * (m0 - m1) ** 2;
    if (between > best) { best = between; thr = (v[k - 1] + v[k]) / 2; }
  }
  return thr;
}

function sampleGrid(img, quad) {
  /* 8x8 cells: the 6x6 marker outlined by quad plus one margin cell around it. */
  const H = homography(quad), cells = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const vals = [];
      for (const oy of OFFSETS) for (const ox of OFFSETS) {
        const [x, y] = H((c - 1 + ox) / 6, (r - 1 + oy) / 6);
        vals.push(pixel(img, x, y));
      }
      vals.sort((a, b) => a - b);
      cells.push(vals[4]);
    }
  }
  const at = (r, c) => cells[r * 8 + c];
  const inner = [];
  for (let r = 1; r < 7; r++) for (let c = 1; c < 7; c++) inner.push(at(r, c));
  const thr = otsu(inner);
  const bits = [];
  for (let r = 2; r < 6; r++) for (let c = 2; c < 6; c++) bits.push(at(r, c) < thr ? 1 : 0);
  let border = 0, margin = 0;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      if (r === 0 || r === 7 || c === 0 || c === 7) margin += at(r, c) >= thr ? 1 : 0;
      else if (r === 1 || r === 6 || c === 1 || c === 6) border += at(r, c) < thr ? 1 : 0;
    }
  }
  return { bits, border: border / 20, margin: margin / 28 };
}

function fliplr(bits) {
  const out = [];
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out.push(bits[r * 4 + 3 - c]);
  return out;
}

function hamming(a, b) {
  let d = 0;
  for (let i = 0; i < 16; i++) d += a[i] !== b[i] ? 1 : 0;
  return d;
}

function matchBits(bits, refs) {
  /* Best (distance, rotation, mirrored) per ID, nearest first; refs[id][rot] = 16 bits, 1 = black.
     A mirrored match counts only when it is strictly better than the bits as seen. */
  const mirrored = fliplr(bits);
  const out = refs.map((rots, id) => {
    let best = { id, d: 99, rot: 0, mirrored: false };
    rots.forEach((ref, rot) => {
      const dp = hamming(bits, ref), dm = hamming(mirrored, ref);
      if (dp < best.d || (dp === best.d && best.mirrored)) best = { id, d: dp, rot, mirrored: false };
      if (dm < best.d) best = { id, d: dm, rot, mirrored: true };
    });
    return best;
  });
  return out.sort((a, b) => a.d - b.d || a.mirrored - b.mirrored);
}

function variants(quad) {
  const cx = quad.reduce((s, p) => s + p[0], 0) / 4, cy = quad.reduce((s, p) => s + p[1], 0) / 4;
  const side = quad.reduce((s, p, i) => s + Math.hypot(p[0] - quad[(i + 1) % 4][0], p[1] - quad[(i + 1) % 4][1]), 0) / 4;
  const shift = JITTER_SHIFT * side, out = [];
  for (const k of JITTER_SCALES) {
    for (const dy of [-shift, 0, shift]) {
      for (const dx of [-shift, 0, shift]) {
        out.push(quad.map(([x, y]) => [cx + (x - cx) * k + dx, cy + (y - cy) * k + dy]));
      }
    }
  }
  return out;
}

function decodeQuad(img, quad, refs) {
  /* {candidates: [{id, votes, of, d, rot, mirrored}], closest: nearest codes of the clicked
     corners, grid: the read at the clicked corners}. */
  const all = variants(quad), votes = new Map();
  for (const q of all) {
    const g = sampleGrid(img, q);
    if (g.border < VOTE_BORDER || g.margin < VOTE_MARGIN) continue;
    const top = matchBits(g.bits, refs)[0];
    if (top.d > VOTE_D) continue;
    const key = `${top.id}${top.mirrored ? "m" : ""}`;
    const v = votes.get(key) || { id: top.id, votes: 0, of: all.length, d: 99, rot: top.rot, mirrored: top.mirrored };
    v.votes += 1;
    if (top.d < v.d) { v.d = top.d; v.rot = top.rot; }
    votes.set(key, v);
  }
  const grid = sampleGrid(img, quad);
  return {
    candidates: [...votes.values()].sort((a, b) => b.votes - a.votes || a.d - b.d),
    closest: matchBits(grid.bits, refs).slice(0, 3),
    grid,
  };
}

function expectedBits(refs, cand) {
  /* The candidate's code turned (and mirrored) the way the tag was read, for drawing. */
  const ref = refs[cand.id][cand.rot];
  return cand.mirrored ? fliplr(ref) : ref;
}

function toCSV(columns, rows) {
  const esc = (v) => {
    const s = v === undefined || v === null ? "" : String(v);
    return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [columns.join(","), ...rows.map((r) => columns.map((c) => esc(r[c])).join(","))];
  return "﻿" + lines.join("\r\n") + "\r\n";
}

function nextManualN(manual, photo) {
  /* m<k> one above the highest manual number of the photo, so a removed row's number is not reused. */
  const used = manual.filter((m) => m.photo === photo).map((m) => parseInt(String(m.n).slice(1), 10)).filter((k) => k > 0);
  return `m${used.length ? Math.max(...used) + 1 : 1}`;
}

function insideQuad(pt, quad) {
  /* Ray casting: is the point inside the polygon? */
  let inside = false;
  for (let i = 0, j = quad.length - 1; i < quad.length; j = i++) {
    const [xi, yi] = quad[i], [xj, yj] = quad[j];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function rowsSignature(rows) {
  /* Identifies the scan a saved browser state belongs to (a rebuilt page discards old decisions). */
  return rows.map((r) => `${r.photo}|${r.n}|${r.tag_id}|${r.crop}`).join(";");
}

if (typeof module !== "undefined") {
  module.exports = { orderCorners, homography, otsu, sampleGrid, fliplr, matchBits, decodeQuad, expectedBits, toCSV,
                     nextManualN, insideQuad, rowsSignature };
}

/* ------------------------------------------------------------------ page */

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) e.appendChild(c);
  return e;
}

function drawBits(canvas, bits, cell = 12) {
  /* 8x8 drawing: white margin, black border, the 4x4 bits. */
  canvas.width = canvas.height = 8 * cell;
  const g = canvas.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0, 0, 8 * cell, 8 * cell);
  g.fillStyle = "#000"; g.fillRect(cell, cell, 6 * cell, 6 * cell);
  g.fillStyle = "#fff";
  bits.forEach((b, i) => { if (!b) g.fillRect((2 + (i % 4)) * cell, (2 + Math.floor(i / 4)) * cell, cell, cell); });
  return canvas;
}

function drawWarp(canvas, img, quad, size = 96) {
  /* The tag straightened by the clicked corners, margin included. */
  canvas.width = canvas.height = size;
  const g = canvas.getContext("2d"), out = g.createImageData(size, size), H = homography(quad);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const [px, py] = H((x + 0.5) / size * 8 / 6 - 1 / 6, (y + 0.5) / size * 8 / 6 - 1 / 6);
      const v = pixel(img, px, py), i = 4 * (y * size + x);
      out.data[i] = out.data[i + 1] = out.data[i + 2] = v; out.data[i + 3] = 255;
    }
  }
  g.putImageData(out, 0, 0);
  return canvas;
}

function browserMain() {
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const KEY = `tag_photos_review_${DATA.stamp}`;
  const sig = rowsSignature(DATA.rows);
  const saved = loadState(KEY, sig);
  const state = saved || { sig, rows: DATA.rows.map((r) => ({ ...r })), manual: [], dates: {} };
  const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* private window: decisions live until reload */ } };
  const app = document.getElementById("app");
  const sections = [];
  /* a typed value only relabels the photos; adding or removing a row rebuilds the tables */
  const changed = (what) => {
    persist();
    sections.forEach((s) => (what === "field" ? s.redrawView() : s.redraw()));
  };
  app.appendChild(toolbar(DATA, state, saved ? KEY : null));
  DATA.photos.forEach((p) => {
    const s = photoSection(DATA, p, state, changed);
    sections.push(s);
    app.appendChild(s.node);
  });
  app.appendChild(toolbar(DATA, state, null));
}

function loadState(key, sig) {
  try {
    const s = JSON.parse(localStorage.getItem(key) || "null");
    if (!s || s.sig !== sig || !Array.isArray(s.rows) || !Array.isArray(s.manual)) return null;
    return { ...s, dates: s.dates || {} };
  } catch (e) {
    return null;
  }
}

function toolbar(DATA, state, restoredKey) {
  const msg = el("span", { class: "msg" });
  const folder = decodeURIComponent(location.pathname).replace(/^\/([A-Za-z]:)/, "$1").replace(/[^/\\]*$/, "");
  const save = async () => {
    const text = toCSV(DATA.columns, [...state.rows, ...state.manual]);
    if (window.showSaveFilePicker) {
      try {
        const handle = await window.showSaveFilePicker({ suggestedName: "review.csv", types: [{ description: "CSV", accept: { "text/csv": [".csv"] } }] });
        const w = await handle.createWritable();
        await w.write(text);
        await w.close();
        msg.textContent = `saved ${handle.name} - now run apply`;
        return;
      } catch (e) {
        if (e.name === "AbortError") return;
      }
    }
    const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    el("a", { href: url, download: "review.csv" }).click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    msg.textContent = `downloaded - move it into ${folder} as review.csv, replacing the old one `
      + "(rename it if the browser called it 'review (1).csv'), then run apply";
  };
  const parts = [el("button", { text: "Save review.csv", onclick: save }), el("span", { text: ` save it as review.csv in ${folder} ` }), msg];
  if (restoredKey) {
    parts.push(el("div", { class: "msg" }, [
      el("span", { text: `Decisions restored from this browser (${state.manual.length} manual rows). ` }),
      el("button", { text: "start over from the scan", onclick: () => { try { localStorage.removeItem(restoredKey); } catch (e) { /* nothing stored */ } location.reload(); } })]));
  }
  return el("div", { class: "toolbar" }, parts);
}

function photoSection(DATA, p, state, changed) {
  const node = el("section");
  node.appendChild(el("h2", { text: `${p.photo} · colony ${p.colony} · ${p.kind} · ${p.taken} (${p.source})` }));
  if (p.source !== "exif") {
    const date = el("input", { type: "date", value: state.dates[p.photo] || "", onchange: (e) => {
      state.dates[p.photo] = e.target.value;
      [...state.rows, ...state.manual].filter((r) => r.photo === p.photo).forEach((r) => { r.date_override = e.target.value; });
      changed("field");
    } });
    node.appendChild(el("p", {}, [el("b", { text: "This photo has no capture date. Date found: " }), date]));
  }
  const view = el("canvas", { class: "view" });
  const tableHolder = el("div");
  const finder = el("div", { class: "finder" });
  node.append(view, tableHolder, finder);
  const image = new Image();
  let grey = null;
  const ready = new Promise((res) => { image.onload = res; });
  image.src = p.img;
  const greyImage = () => {
    if (!grey) {
      const c = el("canvas"); c.width = image.naturalWidth; c.height = image.naturalHeight;
      const g = c.getContext("2d"); g.drawImage(image, 0, 0);
      const rgba = g.getImageData(0, 0, c.width, c.height).data, data = new Uint8Array(c.width * c.height);
      for (let i = 0; i < data.length; i++) data[i] = Math.round(0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2]);
      grey = { w: c.width, h: c.height, data };
    }
    return grey;
  };
  const finderUI = makeFinder(DATA, p, state, finder, greyImage, image, changed);
  const redrawView = () => ready.then(() => drawView(view, image, p, state, finderUI.onPick));
  const redraw = () => {
    redrawView();
    tableHolder.replaceChildren(rowsTable(DATA, p, state, changed));
  };
  redraw();
  return { node, redraw, redrawView };
}

function drawView(canvas, image, p, state, onPick) {
  const f = Math.min(1, VIEW_PX / image.naturalWidth);
  canvas.width = Math.round(image.naturalWidth * f); canvas.height = Math.round(image.naturalHeight * f);
  const g = canvas.getContext("2d");
  g.drawImage(image, 0, 0, canvas.width, canvas.height);
  const colors = { ok: "#2a2", check: "#e80", unreadable: "#d22", manual: "#a2c" };
  const outline = (quad, color, label) => {
    g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
    quad.forEach(([x, y], i) => (i ? g.lineTo : g.moveTo).call(g, x * p.scale * f, y * p.scale * f));
    g.closePath(); g.stroke();
    const top = Math.min(...quad.map((q) => q[1])) * p.scale * f, left = Math.min(...quad.map((q) => q[0])) * p.scale * f;
    g.font = "bold 13px sans-serif"; const w = g.measureText(label).width;
    g.fillStyle = "#fff"; g.fillRect(left, top - 18, w + 6, 16);
    g.fillStyle = color; g.fillText(label, left + 3, top - 5);
  };
  p.reads.forEach((r) => {
    const row = state.rows.find((x) => x.photo === p.photo && String(x.n) === String(r.n));
    outline(r.quad, colors[r.status], `#${r.n} ${row && row.id_override ? row.id_override : (r.tag_id ?? "?")}`);
  });
  state.manual.filter((m) => m.photo === p.photo && m.quad).forEach((m) => outline(m.quad, colors.manual, `${m.n} ${m.id_override}`));
  canvas.onclick = (ev) => {
    const rect = canvas.getBoundingClientRect();
    onPick((ev.clientX - rect.left) * canvas.width / rect.width / f, (ev.clientY - rect.top) * canvas.height / rect.height / f);
  };
}

function rowsTable(DATA, p, state, changed) {
  const rows = state.rows.filter((r) => r.photo === p.photo);
  const manual = state.manual.filter((m) => m.photo === p.photo);
  if (!rows.length && !manual.length) return el("p", { text: "No tag found yet - click a missed tag on the photo above." });
  const head = el("tr", {}, ["n", "status", "ID", "votes", "competing", "photo", "dictionary", "context", "accept", "id_override", ""].map((h) => el("th", { text: h })));
  const table = el("table", {}, [head]);
  const img = (src) => (src ? el("img", { src, width: 96 }) : el("span"));
  const crop = (r, suffix) => `crops/${encodeURIComponent(r.crop)}${suffix}`;
  for (const r of rows) {
    const accept = el("select", { onchange: (e) => { r.accept = e.target.value; changed("field"); } },
      ["", "y", "n"].map((v) => el("option", { value: v, text: v || "-" })));
    accept.value = r.accept || "";
    const override = el("input", { value: r.id_override || "", size: 4, onchange: (e) => { r.id_override = e.target.value.trim(); changed("field"); } });
    table.appendChild(el("tr", { class: r.status }, [
      el("td", { text: String(r.n) }), el("td", { text: r.status }), el("td", {}, [el("b", { text: String(r.tag_id === "" ? "?" : r.tag_id) })]),
      el("td", { text: String(r.votes) }), el("td", { text: r.alt || "" }),
      el("td", {}, [img(crop(r, "_warp.png"))]), el("td", {}, [img(r.tag_id === "" ? "" : crop(r, "_ref.png"))]),
      el("td", {}, [img(crop(r, "_ctx.jpg"))]), el("td", {}, [accept]), el("td", {}, [override]), el("td")]));
  }
  for (const m of manual) {
    table.appendChild(el("tr", { class: "manual" }, [
      el("td", { text: m.n }), el("td", { text: "manual" }), el("td", {}, [el("b", { text: m.id_override })]),
      el("td"), el("td", { text: m.alt }), el("td"), el("td"), el("td"),
      el("td", { text: "" }), el("td", { text: m.id_override }),
      el("td", {}, [el("button", { text: "remove", onclick: () => { state.manual.splice(state.manual.indexOf(m), 1); changed("rows"); } })])]));
  }
  return table;
}

function makeFinder(DATA, p, state, holder, greyImage, image, changed) {
  let zoom = 3, center = null, points = [];
  const zoomCanvas = el("canvas", { class: "zoom", width: ZOOM_PX, height: ZOOM_PX });
  const result = el("div", { class: "result" });
  const hint = el("p", { class: "hint", text: "Missed a tag? Click near it on the photo above." });
  const zoomButtons = el("span", {}, ZOOMS.map((z) => el("button", { text: `${z}x`, onclick: () => { zoom = z; points = []; result.replaceChildren(); drawZoom(); } })));
  const undo = el("button", { text: "undo corner", onclick: () => { points.pop(); drawZoom(); result.replaceChildren(); } });
  holder.append(hint);
  const src = () => {
    const s = ZOOM_PX / zoom;
    const x0 = Math.min(Math.max(center[0] - s / 2, 0), Math.max(image.naturalWidth - s, 0));
    const y0 = Math.min(Math.max(center[1] - s / 2, 0), Math.max(image.naturalHeight - s, 0));
    return { x0, y0, s };
  };
  const drawZoom = () => {
    const { x0, y0, s } = src(), g = zoomCanvas.getContext("2d");
    g.imageSmoothingEnabled = false;
    g.drawImage(image, x0, y0, s, s, 0, 0, ZOOM_PX, ZOOM_PX);
    g.fillStyle = "#e0f"; g.strokeStyle = "#e0f"; g.lineWidth = 2;
    points.forEach(([x, y], i) => {
      const cx = (x - x0) * zoom, cy = (y - y0) * zoom;
      g.beginPath(); g.arc(cx, cy, 4, 0, 2 * Math.PI); g.fill();
      g.fillText(String(i + 1), cx + 6, cy - 6);
    });
    if (points.length === 4) {
      const q = orderCorners(points);
      g.beginPath(); q.forEach(([x, y], i) => (i ? g.lineTo : g.moveTo).call(g, (x - x0) * zoom, (y - y0) * zoom)); g.closePath(); g.stroke();
    }
    hint.textContent = points.length < 4
      ? `Click the ${4 - points.length} remaining outer corner(s) of the black border (any order).`
      : "Read below. Click again to start over.";
  };
  zoomCanvas.onclick = (ev) => {
    const rect = zoomCanvas.getBoundingClientRect(), { x0, y0 } = src();
    if (points.length === 4) { points = []; result.replaceChildren(); }
    points.push([x0 + (ev.clientX - rect.left) * ZOOM_PX / rect.width / zoom, y0 + (ev.clientY - rect.top) * ZOOM_PX / rect.height / zoom]);
    drawZoom();
    if (points.length === 4) showResult(DATA, p, state, result, greyImage(), orderCorners(points), changed);
  };
  const onPick = (x, y) => {
    center = [x, y]; points = []; result.replaceChildren();
    holder.replaceChildren(hint, el("div", {}, [el("span", { text: "zoom " }), zoomButtons, undo]), zoomCanvas, result);
    drawZoom();
  };
  return { onPick };
}

function overlapWarning(p, state, orig) {
  /* Text when the clicked tag already has a scan row or a manual row, else "". */
  const ctr = [orig.reduce((s, q) => s + q[0], 0) / 4, orig.reduce((s, q) => s + q[1], 0) / 4];
  const read = p.reads.find((r) => insideQuad(ctr, r.quad));
  if (read) return `This tag already has row #${read.n} (ID ${read.tag_id ?? "?"}): decide that row instead of adding a manual one.`;
  const manual = state.manual.find((m) => m.photo === p.photo && m.quad && insideQuad(ctr, m.quad));
  return manual ? `This tag was already added as ${manual.n} (ID ${manual.id_override}).` : "";
}

function showResult(DATA, p, state, holder, grey, quad, changed) {
  const read = decodeQuad(grey, quad, DATA.refs);
  const orig = quad.map(([x, y]) => [x / p.scale, y / p.scale]);
  const summary = read.candidates.slice(0, 4).map((c) => `${c.id}${c.mirrored ? "m" : ""}x${c.votes}`).join(" ");
  const use = (id, how) => {
    const n = nextManualN(state.manual, p.photo);
    const cx = orig.reduce((s, q) => s + q[0], 0) / 4, cy = orig.reduce((s, q) => s + q[1], 0) / 4;
    const side = orig.reduce((s, q, i) => s + Math.hypot(q[0] - orig[(i + 1) % 4][0], q[1] - orig[(i + 1) % 4][1]), 0) / 4;
    state.manual.push({
      photo: p.photo, colony: p.colony, kind: p.kind, taken: p.taken, date_source: p.source, n,
      tag_id: "", status: "manual", d: "", votes: "", alt: `${how}; reads of the clicked corners: ${summary || "none"}`,
      accept: "", id_override: String(id), date_override: state.dates[p.photo] || "",
      note: "corners picked by hand on review.html", x: Math.round(cx), y: Math.round(cy), side_px: Math.round(side),
      crop: "", quad: orig,
    });
    holder.replaceChildren(el("p", { text: `added ${n} = ID ${id}` }));
    changed("rows");
  };
  const card = (c) => {
    const low = c.votes < LOW_VOTES || c.d >= VOTE_D ? " - low confidence" : "";
    return el("div", { class: "card" }, [
      drawBits(el("canvas"), expectedBits(DATA.refs, c)),
      el("div", { text: `ID ${c.id}: ${c.votes}/${c.of} reads, d=${c.d}${c.mirrored ? ", mirrored" : ""}${low}` }),
      el("button", { text: `use ${c.id}`, onclick: () => use(c.id, `picked ${c.id}`) })]);
  };
  const cards = read.candidates.slice(0, 4).map(card);
  const typed = el("input", { size: 4, placeholder: "ID" });
  const typedMsg = el("span", { class: "msg" });
  const useTyped = () => {
    const v = typed.value.trim();
    if (!/^\d+$/.test(v) || +v >= DATA.refs.length) { typedMsg.textContent = `an ID is a number from 0 to ${DATA.refs.length - 1}`; return; }
    use(+v, `typed ${+v}`);
  };
  const warning = overlapWarning(p, state, orig);
  holder.replaceChildren(
    warning ? el("p", { class: "warn", text: warning }) : el("span"),
    el("div", { class: "cards" }, [el("div", { class: "card" }, [drawWarp(el("canvas"), grey, quad), el("div", { text: `as clicked: border ${Math.round(read.grid.border * 20)}/20 black, margin ${Math.round(read.grid.margin * 28)}/28 white` })]), ...cards]),
    cards.length ? el("span") : el("p", { text: `No ID within ${VOTE_D} bits. Nearest codes: ${read.closest.map((c) => `${c.id} (d=${c.d})`).join(", ")}. Re-click the corners, or type the ID if you can read it.` }),
    el("div", {}, [el("span", { text: "or type the ID you read: " }), typed, el("button", { text: "use typed ID", onclick: useTyped }), typedMsg]));
}

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", browserMain);
