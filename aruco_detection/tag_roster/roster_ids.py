#!/usr/bin/env python3
"""Which ArUco tag IDs each colony uses now, which it dropped, and cuttable sheets to re-tag them.

Reads map_combine ArUco panorama pickles written with --min_instance_frame_frac 0 (no ID is
pre-filtered): <pano>/<exp>_chunkNNN_aruco_panorama_x_{left,right}<X>.pkl. An ID's detection
rate is the fraction of a chunk's frames it is seen in. Per side (= colony), from the rates of
the last --recent chunks:

  present    recent rate >= 0.10
  dropped    present in some chunk of the window but recent rate < 0.02,
             or "present" in --previous (an earlier run's roster_status.csv) and absent now
  absent     recent rate < 0.02 and never present -> unused, or dropped before the window
  uncertain  anything else -> look at the ants before printing these

Misreads of real tags ("ghosts") reach ~0.014 on laminate tags (20260810/block02), hence 0.02.
Dropped + absent IDs are printed per side as retag_<side>.png/.svg (generate_cuttable_sheet).

  python roster_ids.py --pano DIR --out DIR --npz DICT.npz [--block DIR] [--chunks 0-11]
                       [--recent 4] [--copies 2] [--previous roster_status.csv]
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "custom_dicts"))
import generate_cuttable_sheet as sheet  # noqa: E402

PAT = re.compile(r"_chunk(\d{3})_aruco_panorama_x_(left|right)(\d+)\.pkl$")
SIDES = ("left", "right")
PRESENT, ABSENT = 0.10, 0.02
# Two tags cannot sit within ~1.4 mm (panorama ~110 px/cm): an ID mostly coinciding with ANOTHER
# ID in the same frame is a misread of that tag. Reported as evidence, not used to classify.
COLOC_PX = 15.0
FRAME_SCALE = 1e6        # keeps KD-tree pairs within one frame
JST = timezone(timedelta(hours=9))


def classify(rates, recent, present=PRESENT, absent=ABSENT):
    """(status, recent rate, peak chunk rate) from per-chunk detection rates in time order."""
    recent_rate = float(np.mean(rates[-recent:])) if rates else 0.0
    peak = max(rates) if rates else 0.0
    if recent_rate >= present:
        return "present", recent_rate, peak
    if recent_rate < absent:
        return ("dropped" if peak >= present else "absent"), recent_rate, peak
    return "uncertain", recent_rate, peak


def coincidences(df):
    """Per ID: frames coinciding with another ID within COLOC_PX, and {id: {partner: n}}."""
    pos = df.groupby(["Frame", "Instance"], sort=False)[["X", "Y"]].mean().reset_index()
    pts = np.column_stack([pos["X"].to_numpy(float), pos["Y"].to_numpy(float),
                           pos["Frame"].to_numpy(float) * FRAME_SCALE])
    pairs = cKDTree(pts).query_pairs(COLOC_PX, output_type="ndarray")
    ids = pos["Instance"].to_numpy(int)
    partner = defaultdict(lambda: defaultdict(int))
    if not len(pairs):
        return {}, partner
    n_hit = pd.Series(ids[np.unique(pairs.ravel())]).value_counts().to_dict()
    for a, b in zip(ids[pairs[:, 0]], ids[pairs[:, 1]]):
        partner[int(a)][int(b)] += 1
        partner[int(b)][int(a)] += 1
    return n_hit, partner


def check_pickles(files):
    """Exactly one pickle per chunk and side. Two (a rerun with another split or homography) would
    be mixed; a missing side would read as every ID of that colony unseen, and all get printed."""
    by = defaultdict(list)
    for f in files:
        chunk, side, _ = PAT.search(f).groups()
        by[(chunk, side)].append(os.path.basename(f))
    for (chunk, side), names in sorted(by.items()):
        if len(names) > 1:
            sys.exit(f"chunk {chunk} {side}: {len(names)} pickles {names}; clear the pano dir and rerun")
    for chunk in sorted({c for c, _ in by}):
        missing = [s for s in SIDES if (chunk, s) not in by]
        if missing:
            sys.exit(f"chunk {chunk} has no {'/'.join(missing)} pickle; rerun map_combine for it")


def read_panoramas(files):
    """Stream the pickles into per-side, per-ID, per-chunk counts."""
    data = dict(chunk_frames={}, split=set(), near_split=0,
                per={s: defaultdict(dict) for s in SIDES},
                coloc={s: defaultdict(int) for s in SIDES},
                partners={s: defaultdict(lambda: defaultdict(int)) for s in SIDES})
    for f in files:
        chunk, side, thr = PAT.search(f).groups()
        payload = pd.read_pickle(f)
        df, nf = payload["detections"], int(payload["num_frames"])
        if nf <= 0:
            sys.exit(f"{os.path.basename(f)} has num_frames={nf}")
        data["chunk_frames"][chunk] = max(nf, data["chunk_frames"].get(chunk, 0))
        data["split"].add(int(thr))
        data["near_split"] += int((np.abs(df["X"].to_numpy(float) - int(thr)) < 25).sum())
        n_hit, partner = coincidences(df)
        for iid, n in n_hit.items():
            data["coloc"][side][int(iid)] += int(n)
        for a, d in partner.items():
            for b, n in d.items():
                data["partners"][side][a][b] += n
        for iid, g in df.groupby("Instance"):
            fr = np.unique(g["Frame"].to_numpy())
            data["per"][side][int(iid)][chunk] = dict(
                frames=len(fr), last=int(fr[-1]), x=float(g["X"].median()),
                y=float(g["Y"].median()), cams=set((g["Cam"] + 1).astype(int).tolist()))
        print(f"read {os.path.basename(f)}: {len(df)} rows, {df['Instance'].nunique()} IDs")
    return data


def block_timing(block, chunk_frames):
    """(recording start or None, fps, frames per chunk) from the block's diag.json and contract."""
    t0, fps, chunk_len = None, None, None
    for diag in sorted(glob.glob(os.path.join(block or "", "cam*.diag.json")))[:1]:
        try:
            d = json.load(open(diag))
            t0 = datetime.fromtimestamp(d["provenance"]["firstEncodedFrame"]["hostEpochMs"] / 1000, JST)
            fps = float(d["context"]["fps"])
        except (KeyError, TypeError, ValueError, OSError):
            pass
    state = os.path.join(block or "", "data", "PIPELINE_STATE.json")
    if block and os.path.isfile(state):
        ch = json.load(open(state))["chunking"]
        fps = fps or float(next(iter(ch["videos"].values()))["fps"])
        chunk_len = int(round(ch["chunk_sec"] * fps))
    if fps is None:
        fps = 24.0
        print("[WARN] no diag.json or PIPELINE_STATE.json frame rate; assuming 24 fps for times")
    if chunk_len is None:   # every chunk but the last is nominal length
        chunk_len = pd.Series(list(chunk_frames.values())).mode().max()
    return t0, fps, int(chunk_len)


def frame_time(frame, t0, fps):
    secs = frame / fps
    if t0 is not None:
        return (t0 + timedelta(seconds=secs)).strftime("%m-%d %H:%M:%S")
    return "+%02d:%02d:%02d" % (secs // 3600, secs % 3600 // 60, secs % 60)


def side_table(data, side, n_markers, recent, timing):
    t0, fps, chunk_len = timing
    cf = data["chunk_frames"]
    chunks = sorted(cf)
    total = sum(cf.values())
    rows = []
    for iid in range(n_markers):
        by = data["per"][side].get(iid, {})
        rates = [by[c]["frames"] / cf[c] if c in by else 0.0 for c in chunks]
        status, recent_rate, peak = classify(rates, recent)
        last_seen = ""
        if status == "dropped":
            c = max(c for c, r in zip(chunks, rates) if r >= PRESENT)
            last_seen = frame_time(int(c) * chunk_len + by[c]["last"], t0, fps)
        frames = sum(s["frames"] for s in by.values())
        partners = data["partners"][side].get(iid, {})
        best = max(partners.items(), key=lambda kv: kv[1]) if partners else (-1, 0)
        row = dict(side=side, id=iid, status=status, recent_rate=round(recent_rate, 5),
                   rate=round(frames / total, 5), peak_chunk_rate=round(peak, 5), last_seen=last_seen,
                   coloc_frac=round(data["coloc"][side].get(iid, 0) / frames, 4) if frames else 0.0,
                   coloc_with=best[0], n_cams=len(set().union(*(s["cams"] for s in by.values()))),
                   x_median=round(float(np.median([s["x"] for s in by.values()])), 1) if by else np.nan,
                   y_median=round(float(np.median([s["y"] for s in by.values()])), 1) if by else np.nan)
        row.update({f"r{c}": round(r, 5) for c, r in zip(chunks, rates)})
        rows.append(row)
    return pd.DataFrame(rows)


def apply_previous(table, previous_csv):
    """IDs 'present' in an earlier run but absent now were dropped before this window."""
    prev = pd.read_csv(previous_csv)
    was = set(zip(prev.loc[prev["status"] == "present", "side"], prev.loc[prev["status"] == "present", "id"]))
    hit = table.apply(lambda r: r["status"] == "absent" and (r["side"], r["id"]) in was, axis=1)
    table.loc[hit, "status"] = "dropped"
    table.loc[hit, "last_seen"] = "before this window"
    return table


def ids_of(table, side, *statuses):
    return sorted(table.loc[(table["side"] == side) & table["status"].isin(statuses), "id"].tolist())


def write_sheets(table, npz, out, copies, label):
    dictionary, n_markers, min_d, _ = sheet.load_custom_dictionary(npz)
    made = {}
    for side in SIDES:
        ids = ids_of(table, side, "dropped", "absent")
        if not ids:
            continue
        seq = sheet.tag_sequence(ids, n_markers, copies=copies)
        title = f"{label} {side.upper()} retag {len(ids)} IDs x{copies}"
        cols = sheet.default_cols(copies)
        png = os.path.join(out, f"retag_{side}.png")
        sheet.generate_cuttable_sheet(dictionary, n_markers, min_d, png, ids=seq, cols=cols, title=title)
        sheet.generate_cuttable_svg(dictionary, n_markers, min_d, Path(png).with_suffix(".svg"),
                                    ids=seq, cols=cols, title=title)
        made[side] = png
    return made


def needed_text(table, data, label, recent, timing, copies, made):
    t0, fps, chunk_len = timing
    cf = data["chunk_frames"]
    chunks = sorted(cf)
    total = sum(cf.values())
    # from the chunk grid, not the frame sum: a camera off the GOP grid can make one chunk longer
    start, end = int(chunks[0]) * chunk_len, int(chunks[-1]) * chunk_len + min(cf[chunks[-1]], chunk_len)
    span = f"{frame_time(start, t0, fps)} - {frame_time(end, t0, fps)}"
    lines = [f"Tag roster {label}   chunks {chunks[0]}-{chunks[-1]} ({total / fps / 3600:.2f} h)   {span}",
             f"status from the last {min(recent, len(chunks))} chunks; present >= {PRESENT}, absent < {ABSENT} detection rate",
             f"left/right split X = {sorted(data['split'])}; detections within 25 px of it: {data['near_split']}",
             f"CAUTION: unseen is not proof of a lost tag. An ant hidden (e.g. in the nest) for the last "
             f"{min(recent, len(chunks)) * chunk_len / fps / 3600:.1f} h reads as dropped, for the whole "
             f"window as absent;",
             "re-tagging it would give two ants one ID. A longer window lowers the risk: confirm before tagging.", ""]
    for side in SIDES:
        t = table[table["side"] == side]
        counts = t["status"].value_counts()
        lines.append(f"{side.upper():6s} " + "  ".join(f"{s} {counts.get(s, 0)}" for s in ("present", "dropped", "uncertain", "absent")))
        drop = t[t["status"] == "dropped"]
        joined = lambda *s: " ".join(map(str, ids_of(table, side, *s))) or "-"
        lines.append("  dropped    " + (", ".join(f"{r.id} (last seen {r.last_seen})" for r in drop.itertuples()) or "-"))
        lines.append("  uncertain  " + joined("uncertain") + "   <- check these ants before printing")
        lines.append("  absent     " + joined("absent"))
        target = f"-> {os.path.basename(made[side])} / .svg ({copies} per ID)" if side in made else "(nothing to print)"
        lines.append("  print      " + joined("dropped", "absent") + f"   {target}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pano", required=True, help="map_combine ArUco panorama pickles")
    p.add_argument("--out", required=True)
    p.add_argument("--npz", required=True, help="ArUco dictionary the block was detected with")
    p.add_argument("--block", default=None, help="block dir: recording start time and chunk length")
    p.add_argument("--chunks", default=None, help='only these chunk indices, e.g. "0-11"')
    p.add_argument("--recent", type=int, default=4, help="chunks that decide the current status (default 4)")
    p.add_argument("--copies", type=int, default=2, help="tags per ID on the sheets (default 2)")
    p.add_argument("--previous", default=None, help="roster_status.csv of an earlier run")
    p.add_argument("--label", default=None, help="sheet/summary label (default <date>/<block>)")
    a = p.parse_args(argv)

    files = sorted(f for f in glob.glob(os.path.join(a.pano, "*.pkl")) if PAT.search(f))
    if a.chunks:
        keep = {"%03d" % c for c in sheet.parse_ids(a.chunks, 1000)}
        files = [f for f in files if PAT.search(f).group(1) in keep]
    if not files:
        sys.exit(f"no ArUco panorama pickles in {a.pano}")
    check_pickles(files)
    os.makedirs(a.out, exist_ok=True)
    label = a.label or ("/".join(os.path.normpath(a.block).split(os.sep)[-2:]) if a.block else "roster")

    data = read_panoramas(files)
    timing = block_timing(a.block, data["chunk_frames"])
    n_markers = sheet.load_custom_dictionary(a.npz)[1]
    table = pd.concat([side_table(data, s, n_markers, a.recent, timing) for s in SIDES], ignore_index=True)
    if a.previous:
        table = apply_previous(table, a.previous)
    table.to_csv(os.path.join(a.out, "roster_status.csv"), index=False)

    made = write_sheets(table, a.npz, a.out, a.copies, label)
    text = needed_text(table, data, label, a.recent, timing, a.copies, made)
    with open(os.path.join(a.out, "needed_ids.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
