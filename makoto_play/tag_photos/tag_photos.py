#!/usr/bin/env python3
r"""Read tag IDs from photos of dead ants / loose tags and record them in the colony tables.

Put photos into basler/_colonies/evidence/inbox/<folder>/ (Windows: Z:\ReiterU\Ants\basler\...),
one folder per colony and kind of finding (colony_tables.py): C21_dead, C22_dead, C21_tag, ...

  python tag_photos.py scan --colonies Z:\ReiterU\Ants\basler\_colonies
      Reads every photo and writes inbox/_review/<time>/: review.html (each tag's crop next to the
      dictionary's drawing of the ID it read), review.csv (accept = y for "ok" reads), crops/,
      annotated/, scan.json (with a SHA-256 of every photo) and a log.
  Check review.html. In review.csv set accept = y or n, type the ID you read into id_override,
  and for photos without an EXIF capture date type the date found into date_override.
  python tag_photos.py apply --review <inbox>/_review/<time>/review.csv [--yes]
      Without --yes it only prints what would change. With --yes it (1) copies each photo that
      ends a row into evidence/ and verifies the copy, (2) saves the old tag_assignments.csv in the
      review folder and writes the new one, (3) removes a photo from the inbox once every accepted
      tag on it is recorded. A failure in (1) or (2) undoes the copies and leaves the table as it
      was; running apply again after any failure is safe.

The date found is the photo's EXIF capture date unless date_override is set; apply never uses
file times, which change when photos are copied or sent through a messenger.
"""
import argparse
import codecs
import csv
import hashlib
import io
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colony_tables as T  # noqa: E402
import photo_decode as P  # noqa: E402
import review_page as RP  # noqa: E402

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF = True
except ImportError:
    HEIF = False

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"}
HEIF_EXTS = {".heic", ".heif"}
EXIF_IFD, DATETIME_ORIGINAL, DATETIME = 0x8769, 36867, 306
TIME_FMT = "%Y-%m-%d %H:%M:%S"
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d")          # Excel on Japanese Windows saves dates as 2026/10/20
COLUMNS = ["photo", "colony", "kind", "taken", "date_source", "n", "tag_id", "status", "d", "votes",
           "alt", "accept", "id_override", "date_override", "note", "x", "y", "side_px", "crop"]
ACCEPT, REJECT = {"y", "yes", "1", "true", "ok"}, {"n", "no", "0", "false"}
COLORS = {"ok": (60, 180, 60), "check": (0, 160, 255), "unreadable": (40, 40, 230)}   # BGR
CROP_CELL = 20                   # warped crop = 8 cells x 20 px
CONTEXT_PX = 240
ANNOTATED_LONG_SIDE = 1600


class TeeLog:
    """Copy everything printed to a log file as well (the terminal output is not kept)."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.fh = open(self.path, "w", encoding="utf-8")
        self.stdout, sys.stdout = sys.stdout, self
        return self

    def write(self, s):
        self.stdout.write(s)
        self.fh.write(s)

    def flush(self):
        self.stdout.flush()
        self.fh.flush()

    def __exit__(self, *exc):
        sys.stdout = self.stdout
        self.fh.close()


# ---------------------------------------------------------------- photos

def exif_date(path):
    """EXIF capture time 'YYYY-MM-DD HH:MM:SS', or None when the photo has none or it is unreadable."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            raw = exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exif.get(DATETIME)
        return datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S").strftime(TIME_FMT) if raw else None
    except Exception:  # noqa: BLE001 - any broken EXIF block means "no capture date"; callers report that
        return None


def photo_date(path):
    """('YYYY-MM-DD HH:MM:SS', 'exif' | 'mtime') for display; apply only accepts the EXIF date."""
    taken = exif_date(path)
    if taken:
        return taken, "exif"
    return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime(TIME_FMT), "mtime"


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return {"size": Path(path).stat().st_size, "sha256": h.hexdigest()}


def load_photo(path):
    """Upright BGR image (EXIF orientation applied)."""
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def imwrite(path, img, quality=90):
    """cv2.imwrite fails on some Windows paths; encode in memory instead."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix.lower() == ".jpg" else []
    ok, buf = cv2.imencode(path.suffix, np.ascontiguousarray(img), params)
    if not ok:
        raise OSError(f"cannot encode {path}")
    path.write_bytes(buf.tobytes())


def read_colonies(col):
    with open(col / "colonies.csv", newline="", encoding="utf-8-sig") as fh:
        return {int(r["colony"]) for r in csv.DictReader(fh) if (r.get("colony") or "").strip().isdigit()}


def find_photos(inbox, colonies):
    """[(folder, colony, kind, path)] and messages about everything skipped."""
    jobs, skipped = [], []
    for d in sorted(p for p in inbox.iterdir() if p.is_dir() and not p.name.startswith("_")):
        entries = sorted(d.iterdir())
        photos = [p for p in entries if p.is_file() and p.suffix.lower() in EXTS]
        parsed = T.parse_folder(d.name)
        if parsed is None:
            skipped.append(f"{d.name}/ ({len(photos)} photos): folder name is not C<colony>_<dead|tag|removed|check>")
            continue
        if parsed[0] not in colonies:
            skipped.append(f"{d.name}/ ({len(photos)} photos): colony {parsed[0]} is not in colonies.csv")
            continue
        others = [p.name for p in entries if p not in photos]
        if others:
            skipped.append(f"{d.name}/: {len(others)} entries are not photos and were ignored (e.g. {others[0]})")
        for p in photos:
            if p.suffix.lower() in HEIF_EXTS and not HEIF:
                skipped.append(f"{d.name}/{p.name}: HEIC needs 'pip install pillow-heif' (or export as JPEG)")
            else:
                jobs.append((d.name, parsed[0], parsed[1], p))
    loose = [p.name for p in inbox.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    if loose:
        skipped.append(f"{len(loose)} photos lie directly in the inbox (e.g. {loose[0]}): move them into a folder")
    return jobs, skipped


# ---------------------------------------------------------------- scan

def _reference(gray, r, dec):
    """Dictionary drawing of the read ID, turned (and mirrored) the way the tag was read."""
    m = aruco.generateImageMarker(dec.dictionary, r.tag_id, 6 * CROP_CELL, borderBits=1)
    m = cv2.copyMakeBorder(m, CROP_CELL, CROP_CELL, CROP_CELL, CROP_CELL, cv2.BORDER_CONSTANT, value=255)
    bits = P.read_grid(gray, r.quad).bits
    if r.mirrored:
        return np.fliplr(np.rot90(m, P.rotation_of(np.fliplr(bits), dec.refs, r.tag_id)))
    return np.rot90(m, P.rotation_of(bits, dec.refs, r.tag_id))


def save_crops(color, gray, r, dec, crops, base):
    """<base>_warp.png (the tag as read), _ref.png (dictionary drawing), _ctx.jpg (surroundings)."""
    imwrite(crops / f"{base}_warp.png", P.warp(gray, r.quad, CROP_CELL))
    if r.tag_id is not None:
        imwrite(crops / f"{base}_ref.png", _reference(gray, r, dec))
    half = int(2.5 * r.side) + 8
    x0, y0 = max(0, int(r.x) - half), max(0, int(r.y) - half)
    ctx = color[y0:int(r.y) + half, x0:int(r.x) + half].copy()
    cv2.polylines(ctx, [(r.quad - [x0, y0]).astype(np.int32)], True, COLORS[r.status], 2)
    f = CONTEXT_PX / max(ctx.shape[:2])
    imwrite(crops / f"{base}_ctx.jpg", cv2.resize(ctx, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))


def save_annotated(color, reads, path):
    """Photo with each read outlined in its status colour and labelled '#n ID' (n = the row's n in
    review.csv), on a white box so the label stays readable on dark ants."""
    img = color.copy()
    scale = max(img.shape[:2]) / 1600
    font, size, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, max(2, int(2 * scale))
    for n, r in enumerate(reads, 1):
        c = COLORS[r.status]
        cv2.polylines(img, [r.quad.astype(np.int32)], True, c, max(2, int(r.side / 15)))
        label = f"#{n} {'?' if r.tag_id is None else r.tag_id}"
        (w, h), base = cv2.getTextSize(label, font, size, thick)
        x = int(min(max(r.x - w / 2, 0), img.shape[1] - w))
        y = int(max(r.quad[:, 1].min() - 0.3 * r.side, h + base))
        cv2.rectangle(img, (x - 3, y - h - 3), (x + w + 3, y + base), (255, 255, 255), -1)
        cv2.rectangle(img, (x - 3, y - h - 3), (x + w + 3, y + base), c, 2)
        cv2.putText(img, label, (x, y), font, size, c, thick)
    f = min(1.0, ANNOTATED_LONG_SIDE / max(img.shape[:2]))
    imwrite(path, cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), quality=85)


def scan_photo(job, dec, out):
    """Review rows and an HTML section for one photo."""
    folder, colony, kind, path = job
    color = load_photo(path)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    taken, source = photo_date(path)
    reads = P.decode_photo(gray, dec)
    stem = f"{folder}__{path.name}"
    save_annotated(color, reads, out / "annotated" / f"{stem}.jpg")
    rows = []
    for n, r in enumerate(reads, 1):
        base = f"{stem}__{n}"
        save_crops(color, gray, r, dec, out / "crops", base)
        rows.append({"photo": f"{folder}/{path.name}", "colony": colony, "kind": kind, "taken": taken,
                     "date_source": source, "n": n, "tag_id": "" if r.tag_id is None else r.tag_id,
                     "status": r.status, "d": "" if r.d is None else r.d, "votes": r.votes, "alt": r.alt,
                     "accept": "y" if r.status == "ok" else "", "id_override": "", "date_override": "",
                     "note": "", "x": int(r.x), "y": int(r.y), "side_px": int(r.side), "crop": base})
    ids = ", ".join(f"{'?' if r.tag_id is None else r.tag_id} {r.status}" for r in reads) or "no tag found"
    print(f"{folder}/{path.name}  {taken} ({source})  {ids}")
    img, scale = RP.embed_photo(color)
    return rows, {"photo": f"{folder}/{path.name}", "colony": colony, "kind": kind, "taken": taken,
                  "source": source, "img": img, "scale": scale, "reads": RP.page_reads(reads)}


def write_review_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:      # BOM: Excel opens it as UTF-8
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _scan_all(jobs, dec, out):
    """(rows, pages, {photo: fingerprint}, failures); one unreadable photo must not lose the batch."""
    rows, pages, prints, failed = [], [], {}, []
    for job in jobs:
        rel = f"{job[0]}/{job[3].name}"
        try:
            fp = fingerprint(job[3])
            r, page = scan_photo(job, dec, out)
        except Exception as exc:  # noqa: BLE001 - reported in the log, the summary, the HTML and scan.json
            failed.append(f"{rel}: {type(exc).__name__}: {exc}")
            print(f"[ERR] {failed[-1]}")
            continue
        prints[rel] = fp
        rows += r
        pages.append(page)
    return rows, pages, prints, failed


def scan(args):
    col = Path(args.colonies)
    if not (col / "colonies.csv").is_file():
        print(f"[ERR] {col} has no colonies.csv")
        return 2
    inbox = Path(args.inbox) if args.inbox else col / "evidence" / "inbox"
    if not inbox.is_dir():
        inbox.mkdir(parents=True)
        print(f"created {inbox}: put photos into C<colony>_<dead|tag|removed> folders there and rerun")
        return 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = inbox / "_review" / stamp
    (out / "crops").mkdir(parents=True)
    (out / "annotated").mkdir()
    with TeeLog(out / f"tag_photos_scan_log_{stamp}.txt"):
        print(f"scan {inbox}\n  -> {out}")
        jobs, skipped = find_photos(inbox, read_colonies(col))
        for s in skipped:
            print(f"[SKIP] {s}")
        dec = P.load_decoder(args.npz)
        rows, pages, prints, failed = _scan_all(jobs, dec, out)
        write_review_csv(out / "review.csv", rows)
        RP.write_html(out / "review.html", stamp, COLUMNS, dec.refs, pages, rows, failed)
        meta = {"inbox": str(inbox.absolute()), "colonies": str(col.absolute()), "npz": str(args.npz),
                "n_ids": int(dec.refs.shape[0]), "photos": prints, "failed": failed}
        (out / "scan.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        n = {s: sum(r["status"] == s for r in rows) for s in COLORS}
        print(f"{len(pages)} photos, {len(rows)} tags: ok {n['ok']}, check {n['check']}, "
              f"unreadable {n['unreadable']}" + (f"; {len(failed)} photos FAILED" if failed else ""))
        print(f"review: {out / 'review.html'}")
        print(f'then:   python {Path(__file__).name} apply --review "{out / "review.csv"}"')
    return 0


# ---------------------------------------------------------------- apply

class RowSkip(Exception):
    """A reviewed row that cannot be applied; the message says why."""


@dataclass(frozen=True)
class Selection:
    findings: list
    notes: list             # rows that cannot be applied
    lines: list             # what each applied row means
    name_of: dict           # inbox path -> evidence/ file name
    prints: dict            # inbox path -> fingerprint
    keys_of: dict           # inbox path -> {(colony, tag_id)} accepted on it
    blocked: set            # inbox paths with a row that could not be applied


def read_review(path):
    raw = Path(path).read_bytes()
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):   # Excel "Unicode text"
        text = raw.decode("utf-16")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp932")                                  # Excel CSV on Japanese Windows
    first = text.split("\n", 1)[0]
    reader = csv.DictReader(io.StringIO(text, newline=""),
                            delimiter="\t" if "\t" in first and "," not in first else ",")
    missing = {"photo", "n", "tag_id", "accept", "id_override"} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return list(reader)


def chosen_id(row):
    """The ID a reviewed row asks to apply, or None when it was rejected. A row without a decision
    raises RowSkip, so its photo stays in the inbox until someone decides."""
    accept = (row.get("accept") or "").strip().lower()
    override = (row.get("id_override") or "").strip()
    if accept in REJECT:
        return None
    if override:
        return override
    if accept in ACCEPT:
        return (row.get("tag_id") or "").strip()
    if accept:
        raise RowSkip(f"accept '{row.get('accept')}' is neither y nor n")
    raise RowSkip("no decision yet: set accept to y or n (or type the ID into id_override)")


def inbox_path(row, inbox):
    """<inbox>/<C<colony>_<kind>>/<file> named by a review row, or None if it points elsewhere."""
    rel = PurePosixPath((row.get("photo") or "").replace("\\", "/"))
    if rel.is_absolute() or len(rel.parts) != 2 or ".." in rel.parts or T.parse_folder(rel.parts[0]) is None:
        return None
    return Path(inbox) / rel.parts[0] / rel.parts[1]


def found_date(row, src):
    """(YYYY-MM-DD, source): date_override, else the EXIF capture date."""
    text = (row.get("date_override") or "").strip()
    if text:
        for fmt in DATE_FORMATS:
            try:
                day = datetime.strptime(text, fmt).date()
            except ValueError:
                continue
            if day > datetime.now().date():
                raise RowSkip(f"date_override {day} is in the future")
            return day.isoformat(), "date_override"
        raise RowSkip(f"date_override '{text}' is not YYYY-MM-DD")
    taken = exif_date(src)
    if taken is None:
        raise RowSkip("the photo has no EXIF capture date; type the date found (YYYY-MM-DD) into date_override")
    return taken[:10], "EXIF"


def row_target(row, meta, prints):
    """(inbox path, colony, kind, tag id, date found, date source) for an accepted row, else None."""
    tag = chosen_id(row)
    if tag is None:
        return None
    src = inbox_path(row, meta["inbox"])
    if src is None:
        raise RowSkip("photo must be <C<colony>_<kind> folder>/<file> inside the inbox")
    colony, kind = T.parse_folder(src.parent.name)
    if not tag.isdecimal() or int(tag) >= meta.get("n_ids", 100):
        raise RowSkip(f"'{tag}' is not a tag ID")
    if not src.is_file():
        raise RowSkip("photo is no longer in the inbox (applied already?)")
    if src not in prints:
        prints[src] = fingerprint(src)
    if prints[src] != meta.get("photos", {}).get(f"{src.parent.name}/{src.name}"):
        raise RowSkip("photo changed since the scan; run scan again")
    found, source = found_date(row, src)
    return src, colony, kind, int(tag), found, source


def evidence_names(first, evidence, prints):
    """{inbox path: evidence/ file name}: the default name, or one with the date found (and a
    counter) when another photo already has it. An identical file there is this photo, filed by an
    earlier run, so its name is kept."""
    names, used = {}, set()
    for src, (colony, kind, found) in sorted(first.items()):
        base = Path(T.evidence_name(colony, kind, src.name))
        day = found.replace("-", "")
        options = [base.name] + [f"{base.stem}_{day}{'' if i == 1 else f'_{i}'}{base.suffix}" for i in range(1, 100)]
        for name in options:
            dst = evidence / name
            if name.lower() not in used and (not dst.exists() or fingerprint(dst) == prints[src]):
                names[src] = name
                used.add(name.lower())
                break
        else:
            raise ValueError(f"no free evidence name for {src.name}")
    return names


def collect_findings(rows, meta):
    notes, blocked, prints, targets, checks = [], set(), {}, [], []
    for r in rows:
        src = inbox_path(r, meta["inbox"])
        if src is not None and T.parse_folder(src.parent.name)[1] == T.CHECK:
            tag = (r.get("id_override") or "").strip() or (r.get("tag_id") or "").strip() or "?"
            checks.append(f"[CHECK] {r.get('photo')} #{r.get('n')}: tag {tag} ({r.get('status')}) - "
                          f"live-ant check photo, never written{'; note: ' + r['note'] if r.get('note') else ''}")
            continue
        try:
            t = row_target(r, meta, prints)
        except RowSkip as exc:
            notes.append(f"{r.get('photo')} #{r.get('n')}: {exc}")
            src = inbox_path(r, meta["inbox"])
            if src is not None:
                blocked.add(src)
            continue
        if t:
            targets.append(t)
    first = {}
    for src, colony, kind, _, found, _ in targets:
        first[src] = (colony, kind, min(found, first[src][2]) if src in first else found)
    names = evidence_names(first, Path(meta["colonies"]) / "evidence", prints)
    findings, lines, keys_of = [], [], {}
    for src, colony, kind, tag_id, found, source in targets:
        findings.append(T.Finding(colony, tag_id, kind, found, (names[src],)))
        keys_of.setdefault(src, set()).add((colony, tag_id))
        lines.append(f"[READ] {src.parent.name}/{src.name}: colony {colony} tag {tag_id}, found {found} "
                     f"({source}) -> evidence/{names[src]}")
    return Selection(findings, notes, lines + checks, names, prints, keys_of, blocked)


def rollback(paths):
    for p in paths:
        try:
            p.unlink(missing_ok=True)
            print(f"[UNDO] removed evidence/{p.name}")
        except OSError as exc:
            print(f"[ERR] could not remove {p} ({exc}); delete it by hand")


def file_evidence(updates, sel, evidence, created):
    """Copy each photo named by an update into evidence/ and verify it; every file this creates,
    a partial one included, is appended to created before it is written. Raises on failure."""
    src_of = {name: src for src, name in sel.name_of.items()}
    for name in sorted({n for u in updates for n in u.photos}):
        src, dst = src_of[name], evidence / name
        if dst.exists():
            continue                     # identical copy from an earlier run (checked when it was named)
        created.append(dst)
        shutil.copy2(src, dst)
        if fingerprint(dst) != sel.prints[src]:
            raise OSError(f"the copy of {src.name} in evidence/ differs from the photo")
        print(f"[COPY] {src.parent.name}/{src.name} -> evidence/{name}")


def write_changes(table_path, table, updates, out, stamp):
    """Back up and rewrite tag_assignments.csv; False (nothing changed) when that is impossible."""
    if table_path.read_bytes() != table.raw:
        print("[ERR] tag_assignments.csv changed while this ran; nothing written, run apply again")
        return False
    backup = out / "tag_assignments.before.csv"
    if backup.exists():
        backup = out / f"tag_assignments.before_{stamp}.csv"
    backup.write_bytes(table.raw)
    try:
        T.write_table(table_path, table, T.apply_updates(table.rows, updates))
    except OSError as exc:
        print(f"[ERR] could not write {table_path} ({exc}); is it open in Excel? Nothing written")
        return False
    print(f"wrote {table_path}: {sum(not u.already for u in updates)} rows ended "
          f"(previous version: {backup.name})")
    return True


def finish_inbox(updates, sel, meta):
    """Remove an inbox photo once its copy is in evidence/ and every accepted tag on it is recorded."""
    done = {(u.colony, u.tag_id) for u in updates}
    for src, keys in sel.keys_of.items():
        rel = f"{src.parent.name}/{src.name}"
        if src in sel.blocked or not keys <= done:
            print(f"[KEEP] {rel}: not every tag on it could be applied (see SKIP / PROBLEM); it stays in the inbox")
            continue
        try:
            src.unlink()
            print(f"[DONE] {rel} is now evidence/{sel.name_of[src]}")
        except OSError as exc:
            print(f"[WARN] {rel} is filed in evidence/ but could not be removed from the inbox ({exc})")
    rest = [p for p in meta.get("photos", {}) if (Path(meta["inbox"]) / p).is_file()
            and Path(meta["inbox"]) / p not in sel.keys_of]
    for p in rest:
        if T.parse_folder(PurePosixPath(p).parent.name)[1] == T.CHECK:
            print(f"[KEEP] {p}: live-ant check photo; it stays in the inbox (delete it when you are done)")
        else:
            print(f"[KEEP] {p}: no tag on it was accepted; it stays in the inbox")


def report(updates, problems):
    for u in updates:
        if u.already:
            print(f"[RECORDED] colony {u.colony} tag {u.tag_id}: already ended {u.ended_on} ({u.end_reason}) "
                  f"from {', '.join(u.photos)}")
        else:
            print(f"[END] colony {u.colony} tag {u.tag_id}: ended_on {u.ended_on}, end_reason {u.end_reason}, "
                  f"evidence += {u.evidence_add}")
    for p in problems:
        print(f"[PROBLEM] {p}")


def run_apply(rev, meta, yes, stamp):
    print(f"apply {rev}" + ("" if yes else "  (dry run)"))
    table_path = Path(meta["colonies"]) / "tag_assignments.csv"
    try:
        sel = collect_findings(read_review(rev), meta)
        table = T.read_table(table_path)
    except (OSError, ValueError) as exc:
        print(f"[ERR] {exc}")
        return 2
    for line in sel.lines + [f"[SKIP] {n}" for n in sel.notes]:
        print(line)
    updates, problems = T.plan_updates(table.rows, sel.findings)
    report(updates, problems)
    status = 1 if problems or sel.notes else 0
    if not yes:
        print("dry run: nothing written; rerun with --yes to write")
        return status
    if not updates:
        print("nothing to write")
        return status
    created, written = [], False
    try:
        file_evidence(updates, sel, Path(meta["colonies"]) / "evidence", created)
        if any(not u.already for u in updates):
            written = write_changes(table_path, table, updates, rev.parent, stamp)
            if not written:
                raise RuntimeError("tag_assignments.csv was not written")
    except Exception as exc:  # noqa: BLE001 - whatever fails before the table is written undoes the copies
        if written:
            raise                        # the table cites the copies now: keep them
        print(f"[ERR] {type(exc).__name__}: {exc}; undoing the copies, the table is unchanged")
        rollback(created)
        return 2
    finish_inbox(updates, sel, meta)
    return status


def apply(args):
    rev = Path(args.review)
    meta_path = rev.parent / "scan.json"
    if not rev.is_file() or not meta_path.is_file():
        print(f"[ERR] need {rev} and scan.json beside it (both are written by scan)")
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with TeeLog(rev.parent / f"tag_photos_apply_log_{stamp}.txt"):
        return run_apply(rev, meta, args.yes, stamp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="decode every photo in the inbox and write a review folder")
    s.add_argument("--colonies", required=True, help="basler/_colonies folder")
    s.add_argument("--inbox", default=None, help="default: <colonies>/evidence/inbox")
    s.add_argument("--npz", default=str(P.DEFAULT_NPZ), help="ArUco dictionary (default A100)")
    a = sub.add_parser("apply", help="write the reviewed IDs into tag_assignments.csv")
    a.add_argument("--review", required=True, help="review.csv written by scan")
    a.add_argument("--yes", action="store_true", help="write (default: dry run)")
    args = ap.parse_args(argv)
    return scan(args) if args.cmd == "scan" else apply(args)


if __name__ == "__main__":
    sys.exit(main())
