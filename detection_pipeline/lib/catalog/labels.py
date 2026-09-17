"""Optional user-maintained per-block label overlay.

Reads a small CSV (block_id,labels) and merges its labels into the matching
catalog rows at emit time. Applied AFTER the scan cache, so edits take effect on
every run -- including `build` mode and cache hits -- with no rescan needed.

These labels are display/filter only: they do NOT influence is_stim / is_test
inference (which stays derived from the folder name). Intended for block-level
experimental annotations (e.g. a manipulation indicator) without renaming the
block** directories.

Format (block_labels.csv, default location <outdir>/):
    # comment lines and blank lines are ignored; an optional header 'block_id,labels'
    20260717/block01,manipulation
    20260717/block02,manipulation|drug        # '|' or ',' separate multiple labels
The block_id is the catalog's own '<date>/block**' relative path; it may use
'/' or '\\' and is matched case-insensitively.
"""
import csv
import os

from . import const

LABELS_FILENAME = "block_labels.csv"


def _norm(block_id):
    """Normalize a block_id for matching: unify slashes, trim, casefold."""
    return block_id.strip().replace("\\", "/").strip("/").lower()


def load_block_labels(path, log=None):
    """Return {normalized_block_id: [labels]}. Empty if the file is absent/unreadable."""
    mapping = {}
    if not path or not os.path.isfile(path):
        return mapping
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                bid = row[0].strip()
                if not bid or bid.startswith("#") or bid.lower() == "block_id":
                    continue
                # Labels may sit in one cell ('a|b') or spill across cells if the
                # user separated them with commas -- union everything after col 0.
                labels = []
                for cell in row[1:]:
                    for tok in cell.replace("|", ",").split(","):
                        tok = tok.strip()
                        if tok and tok not in labels:
                            labels.append(tok)
                if not labels:
                    continue
                # Same block_id may recur across lines -> merge, don't overwrite.
                dst = mapping.setdefault(_norm(bid), [])
                for lab in labels:
                    if lab not in dst:
                        dst.append(lab)
    except OSError as e:
        if log:
            log("[WARN] block_labels: could not read %s: %s" % (path, e))
    return mapping


def apply_to_rows(catalog_rows, mapping):
    """Merge mapped labels into each row's 'labels' cell. Returns rows touched."""
    if not mapping:
        return 0
    touched = 0
    for r in catalog_rows:
        extra = mapping.get(_norm(r.get("block_id", "")))
        if not extra:
            continue
        merged = [t for t in r.get("labels", "").split(const.TOKEN_JOIN) if t]
        added = False
        for lab in extra:
            if lab not in merged:
                merged.append(lab)
                added = True
        if added:
            r["labels"] = const.TOKEN_JOIN.join(merged)
            touched += 1
    return touched
