"""Incremental block-level cache (JSONL).

Each block is fingerprinted from cheap signals (video count, data/ file count,
sess-file mtime+size). On an unchanged fingerprint the cached rows are reused,
so a refresh re-reads sidecars / re-parses sess files only for changed blocks.
"""
import json
import os

from . import const


class Cache:
    def __init__(self, path):
        self.path = path
        self.entries = {}   # key -> record dict

    def load(self):
        if not self.path or not os.path.isfile(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("scan_version") != const.SCAN_VERSION:
                        continue
                    self.entries[rec["key"]] = rec
        except (OSError, ValueError):
            self.entries = {}

    def get(self, key, fingerprint):
        rec = self.entries.get(key)
        if rec and rec.get("fingerprint") == fingerprint:
            return rec
        return None

    def put(self, key, fingerprint, catalog_row, video_rows, trial_rows):
        self.entries[key] = {
            "key": key,
            "scan_version": const.SCAN_VERSION,
            "fingerprint": fingerprint,
            "catalog_row": catalog_row,
            "video_rows": video_rows,
            "trial_rows": trial_rows,
        }

    def save(self):
        if not self.path:
            return
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in self.entries.values():
                f.write(json.dumps(rec) + "\n")
        os.replace(tmp, self.path)


def fingerprint(unit, fp):
    """Cheap change-detector for one block/flat-session/aux unit."""
    parts = [
        const.SCAN_VERSION,
        len(unit.video_names), unit.n_global, unit.dead_symlinks,
        fp.file_count, len(unit.subdir_names),
        unit.session_kind, unit.layout,
        int(fp.has_hpc_logs),
    ]
    for p in sorted(unit.sess_paths):
        try:
            st = os.stat(p)
            parts.append(f"{os.path.basename(p)}:{int(st.st_mtime)}:{st.st_size}")
        except OSError:
            parts.append(f"{os.path.basename(p)}:na")
    return "|".join(str(x) for x in parts)
