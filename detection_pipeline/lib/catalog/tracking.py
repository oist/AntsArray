"""Tracking provenance for the catalog: which homography made a block's tracks.

Reads ``<block>/tracks/TRACKING_STATE.json`` (see lib/tracking_state.py). The
mapper writes it while running; ``catalog.py track-init`` backfills it for
blocks tracked before the record existed. The catalog only reads.
"""
import os

# lib/ is on sys.path when catalog.py is the entry point; add it ourselves when
# this package is imported some other way (a REPL, another tool).
try:
    import tracking_state
except ImportError:  # pragma: no cover - import-path fallback
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import tracking_state

TRACKS_DIRNAME = "tracks"


def read_tracking(blockdir):
    """Summary cells for one block, plus ``tracking_error`` (text, or '').

    Absent record -> empty cells, no error: most blocks have not been tracked.
    Malformed record -> empty cells and the reason in ``tracking_error``; the
    catalog reports, it does not refuse.
    """
    out = tracking_state.summary(None)
    out["tracking_error"] = ""
    tracks_dir = os.path.join(blockdir, TRACKS_DIRNAME)
    try:
        state = tracking_state.load(tracks_dir)
    except (ValueError, OSError) as e:
        out["tracking_error"] = str(e)
        return out
    out.update(tracking_state.summary(state))
    return out
