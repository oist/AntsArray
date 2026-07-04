"""Top-level folder taxonomy + session-name date parsing.

classify_top_entry() decides how a top-level (or nested-container child) entry
is handled: scanned as a session, emitted as a thin aux row, recursed into, or
ignored. parse_session_date() decodes the many date-naming styles.
"""
from . import const

_MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "sept": "09", "oct": "10",
    "nov": "11", "dec": "12",
}


class Classification(object):
    # action: "scan" | "thin" | "recurse" | "ignore"
    # session_kind: "session" | "aux" | "unknown"
    def __init__(self, action, session_kind, is_test=False, reason=""):
        self.action = action
        self.session_kind = session_kind
        self.is_test = is_test
        self.reason = reason


def _fmt8(s: str) -> str:
    """YYYYMMDD -> YYYY-MM-DD, or '' if implausible."""
    if len(s) != 8 or not s.isdigit():
        return ""
    y, m, d = s[:4], s[4:6], s[6:8]
    if not ("2000" <= y <= "2099" and "01" <= m <= "12" and "01" <= d <= "31"):
        return ""
    return f"{y}-{m}-{d}"


def _labels(rest: str) -> list:
    if not rest:
        return []
    return [t for t in rest.split("_") if t]


def parse_session_date(name: str) -> dict:
    """Decode a folder name into date_start / date_end / date_kind / ordinal / labels.

    Examples:
      20260420                       -> single 2026-04-20
      20250321_2_test                -> single 2025-03-21, ord 2, labels [test]
      20251008_1_30min_vibration     -> single, ord 1, labels [30min, vibration]
      20260414_20260417_CustomAruco  -> range 2026-04-14..2026-04-17, labels [CustomAruco]
      20251118-121513                -> timestamp 2025-11-18
      2025_Sep_no_pertubation        -> fuzzy 2025-09, labels [no_pertubation]
    """
    out = {"date_start": "", "date_end": "", "date_kind": "none",
           "ordinal": "", "labels": []}

    m = const.DATE_RANGE_RE.match(name)
    if m and _fmt8(m.group(1)) and _fmt8(m.group(2)):
        out.update(date_start=_fmt8(m.group(1)), date_end=_fmt8(m.group(2)),
                   date_kind="range", labels=_labels(m.group(3) or ""))
        return out

    m = const.BARE_TS_RE.match(name)
    if m and _fmt8(m.group(1)):
        out.update(date_start=_fmt8(m.group(1)), date_kind="timestamp")
        return out

    m = const.FUZZY_DATE_RE.match(name)
    if m and m.group(2).lower() in _MONTHS:
        out.update(date_start=f"{m.group(1)}-{_MONTHS[m.group(2).lower()]}",
                   date_kind="fuzzy", labels=_labels(m.group(3) or ""))
        return out

    m = const.DATE_ORD_RE.match(name)
    if m and _fmt8(m.group(1)):
        out.update(date_start=_fmt8(m.group(1)), date_kind="single",
                   ordinal=(m.group(2) or ""), labels=_labels(m.group(3) or ""))
        return out

    return out


def name_hints_stim(labels: list, name: str) -> bool:
    hay = (name + " " + " ".join(labels)).lower()
    return any(tok in hay for tok in const.STIM_NAME_TOKENS)


def classify_top_entry(name: str, is_dir: bool) -> Classification:
    if not is_dir:
        return Classification("ignore", "", reason="loose file")
    if name.startswith((".", "_")):
        return Classification("ignore", "", reason="dot/underscore dir")
    if name in const.WALK_BLACKLIST:
        return Classification("ignore", "", reason="walk blacklist")
    if name in const.KNOWN_AUX:
        return Classification("thin", "aux", reason="known aux folder")
    if name in const.TEST_AUX:
        return Classification("scan", "aux", is_test=True, reason="test/dev folder")
    if name in const.NESTED_SESSION_CONTAINERS:
        return Classification("recurse", "session", reason="nested session container")

    d = parse_session_date(name)
    if d["date_kind"] != "none":
        return Classification("scan", "session",
                              is_test=("test" in d["labels"]), reason="date-named")
    # Unrecognized: still scan; discover.py applies the content-evidence override.
    return Classification("scan", "unknown", reason="unrecognized; content override")
