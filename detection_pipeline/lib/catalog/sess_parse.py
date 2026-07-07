"""Tolerant parser for session/stim files.

Handles two formats and never raises (returns a partial SessDoc with warnings):
  sess_v1            tab-separated: "<ts>\\t<TYPE>\\t<payload>", '# session' header,
                     'SESS ==== START sess_.. (stim) ====', 'AGENT PCn: {json}',
                     consolidated 'CFG str=.. dur=.. int=..', and CSV_PULSE rows.
  arduino_serial_v0  legacy stim_timing.txt: "HH:MM:SS.mmm -> <payload>", CSV_PULSE
                     header uses 'pulse_ms' (no interval_s), trial counted 'N/144'.

CSV_PULSE columns are bound from the header row present in the file, never by
fixed index, so both formats decode with one code path.
"""
import json
import os
import re

from . import naming
from .model import SessDoc, StimTrial

_ARDUINO_PREFIX_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} -> ")
_HEADER_RE = re.compile(r"^# session (sess_\S+) opened (.+)$")
_SESS_START_RE = re.compile(r"START\s+sess_\S+\s*\(([^)]*)\)")
_AGENT_RE = re.compile(r"^(PC[\w-]+):\s*(\{.*\})\s*$")
_CFG_TOKEN_RE = re.compile(r"\b(str|dur|int|seed|window|trials|dutycap)=(\S+)")
_KV_RE = re.compile(
    r"^\s*(strength|duration|interval|trials|window|seed|dutycap)\b[^:=]*[:=]\s*(\S+)",
    re.IGNORECASE)


def _detect_format(lines):
    head = lines[:80]
    if any(l.startswith("# session sess_") for l in head):
        return "sess_v1"
    if any(_ARDUINO_PREFIX_RE.match(l) for l in head):
        return "arduino_serial_v0"
    if any("\tSESS\t" in l or "\tAGENT\t" in l for l in head):
        return "sess_v1"
    return "unknown"


def _payload_v1(line):
    if line.startswith("#"):
        return line
    if "\t" in line:
        parts = line.split("\t")
        return parts[-1] if len(parts) >= 3 else line
    return line


def _payload_legacy(line):
    m = _ARDUINO_PREFIX_RE.match(line)
    return line[m.end():] if m else None


def _fnum(rec, key, cast=float):
    v = rec.get(key)
    if v is None or v == "":
        return None
    try:
        return cast(v)
    except (TypeError, ValueError):
        try:  # "5.0" as int, etc.
            return cast(float(v))
        except (TypeError, ValueError):
            return None


def _mk_trial(rec):
    t = StimTrial()
    t.iso_time = rec.get("iso_time", "")
    t.trial = _fnum(rec, "trial", int)
    t.duty = _fnum(rec, "duty")
    if "dur_s" in rec:
        t.dur_s = _fnum(rec, "dur_s")
    elif "pulse_ms" in rec:
        ms = _fnum(rec, "pulse_ms")
        t.dur_s = ms / 1000.0 if ms is not None else None
    if "interval_s" in rec:
        t.interval_s = _fnum(rec, "interval_s")
    t.cam_frame_start = _fnum(rec, "camFrameStart", int)
    t.cam_frame_end = _fnum(rec, "camFrameEnd", int) if "camFrameEnd" in rec else t.cam_frame_start
    t.fs_hz = _fnum(rec, "fs_hz", int)
    t.samples = _fnum(rec, "samples", int)
    t.gyro_rms_dps = _fnum(rec, "gyro_rms_dps")
    t.gyro_peak_dps = _fnum(rec, "gyro_peak_dps")
    t.acc_rms_g = _fnum(rec, "acc_rms_g")
    t.acc_peak_g = _fnum(rec, "acc_peak_g")
    t.temp_mean_C = _fnum(rec, "temp_mean_C")
    t.imu_ok = bool(t.samples) and not (
        (t.gyro_rms_dps in (None, 0.0)) and (t.acc_rms_g in (None, 0.0)))
    return t


def _apply_config_line(doc, payload):
    for k, v in _CFG_TOKEN_RE.findall(payload):
        if k == "str" and not doc.stim_strength:
            doc.stim_strength = v
        elif k == "dur" and not doc.stim_duration_s:
            doc.stim_duration_s = v
        elif k == "int" and not doc.stim_interval_s:
            doc.stim_interval_s = v
        elif k == "seed" and not doc.stim_seed:
            doc.stim_seed = v
        elif k == "window" and not doc.stim_window_min:
            doc.stim_window_min = v
        elif k == "trials" and not doc.stim_trials_cfg:
            doc.stim_trials_cfg = v
    m = _KV_RE.match(payload)
    if m:
        key, val = m.group(1).lower(), m.group(2)
        if key == "strength" and not doc.stim_strength:
            doc.stim_strength = val
        elif key == "duration" and not doc.stim_duration_s:
            doc.stim_duration_s = val
        elif key == "interval" and not doc.stim_interval_s:
            doc.stim_interval_s = val
        elif key == "seed" and not doc.stim_seed:
            doc.stim_seed = val
        elif key == "window" and not doc.stim_window_min:
            doc.stim_window_min = val
        elif key == "trials" and not doc.stim_trials_cfg:
            doc.stim_trials_cfg = val


def _parse_agent(doc, payload):
    m = _AGENT_RE.match(payload)
    if not m:
        return
    pc = m.group(1)
    try:
        data = json.loads(m.group(2))
    except ValueError:
        doc.warnings.append("bad AGENT json")
        return
    for f in data.get("armed", []) or []:
        base = os.path.basename(str(f).replace("\\", "/"))
        vn = naming.parse_video_name(base)
        drive = f[0].upper() if len(f) >= 2 and f[1] == ":" else ""
        if vn.cam_global is not None:
            doc.cam_pc_map[vn.cam_global] = (pc, drive)


def parse_sess_file(path, name_stim_hint=False):
    doc = SessDoc(path=str(path))
    try:
        if os.path.getsize(path) > 50 * 1024 * 1024:  # guard against a mis-named binary
            doc.stim_format = "unknown"
            doc.parse_error = "file too large (>50MB); not parsed"
            return doc
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError as e:
        doc.stim_format = "unknown"
        doc.parse_error = str(e)
        return doc

    fmt = _detect_format(lines)
    doc.stim_format = fmt
    extract = _payload_legacy if fmt == "arduino_serial_v0" else _payload_v1

    pulse_header = None
    for raw in lines:
        p = extract(raw)
        if p is None:
            continue
        p = p.strip()
        if not p:
            continue

        if p.startswith("CSV_PULSE,"):
            cols = p.split(",")[1:]
            if cols and cols[0] == "iso_time":
                pulse_header = cols
            elif pulse_header is not None:
                doc.trials.append(_mk_trial(dict(zip(pulse_header, cols))))
            continue

        if p.startswith("#"):
            m = _HEADER_RE.match(p)
            if m and not doc.opened:
                doc.opened = m.group(2).strip()
            _apply_config_line(doc, p)
            continue

        if "START" in p and "sess_" in p:
            sm = _SESS_START_RE.search(p)
            if sm:
                doc.is_stim = sm.group(1).strip().lower() == "stim"
                doc.stim_source = "sessfile"
            continue

        if ("STOP" in p or "closed" in p.lower()) and "sess_" in p:
            doc.clean_stop = True
            continue

        if p.startswith("PC") and "armed" in p:
            _parse_agent(doc, p)
            continue

        _apply_config_line(doc, p)

    if doc.is_stim is None:
        if doc.trials:
            doc.is_stim = True
            doc.stim_source = doc.stim_source or "sessfile"
        elif fmt == "arduino_serial_v0":
            doc.is_stim = True
            doc.stim_source = "sessfile"
        elif name_stim_hint:
            doc.is_stim = True
            doc.stim_source = "foldername"
    return doc
