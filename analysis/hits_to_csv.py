#!/usr/bin/env python3
"""
Parse stimulus hits (cam-frame events) from controller logs and write:
  - cam_frame_hits.csv
  - cam_frame_hits_only.txt

Supported log formats:
1) Legacy:
   [YYYY/MM/DD HH:MM:SS.mmm] Trigger pulse #N, Duty XX.X%, Duration D ms, CamFrame F

2) New (as in stim_timing.txt):
   [YYYY-MM-DD HH:MM:SS] PWM trial N/..., duty=XX.X%, CamFrame=F
   CSV_PULSE,iso_time,trial,duty,pulse_ms,camFrameStart,camFrameEnd,...
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re

import pandas as pd


DEFAULT_LOG_PATH = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/single_ants/test_12h_vib/stim_timing.txt"
)


PREFIX = r"(?:^\d{2}:\d{2}:\d{2}\.\d{3}\s*->\s*)?"

PAT_LEGACY = re.compile(
    PREFIX
    + r"""
    \[(?P<rtc>\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{3})?)\]
    \s+Trigger\spulse\s+\#(?P<pulse>\d+),\s+
    Duty\s+(?P<duty>\d+(?:\.\d+)?)%,\s+
    Duration\s+(?P<duration_ms>\d+)\s+ms,\s+
    CamFrame\s+(?P<cam_frame>\d+)
    """,
    re.VERBOSE | re.MULTILINE,
)

PAT_PWM_TRIAL = re.compile(
    PREFIX
    + r"""
    \[(?P<rtc>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]
    \s+PWM\strial\s+(?P<trial>\d+)\/\d+\s+
    duty=(?P<duty>\d+(?:\.\d+)?)%,\s+
    CamFrame=(?P<cam_frame>\d+)
    """,
    re.VERBOSE | re.MULTILINE,
)

PAT_CSV_PULSE = re.compile(
    PREFIX
    + r"""
    CSV_PULSE,
    (?P<iso_time>[^,\r\n]+),
    (?P<trial>\d+),
    (?P<duty_frac>[-+]?\d+(?:\.\d+)?),
    (?P<pulse_ms>\d+),
    (?P<cam_frame_start>-?\d+),
    (?P<cam_frame_end>-?\d+)
    (?:,|\r|\n|$)
    """,
    re.VERBOSE | re.MULTILINE,
)


def parse_legacy(log_text: str) -> pd.DataFrame:
    rows = []
    for m in PAT_LEGACY.finditer(log_text):
        rows.append(
            {
                "rtc_time": m.group("rtc"),
                "pulse_n": int(m.group("pulse")),
                "duty_pct": float(m.group("duty")),
                "duration_ms": int(m.group("duration_ms")),
                "cam_frame": int(m.group("cam_frame")),
                "source": "legacy_trigger",
            }
        )
    return pd.DataFrame(rows)


def parse_pwm_trial(log_text: str) -> pd.DataFrame:
    rows = []
    for m in PAT_PWM_TRIAL.finditer(log_text):
        rows.append(
            {
                "rtc_time": m.group("rtc"),
                "pulse_n": int(m.group("trial")),
                "duty_pct": float(m.group("duty")),
                "duration_ms": pd.NA,
                "cam_frame": int(m.group("cam_frame")),
                "source": "pwm_trial",
            }
        )
    return pd.DataFrame(rows)


def parse_csv_pulse(log_text: str) -> pd.DataFrame:
    rows = []
    for m in PAT_CSV_PULSE.finditer(log_text):
        rows.append(
            {
                "rtc_time": m.group("iso_time").strip(),
                "pulse_n": int(m.group("trial")),
                "duty_pct": float(m.group("duty_frac")) * 100.0,
                "duration_ms": int(m.group("pulse_ms")),
                "cam_frame_start": int(m.group("cam_frame_start")),
                "cam_frame_end": int(m.group("cam_frame_end")),
                "source": "csv_pulse",
            }
        )
    return pd.DataFrame(rows)


def normalize_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["rtc_dt"] = pd.to_datetime(out["rtc_time"], errors="coerce")
    if out["rtc_dt"].notna().any():
        out = out.sort_values(["rtc_dt", "pulse_n"], kind="mergesort")
    else:
        out = out.sort_values(["pulse_n"], kind="mergesort")
    out = out.drop(columns=["rtc_dt"]).reset_index(drop=True)

    if "duration_ms" in out.columns:
        out["duration_ms"] = pd.to_numeric(out["duration_ms"], errors="coerce").astype("Int64")
    return out


def build_hits_dataframe(log_text: str) -> pd.DataFrame:
    # Preferred extraction order for newer logs:
    # 1) PWM trial CamFrame (stim timing frame)
    # 2) Legacy Trigger pulse
    # 3) CSV_PULSE camFrameStart fallback
    df_trial = parse_pwm_trial(log_text)
    df_legacy = parse_legacy(log_text)
    df_csv = parse_csv_pulse(log_text)

    if not df_trial.empty:
        base = df_trial
        if not df_csv.empty:
            duration_map = (
                df_csv.drop_duplicates(subset=["pulse_n"], keep="last")
                .set_index("pulse_n")["duration_ms"]
            )
            base = base.copy()
            base["duration_ms"] = (
                base["pulse_n"].map(duration_map).astype("Int64")
            )
        return normalize_and_sort(base)

    if not df_legacy.empty:
        return normalize_and_sort(df_legacy)

    if not df_csv.empty:
        base = df_csv.rename(columns={"cam_frame_start": "cam_frame"}).copy()
        keep_cols = ["rtc_time", "pulse_n", "duty_pct", "duration_ms", "cam_frame", "source"]
        return normalize_and_sort(base[keep_cols])

    return pd.DataFrame(
        columns=["rtc_time", "pulse_n", "duty_pct", "duration_ms", "cam_frame", "source"]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log_path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to stim/controller log text file.",
    )
    args = ap.parse_args()

    log_path = args.log_path
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    log_text = log_path.read_text(errors="replace")
    out_dir = log_path.parent

    df = build_hits_dataframe(log_text)
    if df.empty:
        raise ValueError(
            "No hits found. Expected legacy Trigger pulse lines or new PWM trial/CSV_PULSE lines."
        )

    out_csv = out_dir / "cam_frame_hits.csv"
    df.to_csv(out_csv, index=False)

    cam_frames = pd.to_numeric(df["cam_frame"], errors="coerce").dropna().astype(int).tolist()
    out_frames_txt = out_dir / "cam_frame_hits_only.txt"
    out_frames_txt.write_text("\n".join(map(str, cam_frames)) + "\n")

    print(f"Parsed {len(df)} hits.")
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote cam_frame list: {out_frames_txt}")
    print("First 5 rows:")
    print(df.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
