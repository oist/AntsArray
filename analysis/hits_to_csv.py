# %% Parse cam-frame hits from the log text and write to CSV (same folder as input)
from pathlib import Path
import re
import pandas as pd

# -----------------------
# input
# -----------------------
log_path = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20251117_2_stim/Stim_timetable.txt")
log_text = log_path.read_text()

out_dir = log_path.parent  # <-- key change

# -----------------------
# regex: extract pulse events with CamFrame
# -----------------------
pat = re.compile(
    r"""
    \[(?P<rtc>\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\]
    \s+Trigger\spulse\s+\#(?P<pulse>\d+),\s+
    Duty\s+(?P<duty>\d+(?:\.\d+)?)%,\s+
    Duration\s+(?P<duration_ms>\d+)\s+ms,\s+
    CamFrame\s+(?P<cam_frame>\d+)
    """,
    re.VERBOSE,
)

rows = []
for m in pat.finditer(log_text):
    rows.append({
        "rtc_time": m.group("rtc"),
        "pulse_n": int(m.group("pulse")),
        "duty_pct": float(m.group("duty")),
        "duration_ms": int(m.group("duration_ms")),
        "cam_frame": int(m.group("cam_frame")),
    })

df = pd.DataFrame(rows).sort_values("pulse_n").reset_index(drop=True)

if df.empty:
    raise ValueError("No Trigger pulse / CamFrame hits found. Check log content.")

# -----------------------
# outputs (same folder)
# -----------------------
out_csv = out_dir / "cam_frame_hits.csv"
df.to_csv(out_csv, index=False)

cam_frames = df["cam_frame"].tolist()
out_frames_txt = out_dir / "cam_frame_hits_only.txt"
out_frames_txt.write_text("\n".join(map(str, cam_frames)) + "\n")

print(f"Parsed {len(df)} hits.")
print(f"Wrote CSV: {out_csv}")
print(f"Wrote cam_frame list: {out_frames_txt}")
print("First 5 rows:")
print(df.head())

# %%
