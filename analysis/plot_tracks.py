import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1)  LOAD / PREPARE YOUR DATA
# --------------------------------------------------
# If your data are in a CSV, uncomment the next line and supply the path.
df = pd.read_pickle("/home/sam-reiter/bucket/ReiterU/Ants/basler/20250321_2_test/test_data/cam06_cam5_2025-03-21-15-17-39_001.tracks")

# If the DataFrame is already in memory, just make sure it’s called `df`.
assert set(["Frame", "TrackID", "Bodypoint", "X", "Y"]).issubset(df.columns), \
       "DataFrame is missing required columns."

# Focus only on Bodypoint 0
bp0 = df[df["Bodypoint"] == 0].copy()

# --------------------------------------------------
# 2)  PLOT EACH TRACK
# --------------------------------------------------
out_dir = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20250321_2_test/test_data/plots")
out_dir.mkdir(exist_ok=True)

import matplotlib.pyplot as plt

for tid, grp in bp0.groupby("TrackID"):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    sc = ax.scatter(grp["X"], grp["Y"], c=grp["Frame"], cmap="viridis", s=8)
    ax.set_title(f"Track {tid} – X vs Y (time-colored)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()  # Optional: depends on coordinate system (e.g., image coordinates)
    
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Frame")

    fig.tight_layout()
    fig.savefig(out_dir / f"track_{tid}_bodypoint0_xy.png", dpi=300)
    plt.close(fig)


print(f"Saved {len(bp0['TrackID'].unique())} figure(s) in {out_dir.resolve()}")

# --------------------------------------------------
# 3)  SPEED ESTIMATION  (NO interpolation across gaps)
# --------------------------------------------------
import numpy as np

FPS         = 30        # frames · s⁻¹
PIXEL_TO_MM = 10        # px → mm

# Sort within every (TrackID, Bodypoint) so Frame is monotonic
df_srt = df.sort_values(["TrackID", "Bodypoint", "Frame"]).copy()

# Shift X, Y and Frame to compute finite differences
for col in ["X", "Y", "Frame"]:
    df_srt[f"prev_{col}"] = df_srt.groupby(["TrackID", "Bodypoint"])[col].shift()

# ΔFrame lets us detect gaps
df_srt["dF"] = df_srt["Frame"] - df_srt["prev_Frame"]

# Speed only for *consecutive* frames (dF == 1); else NaN
mask_consecutive = df_srt["dF"] == 1
dx   = df_srt["X"]  - df_srt["prev_X"]
dy   = df_srt["Y"]  - df_srt["prev_Y"]

df_srt["speed_mm_s"] = np.where(
    mask_consecutive,
    np.hypot(dx, dy) * FPS * PIXEL_TO_MM,   # mm · s⁻¹
    np.nan                                   # leave gaps as NaN
)

# --------------------------------------------------
# 3A)  COLLAPSE TO ONE SPEED PER (TrackID, Frame)
# --------------------------------------------------
track_speed = (df_srt
               .groupby(["TrackID", "Frame"], sort=False)["speed_mm_s"]
               .mean()
               .reset_index())

# --------------------------------------------------
# 3B)  PLOT SPEED-vs-TIME  (unchanged)
# --------------------------------------------------
for tid, g in track_speed.groupby("TrackID"):
    fig, ax = plt.subplots(figsize=(8, 4))
    t_frame = g["Frame"] 
    ax.plot(t_frame, g["speed_mm_s"], lw=1)
    ax.set_title(f"Track {tid} – mean speed (no interpolation across gaps)")
    ax.set_xlabel("Time (frame)")
    ax.set_ylabel("Speed (mm · s⁻¹)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"track_{tid}_speed_mm_s.png", dpi=300)
    plt.close(fig)

# --------------------------------------------------
# 3C)  GLOBAL HISTOGRAM  (unchanged)
# --------------------------------------------------
speeds = track_speed["speed_mm_s"].dropna()
if not speeds.empty:
    bins = np.logspace(np.log10(speeds.min()), np.log10(speeds.max()), 50)
    fig, ax = plt.subplots()
    ax.hist(speeds, bins=bins, density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Speed (mm · s⁻¹)")
    ax.set_ylabel("Probability density")
    ax.set_title("All tracks – distribution of speed (no interpolation)")
    ax.grid(True, which="both", axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "all_tracks_speed_histogram.png", dpi=300)
    plt.show()
    plt.close(fig)
