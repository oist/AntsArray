
#%% Plot the XY positions of a given bodypoint for each TrackID,with sleep points highlighted in red.
# 
%matplotlib qt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# params
# -----------------------
bodypoint = 0
out_dir = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20251117_2_stim/xy_speed_sleep_pngs")  # change as desired
out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------
# global XY limits (once)
# -----------------------
xy_all = sleap_df.loc[sleap_df["Bodypoint"] == bodypoint, ["X", "Y"]].copy()
xy_all["X"] = pd.to_numeric(xy_all["X"], errors="coerce")
xy_all["Y"] = pd.to_numeric(xy_all["Y"], errors="coerce")
xy_all = xy_all[np.isfinite(xy_all["X"]) & np.isfinite(xy_all["Y"])]

if xy_all.empty:
    raise ValueError("No finite XY values found for global limits.")

pad = 2.0  # pixels
xlim_global = (xy_all["X"].min() - pad, xy_all["X"].max() + pad)
ylim_global = (xy_all["Y"].min() - pad, xy_all["Y"].max() + pad)

# -----------------------
# unique TrackIDs
# -----------------------
track_ids = (
    pd.to_numeric(sleap_df["TrackID"], errors="coerce")
      .dropna()
      .astype(int)
      .unique()
)
track_ids = np.sort(track_ids)

# -----------------------
# batch plot
# -----------------------
saved = 0
skipped = 0

for tid in track_ids:
    # --- XY ---
    xy = sleap_df.loc[
        (sleap_df["TrackID"] == tid) & (sleap_df["Bodypoint"] == bodypoint),
        ["Frame", "X", "Y"]
    ].copy()

    if xy.empty:
        skipped += 1
        continue

    xy["Frame"] = pd.to_numeric(xy["Frame"], errors="coerce")
    xy["X"] = pd.to_numeric(xy["X"], errors="coerce")
    xy["Y"] = pd.to_numeric(xy["Y"], errors="coerce")

    xy = (
        xy.dropna(subset=["Frame"])
          .sort_values("Frame", kind="mergesort")
          .drop_duplicates(subset=["Frame"], keep="first")
    )
    xy["Frame"] = xy["Frame"].astype(int)

    # --- speed + sleep ---
    ss = sleep_out.loc[
        sleep_out["TrackID"] == tid,
        ["Frame", "speed_pix_s", "is_sleep"]
    ].copy()

    if ss.empty:
        skipped += 1
        continue

    ss["Frame"] = pd.to_numeric(ss["Frame"], errors="coerce")
    ss["speed_pix_s"] = pd.to_numeric(ss["speed_pix_s"], errors="coerce")
    ss = ss.dropna(subset=["Frame"]).sort_values("Frame", kind="mergesort")
    ss["Frame"] = ss["Frame"].astype(int)
    ss["is_sleep"] = ss["is_sleep"].astype(int)

    # --- align by frame ---
    m = xy.merge(ss, on="Frame", how="inner")

    finite = (
        np.isfinite(m["X"].to_numpy(float)) &
        np.isfinite(m["Y"].to_numpy(float)) &
        np.isfinite(m["speed_pix_s"].to_numpy(float))
    )
    m = m.loc[finite].copy()

    if m.empty:
        skipped += 1
        continue

    X = m["X"].to_numpy(float)
    Y = m["Y"].to_numpy(float)
    speed = m["speed_pix_s"].to_numpy(float)
    is_sleep = m["is_sleep"].to_numpy(int)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 7))

    sc = ax.scatter(X, Y, c=speed, s=10, linewidths=0, alpha=0.9)

    sleep_mask = (is_sleep == 1)
    if np.any(sleep_mask):
        ax.scatter(
            X[sleep_mask], Y[sleep_mask],
            s=18, c="red", linewidths=0, alpha=0.9, label="Sleep"
        )
        ax.legend(loc="best")

    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_title(f"XY colored by speed; sleep in red (TrackID={tid}, Bodypoint={bodypoint})")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim_global)
    ax.set_ylim(*ylim_global)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Speed (px/s)")

    plt.tight_layout()
    fig.savefig(out_dir / f"TrackID_{tid:04d}_Bodypoint_{bodypoint}.png", dpi=200)
    plt.close(fig)

    saved += 1

print(f"Saved {saved} PNGs to {out_dir.resolve()}")
print(f"Skipped {skipped} tracks.")

#%%
%matplotlib qt
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.sleep_analysis import classify_sleep_wake_from_sleap


# --------------------
# Parameters
# --------------------
# video / acquisition
FPS = 24.0

# speed computation / smoothing
SPEED_BODYPOINT = 0
MAX_INTERP_GAP = 24*50        # interpolate gaps up to this many frames (for smoothing support)
XY_SMOOTH_SIGMA = 24*5    # gaussian sigma in frames; larger => smoother speed
SPEED_SMOOTH_WINDOW = None  # only used if XY_SMOOTH_SIGMA <= 0

# sanity / clipping
MAX_SPEED_PIX_S = 1500.0

# sleep thresholds
THR_INL = 90.0
THR_INR = 90.0
THR_OUTL = 130.0
THR_OUTR = 130.0
THR_SPEED_PIX_S = 70.0

# --------------------
# Load parquet
# --------------------
file = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/20251118-121513/per_track_left/TrackID_0151_all_121514_left.parquet"
)
df = pd.read_parquet(file, engine="pyarrow")


#file = Path(
 #   "/home/sam-reiter/bucket/ReiterU/Ants/basler/single_ants/test_2/20251218-163512/data/cam02_cam1_2025-12-18-16-35-11_002_sleap_data.csv"
#)
#df = pd.read_csv(file)

# --------------------
# Compute speed
# --------------------
sleep_out = classify_sleep_wake_from_sleap(
    df,
    fps=FPS,
    speed_bodypoint=SPEED_BODYPOINT,
    max_interp_gap=MAX_INTERP_GAP,
    xy_smooth_sigma=XY_SMOOTH_SIGMA,
    speed_smooth_window=SPEED_SMOOTH_WINDOW,
    max_speed_pix_s=MAX_SPEED_PIX_S,
    thr_inL=THR_INL,
    thr_inR=THR_INR,
    thr_outL=THR_OUTL,
    thr_outR=THR_OUTR,
    thr_speed_pix_s=THR_SPEED_PIX_S,
)
# Ensure clean, ordered frame axis
sleep_out["Frame"] = pd.to_numeric(sleep_out["Frame"], errors="coerce")
sleep_out = sleep_out.dropna(subset=["Frame"]).copy()
sleep_out["Frame"] = sleep_out["Frame"].astype(int)
sleep_out = sleep_out.sort_values("Frame", kind="mergesort")

frame_vec = sleep_out["Frame"].to_numpy(dtype=int)
speed_vec = pd.to_numeric(
    sleep_out["speed_pix_s"], errors="coerce"
).to_numpy(dtype=float)

#%%
x_vec = pd.to_numeric(
    sleep_out["speed_X_s"], errors="coerce"
).to_numpy(dtype=float)

y_vec = pd.to_numeric(
    sleep_out["speed_Y_s"], errors="coerce"
).to_numpy(dtype=float)

#%%
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))

axes[0].plot(frame_vec, speed_vec)
axes[0].set_ylabel("Speed (pix/s)")
axes[0].set_title(file.name)

axes[1].plot(frame_vec, x_vec)
axes[1].set_ylabel("X position (pix)")

axes[2].plot(frame_vec, y_vec)
axes[2].set_ylabel("Y position (pix)")
axes[2].set_xlabel("Frame")

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

FPS = 24.0

def frame_to_hms(x, pos):
    """
    Convert frame index to HH:MM:SS at fixed FPS.
    """
    total_seconds = x / FPS
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))

axes[0].plot(frame_vec, speed_vec)
axes[0].set_ylabel("Speed (pix/s)")
axes[0].set_title(file.name)

axes[1].plot(frame_vec, x_vec)
axes[1].set_ylabel("X position (pix)")

axes[2].plot(frame_vec, y_vec)
axes[2].set_ylabel("Y position (pix)")
axes[2].set_xlabel("Time (HH:MM:SS)")

# Apply formatter to shared x-axis
axes[2].xaxis.set_major_formatter(FuncFormatter(frame_to_hms))

plt.tight_layout()
plt.show()


# %%
