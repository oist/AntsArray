#%%
%matplotlib qt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.sleep_analysis import classify_sleep_wake_from_sleap


pkl_path = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20251117_2_stim/chunk000_left.pkl")
sleap_df = pd.read_pickle(pkl_path)


#sleep_out=classify_sleep_wake_from_sleap(sleap_df, speed_smooth_window=241,fps=25.0,thr_speed_pix_s=50.0,max_speed_pix_s=1500)

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

fps = 24.0  # adjust if needed

# -----------------------
# select single TrackID
# -----------------------
x = 81  # <-- TrackID to plot

sleep_out_x = (
    sleep_out.loc[sleep_out["TrackID"] == x]
    .sort_values("Frame")
    .copy()
)

if sleep_out_x.empty:
    raise ValueError(f"No rows found for TrackID = {x}")

# -----------------------
# x data
# -----------------------
frames = sleep_out_x["Frame"].to_numpy()

if "global_time_s" in sleep_out_x.columns:
    t_sec = sleep_out_x["global_time_s"].to_numpy()
else:
    t_sec = frames / fps

speed = sleep_out_x["speed_pix_s"].to_numpy()
sleep = sleep_out_x["is_sleep"].astype(int).to_numpy()

# -----------------------
# formatter: seconds -> HH:MM:SS
# -----------------------
def sec_to_hhmmss(x, pos=None):
    if not np.isfinite(x):
        return ""
    total = int(x)
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

# -----------------------
# plot
# -----------------------
fig, ax_time = plt.subplots(figsize=(12, 3))

# Speed
ax_time.plot(t_sec, speed, color="black", linewidth=1, label="Speed")
ax_time.set_xlabel("Time (HH:MM:SS)")
ax_time.set_ylabel("Speed (px/s)")
ax_time.xaxis.set_major_formatter(mticker.FuncFormatter(sec_to_hhmmss))

# Sleep overlay
ax_sleep = ax_time.twinx()
ax_sleep.plot(t_sec, sleep, color="red", alpha=0.6, linewidth=1.0, label="Sleep")
ax_sleep.set_ylabel("Is Sleep (0/1)")
ax_sleep.set_yticks([0, 1])

# -----------------------
# top x-axis: frames
# -----------------------
ax_frame = ax_time.twiny()
ax_frame.set_xlim(ax_time.get_xlim())

n_ticks = 8
frame_ticks = np.linspace(frames.min(), frames.max(), n_ticks).astype(int)
time_ticks = frame_ticks / fps

ax_frame.set_xticks(time_ticks)
ax_frame.set_xticklabels(frame_ticks)
ax_frame.set_xlabel("Frame")

# -----------------------
# title / layout
# -----------------------
plt.title(f"Speed and Sleep/Wake Classification (TrackID = {x})")
plt.tight_layout()
plt.show()





# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

fps = 24.0  # adjust if needed

# -----------------------
# select single TrackID + Bodypoint
# -----------------------
track_id = 52
bodypoint = 0

df_x_bp = (
    sleap_df.loc[
        (sleap_df["TrackID"] == track_id) &
        (sleap_df["Bodypoint"] == bodypoint)
    ]
    .copy()
)

if df_x_bp.empty:
    raise ValueError(f"No rows found for TrackID={track_id}, Bodypoint={bodypoint}")

# Coerce and sort
df_x_bp["Frame"] = pd.to_numeric(df_x_bp["Frame"], errors="coerce")
df_x_bp["X"] = pd.to_numeric(df_x_bp["X"], errors="coerce")
df_x_bp["Y"] = pd.to_numeric(df_x_bp["Y"], errors="coerce")

df_x_bp = (
    df_x_bp.dropna(subset=["Frame"])
           .sort_values("Frame", kind="mergesort")
           .drop_duplicates(subset=["Frame"], keep="first")
)

frames = df_x_bp["Frame"].astype(int).to_numpy()

# Time per point
if "global_time_s" in df_x_bp.columns:
    t_sec = pd.to_numeric(df_x_bp["global_time_s"], errors="coerce").to_numpy(float)
else:
    t_sec = frames / fps

X = df_x_bp["X"].to_numpy(float)
Y = df_x_bp["Y"].to_numpy(float)

valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(t_sec)
Xv, Yv, Tv = X[valid], Y[valid], t_sec[valid]

if Xv.size == 0:
    raise ValueError("No finite XY points to plot")

# -----------------------
# formatter: seconds -> HH:MM:SS
# -----------------------
def sec_to_hhmmss(val, pos=None):
    if not np.isfinite(val):
        return ""
    total = int(val)
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

# -----------------------
# plot: XY with time encoded by color
# -----------------------
fig, ax = plt.subplots(figsize=(6, 6))

sc = ax.scatter(Xv, Yv, c=Tv, s=8, linewidths=0)

ax.set_xlabel("X (px)")
ax.set_ylabel("Y (px)")
ax.set_title(f"XY trajectory (TrackID={track_id}, Bodypoint={bodypoint})")

# Enforce 1 px == 1 px
ax.set_aspect("equal", adjustable="box")

# Pad limits slightly
pad = 2.0
ax.set_xlim(Xv.min() - pad, Xv.max() + pad)
ax.set_ylim(Yv.min() - pad, Yv.max() + pad)

# Colorbar in separate axes (does NOT break aspect)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="4%", pad=0.1)

#cbar = plt.colorbar(sc, cax=cax)
#cbar.set_label("Time (HH:MM:SS)")
#cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(sec_to_hhmmss))

#plt.tight_layout()
plt.show()


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# params
# -----------------------
track_id = 52
bodypoint = 0

# -----------------------
# 1) pull XY from sleap_df
# -----------------------
xy = (
    sleap_df.loc[(sleap_df["TrackID"] == track_id) & (sleap_df["Bodypoint"] == bodypoint), ["Frame", "X", "Y"]]
    .copy()
)

if xy.empty:
    raise ValueError(f"No XY rows found for TrackID={track_id}, Bodypoint={bodypoint}")

xy["Frame"] = pd.to_numeric(xy["Frame"], errors="coerce")
xy["X"] = pd.to_numeric(xy["X"], errors="coerce")
xy["Y"] = pd.to_numeric(xy["Y"], errors="coerce")

xy = (
    xy.dropna(subset=["Frame"])
      .sort_values("Frame", kind="mergesort")
      .drop_duplicates(subset=["Frame"], keep="first")
)

xy["Frame"] = xy["Frame"].astype(int)

# -----------------------
# 2) pull speed + sleep from sleep_out
# -----------------------
ss = (
    sleep_out.loc[sleep_out["TrackID"] == track_id, ["Frame", "speed_pix_s", "is_sleep"]]
    .copy()
)

if ss.empty:
    raise ValueError(f"No sleep_out rows found for TrackID={track_id}")

ss["Frame"] = pd.to_numeric(ss["Frame"], errors="coerce")
ss = ss.dropna(subset=["Frame"]).sort_values("Frame", kind="mergesort")
ss["Frame"] = ss["Frame"].astype(int)

# Ensure numeric
ss["speed_pix_s"] = pd.to_numeric(ss["speed_pix_s"], errors="coerce")
# is_sleep might be bool already
ss["is_sleep"] = ss["is_sleep"].astype(int)

# -----------------------
# 3) align by frame (inner join)
# -----------------------
m = xy.merge(ss, on="Frame", how="inner")

# Keep only finite XY and finite speed
valid = np.isfinite(m["X"].to_numpy(float)) & np.isfinite(m["Y"].to_numpy(float)) & np.isfinite(m["speed_pix_s"].to_numpy(float))
m = m.loc[valid].copy()

if m.empty:
    raise ValueError("After aligning frames and filtering finite values, nothing left to plot.")

X = m["X"].to_numpy(float)
Y = m["Y"].to_numpy(float)
speed = m["speed_pix_s"].to_numpy(float)
is_sleep = m["is_sleep"].to_numpy(int)

# -----------------------
# 4) plot
# -----------------------
fig, ax = plt.subplots(figsize=(7, 7))

sc = ax.scatter(
    X, Y,
    c=speed,
    s=10,
    linewidths=0,
    alpha=0.9
)

# overlay sleep points in red
sleep_mask = (is_sleep == 1)
ax.scatter(
    X[sleep_mask], Y[sleep_mask],
    s=16,
    c="red",
    linewidths=0,
    alpha=0.9,
    label="Sleep"
)

ax.set_xlabel("X (px)")
ax.set_ylabel("Y (px)")
ax.set_title(f"XY position colored by speed; sleep in red (TrackID={track_id}, Bodypoint={bodypoint})")

ax.set_aspect("equal", adjustable="box")

# colorbar = speed
cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Speed (px/s)")

ax.legend(loc="best")
plt.tight_layout()
plt.show()

# %%
