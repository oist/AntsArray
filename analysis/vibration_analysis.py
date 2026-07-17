#%%
def _enable_qt_matplotlib() -> None:
    try:
        get_ipython().run_line_magic("matplotlib", "qt")  # type: ignore[name-defined]
    except Exception:
        pass


_enable_qt_matplotlib()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.sleep_analysis_utils import classify_sleep_wake_from_sleap

#%% Testing parquet files. I have converted the tracking pkl
# output to parquet for faster loading. Can change to just save parquet directly in future.
#then used parquet files to generate single ant data over chunks.
#Now testing sleep classification on parquet data using the hits.
_enable_qt_matplotlib()
import numpy as np
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.sleep_analysis_utils import classify_sleep_wake_from_sleap, get_event_trig_avg

# -----------------------
# CONFIG
# -----------------------
per_track_dir = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/single_ants/test_12h_vib/per_track")
pattern = "TrackID_0000_all_cam06_Arena00.parquet"

hit_csv = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/single_ants/test_12h_vib/cam_frame_hits.csv")

FPS = 24.0
# speed computation / smoothing
SPEED_BODYPOINT = 0
MAX_INTERP_GAP = 24*50        # interpolate gaps up to this many frames (for smoothing support)
XY_SMOOTH_SIGMA = 24*5    # gaussian sigma in frames; larger => smoother speed
SPEED_SMOOTH_WINDOW = None  # only used if XY_SMOOTH_SIGMA <= 0

# sanity / clipping
MAX_SPEED_PIX_S = 1500.0

backlag = int(24 * 15)    # 15 s pre
forwardlag = int(24 * 200)  # 200 s post
lags = np.arange(-backlag, forwardlag + 1, dtype=int)
T = lags.size

pad = 1  # 1-sample guard band at BOTH ends to prevent edge-event dropping in get_event_trig_avg

out_dir = per_track_dir / "event_triggered_tensors"
out_dir.mkdir(parents=True, exist_ok=True)
out_npz = out_dir / "event_triggered_speed_sleep_tensors.npz"
out_meta_csv = out_dir / "event_triggered_ant_metadata.csv"

# -----------------------
# LOAD HIT FRAMES + STRENGTH (paired + sorted)
# -----------------------
hits = pd.read_csv(hit_csv)
hits["cam_frame"] = pd.to_numeric(hits["cam_frame"], errors="coerce")
hits["duty_pct"] = pd.to_numeric(hits["duty_pct"], errors="coerce")
hits = hits.dropna(subset=["cam_frame", "duty_pct"]).sort_values("cam_frame").reset_index(drop=True)

hit_frames = hits["cam_frame"].astype(int).to_numpy()
hit_strength = hits["duty_pct"].to_numpy(float)

E = hit_frames.size
if E == 0:
    raise ValueError("No hit frames loaded.")

# -----------------------
# COMMON WINDOW [f0..f1] (with guard band)
# -----------------------
f0 = int(hit_frames[0]) - backlag - pad
f1 = int(hit_frames[-1]) + forwardlag + pad
if f0 < 0:
    raise ValueError(f"Window start negative (f0={f0}). Reduce backlag/pad or shift frames.")

W = (f1 - f0) + 1  # length of speed_win/sleep_win
event_inds_global = (hit_frames - f0).astype(int)

# Sanity: no events should be dropped by get_event_trig_avg
valid_mask_global = (event_inds_global > backlag) & (event_inds_global < (W - forwardlag))
if valid_mask_global.sum() != E:
    bad = np.where(~valid_mask_global)[0]
    raise ValueError(
        f"{len(bad)} events would be dropped even with pad={pad}.\n"
        f"Bad hit_frames: {hit_frames[bad].tolist()}\n"
        f"Bad event_inds: {event_inds_global[bad].tolist()}\n"
        f"(W={W}, backlag={backlag}, forwardlag={forwardlag}). Increase pad."
    )

# Sorting order by increasing stimulus strength (same for all ants)
order_strength = np.argsort(hit_strength)

# -----------------------
# LOOP OVER ANTS
# -----------------------
files = sorted(per_track_dir.glob(pattern))
if not files:
    raise RuntimeError(f"No per-track parquet files found in {per_track_dir} matching {pattern}")

speed_tensor_list = []  # each entry: (E, T)
sleep_tensor_list = []  # each entry: (E, T)
track_id_list = []
meta_rows = []
first_frame_vec = None
first_speed_vec = None
first_track_id = None

for fp in tqdm(files, desc="Ants (TrackIDs)", unit="ant", dynamic_ncols=True):

    m = re.match(r"TrackID_(\d+)_all_cam\d+_Arena\d+$", fp.stem)

    track_id = int(m.group(1))

    df = pd.read_parquet(fp, engine="pyarrow")


    # Compute per-frame speed + sleep. Assumption: frames are 0..N with NaNs where missing.
    sleep_out = classify_sleep_wake_from_sleap(
    df,
    fps=FPS,
    speed_bodypoint=SPEED_BODYPOINT,
    max_interp_gap=MAX_INTERP_GAP,
    xy_smooth_sigma=XY_SMOOTH_SIGMA,
    speed_smooth_window=SPEED_SMOOTH_WINDOW,
    max_speed_pix_s=MAX_SPEED_PIX_S,
    sleep_median_win_sec=20,
    output_frame_index="full"

)

    sleep_out = sleep_out.sort_values("Frame", kind="mergesort")

    frame_vec = pd.to_numeric(sleep_out["Frame"], errors="coerce").to_numpy(dtype=int)
    speed_vec = pd.to_numeric(sleep_out["speed_pix_s"], errors="coerce").to_numpy(dtype=float)
    sleep_vec = sleep_out["is_sleep"].astype(float).to_numpy()

    if first_frame_vec is None:
        first_frame_vec = frame_vec.copy()
        first_speed_vec = speed_vec.copy()
        first_track_id = track_id

    # Preallocate fixed window vectors for THIS ant
    speed_win = np.full(W, np.nan, dtype=float)
    sleep_win = np.full(W, np.nan, dtype=float)

    # Fill overlap where ant has frames within [f0, f1]
    in_win = (frame_vec >= f0) & (frame_vec <= f1)
    if np.any(in_win):
        dst_idx = frame_vec[in_win] - f0               # 0..W-1
        src_idx = np.flatnonzero(in_win)

        # Safety (guards against any unexpected frames)
        ok = (dst_idx >= 0) & (dst_idx < W)
        dst_idx = dst_idx[ok]
        src_idx = src_idx[ok]

        speed_win[dst_idx] = speed_vec[src_idx]
        sleep_win[dst_idx] = sleep_vec[src_idx]

    # Compute event-triggered matrices using the same event_inds for all ants
    speed_avg, speed_ev_mat = get_event_trig_avg(speed_win, event_inds_global, backlag, forwardlag)
    sleep_avg, sleep_ev_mat = get_event_trig_avg(sleep_win, event_inds_global, backlag, forwardlag)

    # get_event_trig_avg drops edge events; with pad>=1 it should not.
    if not isinstance(speed_ev_mat, np.ndarray) or speed_ev_mat.shape[0] != E:
        raise ValueError(
            f"Event drop detected for TrackID={track_id}: "
            f"rows={None if not isinstance(speed_ev_mat, np.ndarray) else speed_ev_mat.shape[0]} vs E={E}. "
            f"Increase pad or check hit_frames/f0/f1."
        )

    # Sort events by increasing hit strength
    speed_ev_mat = speed_ev_mat[order_strength, :]
    sleep_ev_mat = sleep_ev_mat[order_strength, :]

    speed_tensor_list.append(speed_ev_mat)  # (E, T)
    sleep_tensor_list.append(sleep_ev_mat)  # (E, T)
    track_id_list.append(track_id)



# %% Diagnostic plots (speed + sleep) with FIXED hit-axis direction and sanity checks
# Produces:
#   (A) Grand-average SPEED trace (ants × hits)
#   (B) SPEED hit×time matrix (mean over ants)
#   (C) Grand-average SLEEP probability trace (ants × hits)
#   (D) SLEEP hit×time matrix (mean over ants)

import numpy as np
import matplotlib.pyplot as plt

# ---- REQUIRED INPUTS (must already exist) ----
# speed_tensor_list: list of (E, T) arrays
# sleep_tensor_list: list of (E, T) arrays
# backlag, forwardlag, fps
# hit_strength_sorted OR hit_strength + order_strength

# -----------------------
# stack + checks
# -----------------------
speed_tensor = np.stack(speed_tensor_list, axis=0)   # (A, E, T)
sleep_tensor = np.stack(sleep_tensor_list, axis=0)   # (A, E, T)

A, E, T = speed_tensor.shape
assert sleep_tensor.shape == (A, E, T), "Speed/Sleep tensor shape mismatch."

t = np.linspace(-backlag / fps, forwardlag / fps, T)

# Hit-strength ordering (for annotation / sanity)
if "hit_strength_sorted" in globals():
    hs = np.asarray(hit_strength_sorted, dtype=float)
elif "hit_strength" in globals() and "order_strength" in globals():
    hs = np.asarray(hit_strength, dtype=float)[np.asarray(order_strength)]
else:
    hs = None

if hs is not None:
    assert hs.size == E
    if not np.all(np.diff(hs) >= -1e-12):
        raise ValueError("Hit strengths are not sorted increasing.")

# -----------------------
# computations
# -----------------------
# SPEED
speed_grand_avg = np.nanmean(speed_tensor, axis=(0, 1))  # (T,)
speed_neff = np.sum(np.isfinite(speed_tensor), axis=(0, 1)).clip(min=1)
speed_grand_sem = np.nanstd(speed_tensor, axis=(0, 1)) / np.sqrt(speed_neff)
speed_mat_mean = np.nanmean(speed_tensor, axis=0)        # (E, T)

# SLEEP (mean of 0/1 → probability)
sleep_grand_avg = np.nanmean(sleep_tensor, axis=(0, 1))  # (T,)
sleep_neff = np.sum(np.isfinite(sleep_tensor), axis=(0, 1)).clip(min=1)
sleep_grand_sem = np.nanstd(sleep_tensor, axis=(0, 1)) / np.sqrt(sleep_neff)
sleep_mat_mean = np.nanmean(sleep_tensor, axis=0)        # (E, T)

# -----------------------
# plots
# -----------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
(ax1, ax2), (ax3, ax4) = axes

# (A) Grand-average SPEED trace
ax1.plot(t, speed_grand_avg, linewidth=1)
ax1.fill_between(
    t,
    speed_grand_avg - speed_grand_sem,
    speed_grand_avg + speed_grand_sem,
    alpha=0.3,
)
ax1.axvline(0, linewidth=1)
ax1.set_title(f"Grand average SPEED (mean±SEM)\n{A} ants × {E} hits")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Speed (px/s)")

# (B) SPEED hit×time matrix (mean over ants)
im2 = ax2.imshow(
    speed_mat_mean,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    extent=[t[0], t[-1], 0, E],
    vmax=np.nanpercentile(speed_mat_mean, 99),
    vmin=np.nanpercentile(speed_mat_mean, 1))
ax2.axvline(0, linewidth=1)
ax2.set_title("SPEED: mean over ants (hit × time)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Hit index (sorted by strength)")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Speed (px/s)")

# (C) Grand-average SLEEP probability trace
ax3.plot(t, sleep_grand_avg, linewidth=1)
ax3.fill_between(
    t,
    sleep_grand_avg - sleep_grand_sem,
    sleep_grand_avg + sleep_grand_sem,
    alpha=0.3,
)
ax3.axvline(0, linewidth=1)
ax3.set_ylim(-0.05, 0.2)
ax3.set_title(f"Grand average SLEEP probability (mean±SEM)\n{A} ants × {E} hits")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("P(sleep)")

# (D) SLEEP hit×time matrix (mean over ants)
im4 = ax4.imshow(
    sleep_mat_mean,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    extent=[t[0], t[-1], 0, E],
    vmin=0,
    vmax=0.2,
)
ax4.axvline(0, linewidth=1)
ax4.set_title("SLEEP: mean over ants (hit × time)")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Hit index (sorted by strength)")
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label="P(sleep)")

# Optional annotations to confirm hit ordering
if hs is not None:
    ax2.text(t[0], 0.5, f"low {hs[0]:.1f}%", va="bottom")
    ax2.text(t[0], E - 0.5, f"high {hs[-1]:.1f}%", va="top")
    ax4.text(t[0], 0.5, f"low {hs[0]:.1f}%", va="bottom")
    ax4.text(t[0], E - 0.5, f"high {hs[-1]:.1f}%", va="top")

plt.tight_layout()
plt.show()

# -----------------------
# explicit sanity prints
# -----------------------
print("OK")
print("speed_tensor shape (A,E,T):", speed_tensor.shape)
print("sleep_tensor shape (A,E,T):", sleep_tensor.shape)
if hs is not None:
    print("hit_strength_sorted first/last:", hs[0], hs[-1])
print("speed_mat_mean shape:", speed_mat_mean.shape)
print("sleep_mat_mean shape:", sleep_mat_mean.shape)

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


if first_frame_vec is None or first_speed_vec is None:
    raise RuntimeError("No classified speed trace available to plot.")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(first_frame_vec, first_speed_vec, linewidth=1)

for hf in hit_frames:
    ax.axvline(hf, color="crimson", alpha=0.25, linewidth=0.8)

ax.set_xlabel("Frame (global cam frame)")
ax.set_ylabel("Speed (pix/s)")
ax.set_title(f"Speed time series for first ant (TrackID_{int(first_track_id):04d})")
ax.xaxis.set_major_formatter(FuncFormatter(frame_to_hms))

fig.tight_layout()
plt.show()
# %%
