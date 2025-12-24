
#%%

%matplotlib qt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_event_trig_avg(sig, event_inds, backlag, forwardlag):
    """
    Calculate the event-triggered average.

    Parameters:
    - sig (numpy.ndarray): Input signal.
    - event_inds (numpy.ndarray): Indices of events.
    - backlag (int): Backward time lag.
    - forwardlag (int): Forward time lag.

    Returns:
    - ev_avg (numpy.ndarray): Event-triggered average.
    - ev_mat (numpy.ndarray): Event-triggered matrix.

    """
    event_inds = np.round(event_inds).astype(int) 
    if sig.ndim==1:
        sig=np.expand_dims(sig,0)
        
    min_nevents = 1  # minimum number of events where we will even compute a triggered avg

    orig_size = sig.shape

    lags = np.arange(-backlag, forwardlag + 1)

    # get rid of events that happen within the lag-range of the end points
    bad_ids = np.where(event_inds <= backlag)[0]
    if len(bad_ids) > 0:
        print(f'Dropping {len(bad_ids)} early events')
        event_inds = np.delete(event_inds, bad_ids)

    bad_ids = np.where(event_inds >= (orig_size[1] - forwardlag))[0]
    if len(bad_ids) > 0:
        print(f'Dropping {len(bad_ids)} late events')
        event_inds = np.delete(event_inds, bad_ids)

    n_events = len(event_inds)

    # check that we have at least the minimum number of events to work with
    if n_events < min_nevents:
        ev_avg = np.full((orig_size[0], len(lags)), np.nan)
        ev_mat = np.nan
        return ev_avg, ev_mat

    ev_avg = np.zeros((orig_size[0], len(lags)))
    ev_mat = np.zeros((n_events, orig_size[0], len(lags)))

    for i in range(n_events):
        cur_ids = np.arange(event_inds[i] - backlag, event_inds[i] + forwardlag + 1)
        temp_sig = sig[:, cur_ids]
        ev_avg += temp_sig
        ev_mat[i,:, :] = temp_sig

    ev_avg /= n_events

    return np.squeeze(ev_avg), np.squeeze(ev_mat)

def plot_ant_posture(df, track_id, frame, color='k', annotate=True):
    """
    Plot the bodypoints of a given ant (TrackID) at a given frame, centered on bodypoint 0.

    Parameters:
    - df: DataFrame with columns ["Frame", "TrackID", "Bodypoint", "X", "Y"]
    - track_id: integer TrackID
    - frame: integer frame number
    - color: matplotlib color string for bodypoints and labels
    - annotate: whether to label bodypoint indices
    """
    # Extract data for the specific ant and frame
    d = df[(df["TrackID"] == track_id) & (df["Frame"] == frame)]

    if d.empty:
        print(f"No data for TrackID={track_id} at Frame={frame}")
        return

    if not set(["X", "Y", "Bodypoint"]).issubset(d.columns):
        raise ValueError("Missing required columns.")

    # Ensure bodypoints are sorted (optional but consistent)
    d = d.sort_values("Bodypoint")

    # Center on bodypoint 0
    ref = d[d.Bodypoint == 0][["X", "Y"]].values
    if ref.size == 0:
        print("Bodypoint 0 not found for centering.")
        return
    dx, dy = ref[0]
    x = d["X"].values - dx
    y = d["Y"].values - dy

    plt.figure()
    plt.scatter(x, y, s=30, color=color)

    if annotate:
        for i, (xi, yi) in enumerate(zip(x, y)):
            bp = d.iloc[i]["Bodypoint"]
            plt.text(xi, yi, str(bp), fontsize=10,
                     ha='center', va='center',
                     bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec=color, lw=1))

    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.gca().set_aspect('equal')
    plt.title(f"TrackID {track_id}, Frame {frame} (centered on bp0)")
    plt.xlabel("X (centered)")
    plt.ylabel("Y (centered)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Load data
df = pd.read_pickle("/home/sam-reiter/bucket/ReiterU/Ants/basler/20251020_1_30min_vibration/chunk000/chunk000_right.pkl")

# Check for required columns
REQUIRED = {"Frame", "TrackID", "Bodypoint", "X", "Y"}
assert REQUIRED.issubset(df.columns), f"DataFrame missing required columns: {REQUIRED - set(df.columns)}"

# Filter only bodypoint == 0
df = df[df["Bodypoint"] == 0].copy()

# Compute frame-to-frame displacement for consecutive detections per track
df = df.sort_values(["TrackID", "Frame"])
df["FrameDiff"] = df.groupby("TrackID")["Frame"].diff()
df["dX"] = df.groupby("TrackID")["X"].diff()
df["dY"] = df.groupby("TrackID")["Y"].diff()

# Speed = displacement per frame (set to NaN if frames not consecutive)
df["Speed"] = np.where(
    df["FrameDiff"] == 1,
    np.sqrt(df["dX"]**2 + df["dY"]**2),
    np.nan
)

# Mean speed per frame across all tracks
speed_over_time = df.groupby("Frame")["Speed"].mean()

# Apply Gaussian smoothing
sigma = 5
smoothed_speed = gaussian_filter1d(speed_over_time.values, sigma=sigma)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(speed_over_time.index, smoothed_speed, label="Smoothed speed (Bodypoint 0)")
plt.xlabel("Frame")
plt.ylabel("Mean speed (pixels/frame)")
plt.legend()
plt.tight_layout()
plt.show()


# Plot
plt.figure(figsize=(8, 4))
#plt.plot(speed_over_time.index, speed_over_time.values, color='gray', alpha=0.4, label='Raw mean speed')
plt.plot(speed_over_time.index, smoothed_speed, color='steelblue', lw=2, label=f'Gaussian smoothed (σ={sigma})')
plt.xlabel("Frame")
plt.ylabel("Mean Speed (px/frame)")
plt.title("Mean Speed Over Time (averaged over visible TrackIDs)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

#%%
stim_times= [28020,35340,42660,49980,57300,64620]
backlag=500
forlag=2000
ev_avg,ev_mat=get_event_trig_avg(smoothed_speed, stim_times, backlag, forlag)

plt.figure(figsize=(8,4))
plt.plot(np.arange(len(ev_avg))/25-backlag/25, ev_avg, color='darkorange', lw=2)
plt.xlabel('Time relative to stimulus (s)')

plt.figure(figsize=(8,4))
plt.imshow(ev_mat, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Mean Speed (px/frame)')
plt.xlabel('Frames relative to stimulus')






#%%

def align_to_47_axis(row):
    """
    1. Translate every bodypoint so the midpoint of segment 4–7 is at (0, 0).
    2. Rotate so the vector 4→7 lies on +X.
    3. If the abdomen (bp1) ends up above the axis, flip 180° so antennae (5-6, 8-9)
       are always in +Y.
    Returns a flat vector [x0',y0',x1',y1',…,xN',yN'] (shape-aligned coordinates).
    """
    x = row["X"].values
    y = row["Y"].values

    # ── 1. translate to midpoint of bp4–bp7 ─────────────────────────────────
    mid_x = 0.5 * (x[4] + x[7])
    mid_y = 0.5 * (y[4] + y[7])
    x = x - mid_x
    y = y - mid_y

    # ── 2. rotation to send 4→7 onto +X ─────────────────────────────────────
    dx, dy = x[7] - x[4], y[7] - y[4]
    theta  = -np.arctan2(dy, dx)
    c, s   = np.cos(theta), np.sin(theta)
    R      = np.array([[c, -s],
                       [s,  c]])
    xy_rot = R @ np.vstack([x, y])           # shape (2, N)

    # ── 3. flip if abdomen (bp1) ended up in +Y (we want antennae +Y) ──────
    if xy_rot[1, 1] > 0:                     # bp1 is index 1
        xy_rot *= -1

    return xy_rot.T.flatten()


HEAD_BP, THORAX_BP = 1, 0                       # unchanged
AXIS_P0, AXIS_P1   = 4, 7                       # segment that defines +X
ANTENNAE_BPS       = [5, 6, 8, 9]               # tips & bases

STATIONARY_THR_MM_S = 1.0

# wide table of all bodypoints
xy = (df.pivot_table(index=["TrackID", "Frame"],
                     columns="Bodypoint", values=["X", "Y"])
        .dropna()) 

# apply alignment
aligned = xy.apply(align_to_47_axis, axis=1, result_type="expand")
aligned.columns = [f"{axis}{bp}"
                   for bp in range(len(aligned.columns)//2)
                   for axis in ("x", "y")]

# merge with speed table (‘pos’ from earlier)
analysis = pos[["speed_mm_s"]].join(aligned, how="inner")

# split behavioural states (optional)
stationary = analysis.query("speed_mm_s < @STATIONARY_THR_MM_S")
moving     = analysis.query("speed_mm_s >= @STATIONARY_THR_MM_S")



#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def plot_combined_antennae_hist(df_cond,
                                 title,
                                 bps=(6, 9),
                                 xlim=(-300, 300), ylim=(-200, 400),
                                 bins=200, sigma=2.0,
                                 cmap='viridis'):
    """
    Plot a single smoothed 2-D histogram combining multiple antenna-tip bodypoints.

    Parameters
    ----------
    df_cond : DataFrame with columns 'x6','y6','x9','y9' (or other bps).
    title   : str – figure title.
    bps     : tuple[int] – which bodypoints to include (default: 6, 9).
    xlim, ylim : tuple – axis limits in aligned pixel space.
    bins    : int or (nx, ny) – histogram grid resolution.
    sigma   : float – Gaussian smoothing σ in bin units.
    cmap    : str – matplotlib colormap.
    """
    # Shared bin edges
    x_edges = np.linspace(*xlim, bins + 1)
    y_edges = np.linspace(*ylim, bins + 1)

    # Combine all bp positions into 1 vector
    all_x = np.concatenate([df_cond[f"x{bp}"].values for bp in bps])
    all_y = np.concatenate([df_cond[f"y{bp}"].values for bp in bps])

    # Histogram and smoothing
    H, _, _ = np.histogram2d(all_y, all_x, bins=[y_edges, x_edges])  # rows then cols
    H_smooth = gaussian_filter(H, sigma=sigma, mode='constant')

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(H_smooth,
                   extent=(*xlim, *ylim),
                   origin='lower',
                   aspect='equal',
                   cmap=cmap)

    ax.axhline(0, lw=0.8, c='k')
    ax.axvline(0, lw=0.8, c='k')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Aligned X (px)")
    ax.set_ylabel("Aligned Y (px)")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)

    cbar = fig.colorbar(im, ax=ax, label="Count (smoothed)")
    plt.tight_layout()
    plt.show()

plot_combined_antennae_hist(
    stationary,
    "Smoothed density – antennae tips (bp6 + bp9), stationary",
    bins=180,
    sigma=1.5
)

plot_combined_antennae_hist(
    moving,
    "Smoothed density – antennae tips (bp6 + bp9), moving",
    bins=180,
    sigma=1.5
)

# %%
