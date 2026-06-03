# %%
from pathlib import Path
import pandas as pd

# -----------------------
# pick ONE file to test
# -----------------------
pkl_path = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20251117_2_stim/chunk000_left.pkl")   # <-- change this
parquet_path = pkl_path.with_suffix(".parquet")

# -----------------------
# load pickle
# -----------------------
df_pkl = pd.read_pickle(pkl_path)

# -----------------------
# write parquet
# -----------------------
df_pkl.to_parquet(
    parquet_path,
    index=False,
    engine="pyarrow",
    compression="zstd",
)

# -----------------------
# reload parquet
# -----------------------
df_parq = pd.read_parquet(parquet_path)

# -----------------------
# sanity checks
# -----------------------
assert df_pkl.shape == df_parq.shape, "Shape mismatch"
assert list(df_pkl.columns) == list(df_parq.columns), "Column mismatch"
assert df_pkl.dtypes.equals(df_parq.dtypes), "Dtype mismatch"

# For numeric tracking data this should pass; if not, investigate NaNs / floats
assert df_pkl.equals(df_parq), "Data mismatch"

print("✔ Parquet conversion test passed:", parquet_path)

# %%
from pathlib import Path
import pandas as pd

# -----------------------
# paths
# -----------------------
pkl_dir = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20251117_2_stim")        # directory containing *.pkl
parquet_dir = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20251117_2_stim") # output directory
parquet_dir.mkdir(parents=True, exist_ok=True)

# -----------------------
# convert
# -----------------------
pkls = sorted(pkl_dir.glob("*left.pkl"))
if not pkls:
    raise RuntimeError(f"No .pkl files found in {pkl_dir}")

for pkl in pkls:
    print(f"Converting {pkl.name} ...", flush=True)

    df = pd.read_pickle(pkl)   # loads one chunk only
    out = parquet_dir / (pkl.stem + ".parquet")

    df.to_parquet(
        out,
        index=False,
        engine="pyarrow",      # recommended
        compression="zstd"     # fast + compact; use "snappy" if you prefer
    )

print(f"\nConverted {len(pkls)} files → {parquet_dir.resolve()}")


# %% Build per-TrackID stitched files across chunks
# Assumes ALL valid TrackIDs appear in the FIRST parquet file

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------
# config
# -----------------------
parquet_dir = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20251117_2_stim")
pattern = "*left.parquet"
out_dir = parquet_dir / "per_track"
out_dir.mkdir(parents=True, exist_ok=True)

columns = ["Frame", "TrackID", "X", "Y", "Bodypoint"]  # must include Frame + TrackID

# -----------------------
# discover files
# -----------------------
files = sorted(parquet_dir.glob(pattern))
if not files:
    raise RuntimeError(f"No parquet files found in {parquet_dir} matching {pattern}")

print(f"Found {len(files)} parquet files.")
print(f"Output directory: {out_dir.resolve()}")

# -----------------------
# initialize TrackIDs from FIRST file
# -----------------------
first = pd.read_parquet(files[0], columns=["TrackID"], engine="pyarrow")
first["TrackID"] = pd.to_numeric(first["TrackID"], errors="coerce")
track_ids = sorted(first.dropna()["TrackID"].astype(int).unique().tolist())

print(f"Initialized {len(track_ids)} TrackIDs from first file.")

# pre-allocate buffers
parts_by_track: dict[int, list[pd.DataFrame]] = {tid: [] for tid in track_ids}

# -----------------------
# stitch chunks
# -----------------------
frame_offset = 0
pbar = tqdm(files, desc="Chunks", unit="file", dynamic_ncols=True)

for fp in pbar:
    # --- determine chunk length ---
    frame_col = pd.read_parquet(fp, columns=["Frame"], engine="pyarrow")
    if frame_col.empty:
        pbar.set_postfix_str(f"offset={frame_offset} (empty)")
        continue

    frame_col["Frame"] = pd.to_numeric(frame_col["Frame"], errors="coerce")
    chunk_max_frame = int(frame_col["Frame"].max())
    chunk_len = chunk_max_frame + 1

    # --- load chunk data ---
    df = pd.read_parquet(fp, columns=columns, engine="pyarrow")
    if df.empty:
        frame_offset += chunk_len
        pbar.set_postfix_str(f"offset={frame_offset}")
        continue

    df = df.copy()
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df["TrackID"] = pd.to_numeric(df["TrackID"], errors="coerce")
    df = df.dropna(subset=["Frame", "TrackID"])
    df["Frame"] = df["Frame"].astype(np.int64) + frame_offset
    df["TrackID"] = df["TrackID"].astype(int)
    df["source_file"] = fp.name  # optional provenance

    # --- append rows to known TrackIDs only ---
    for tid, g in df.groupby("TrackID", sort=False):
        if tid in parts_by_track:
            parts_by_track[tid].append(g)

    frame_offset += chunk_len
    pbar.set_postfix_str(f"offset={frame_offset}")

pbar.close()

# -----------------------
# write outputs
# -----------------------
saved = 0
empty = 0

for tid in tqdm(track_ids, desc="Tracks", unit="track", dynamic_ncols=True):
    parts = parts_by_track[tid]
    if not parts:
        empty += 1
        continue

    out = pd.concat(parts, ignore_index=True)
    out_path = out_dir / f"TrackID_{tid:04d}_all_chunks.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
    saved += 1

print(
    f"Done.\n"
    f"  TrackIDs total:      {len(track_ids)}\n"
    f"  Files written:       {saved}\n"
    f"  Empty tracks:        {empty}\n"
    f"  Final frame offset:  {frame_offset}\n"
    f"  Output dir:          {out_dir.resolve()}"
)

# %%
