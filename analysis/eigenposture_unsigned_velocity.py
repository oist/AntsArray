"""Cache unsigned clip peaks from the original motion samples, before averaging.

Run once with --pose-cache, --source (published eigenposture folder), and --output.
The interactive task-state script then needs only this small NPZ and its other
published inputs. Reuses the original camera, body-axis and speed QC.
"""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from analysis.eigenposture_features import DAY1_FRAME, windows


def unsigned_peaks(velocity):
    """Per clip, maximum absolute component among jointly valid motion samples."""
    velocity = np.asarray(velocity, dtype=float).copy()
    valid = np.isfinite(velocity).all(axis=-1)
    velocity[~valid] = np.nan
    counts = valid.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peaks = np.nanmax(np.abs(velocity), axis=1)
    peaks[counts < 8] = np.nan
    return peaks.astype(np.float32), counts


def extract(pose_cache, source, output):
    coverage = pd.read_csv(source / "all_ant_coverage.csv")
    with np.load(source / "full_rank" / "eigenposture_measurements.npz") as z:
        np.testing.assert_array_equal(z["ants"], coverage.ant)
        signed_means = z["minute"][:, :, :2]
    results, sample_counts, provenance = {}, {}, []
    for path in sorted(pose_cache.glob("*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())["signature"]
        assert meta["feature_version"] == "posture_velocity_v2"
        task = meta["task"]
        track = Path(task["source"]["path"])
        stat = track.stat()
        if (stat.st_size, stat.st_mtime_ns) != (
            task["source"]["size"],
            task["source"]["mtime_ns"],
        ):
            raise ValueError(f"Changed tracking source: {track}")
        ant = task["ant"]
        assert ant not in results
        with np.load(path) as z:
            np.testing.assert_array_equal(
                (z["frames"] - DAY1_FRAME) // (60 * 24), np.arange(2880)
            )
            _, velocity, _, _ = windows(z, scale=1.0)
        peaks, counts = unsigned_peaks(velocity)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(velocity, axis=1)
        means[counts < 8] = np.nan
        row = np.flatnonzero(coverage.ant.eq(ant))
        assert len(row) == 1
        np.testing.assert_allclose(
            means, signed_means[row[0]], rtol=1e-5, atol=1e-6, equal_nan=True
        )
        assert np.all(peaks[np.isfinite(peaks)] >= 0)
        results[ant], sample_counts[ant] = peaks, counts
        provenance.append(
            dict(
                ant=ant,
                cache=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                tracking_source=task["source"],
            )
        )
        print(f"{len(results)}/{len(coverage)} {ant}", flush=True)
    assert set(results) == set(coverage.ant)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ants=coverage.ant.to_numpy(str),
        unsigned_max=np.stack([results[a] for a in coverage.ant]),
        valid_samples=np.stack([sample_counts[a] for a in coverage.ant]),
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                statistic="max(abs(component)) over valid within-clip motion samples",
                units="mm/s",
                sampling="one 2.5-second clip/minute, 28 velocity samples/clip",
                minimum_valid_samples=8,
                sources=provenance,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-cache", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.pose_cache, args.source, args.output)
