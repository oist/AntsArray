"""Cache five-minute counts of new pair-contact bouts for the posture analysis."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def count_onsets(bouts, ants, coverage, start_frame, n_bins, bin_frames, fps):
    """Count each undirected onset for both ants, on half-open time intervals.

    Zero means no detected new bout during covered time. Missing interaction
    coverage remains NaN. Left-censored bouts are not counted as new onsets.
    """
    assert start_frame % fps == 0 and bin_frames % fps == 0
    result = np.zeros((len(ants), n_bins), dtype=float)
    lookup = {(r.side, r.track_id): i for i, r in enumerate(ants.itertuples())}
    selected = bouts[
        bouts.is_new_onset
        & bouts.start_frame.ge(start_frame)
        & bouts.start_frame.lt(start_frame + n_bins * bin_frames)
    ]
    for row in selected.itertuples():
        if row.ant_a == row.ant_b:
            raise ValueError("Self-contact in pair-contact cache")
        column = (row.start_frame - start_frame) // bin_frames
        for ant in (row.ant_a, row.ant_b):
            index = lookup.get((row.side, ant))
            if index is not None:
                result[index, column] += 1
    first_second = start_frame // fps
    seconds_per_bin = bin_frames // fps
    for side, observed in coverage.items():
        observed = observed[first_second:first_second + n_bins * seconds_per_bin]
        if len(observed) != n_bins * seconds_per_bin:
            raise ValueError("Interaction coverage does not span the analysis window")
        valid = observed.reshape(n_bins, seconds_per_bin).all(axis=1)
        result[np.ix_(ants.side.eq(side), ~valid)] = np.nan
    return result


def extract(block, source, output):
    # This loader validates every published chunk/track and the cached-bout key.
    from analysis.behavior_landscape_features import load_contacts

    info = dict(start_time="2026-07-24 10:00:00", frame_start=41520,
                frame_stop=4188720, fps=24., context_settings=dict(fps=24.))
    ants = pd.read_csv(source / "all_ant_coverage.csv")
    bouts, coverage, sources, run_id = load_contacts(block, info)
    counts = count_onsets(bouts, ants, coverage, 41520, 576, 7200, 24)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ants=ants.ant.to_numpy(str), counts=counts,
                        start_frame=41520, bin_frames=7200, fps=24)
    cache = sources[-1]
    output.with_suffix(".json").write_text(json.dumps(dict(
        definition="New undirected pair-contact bouts per ant per five minutes; each onset counts for both ants",
        geometry="all skeleton segments and nodes within 0.1 mm",
        merge_gap_seconds=2, minimum_detection_frames=1,
        exclude_left_censored_onsets=True,
        exposure="Full interaction recording, not only sampled posture clips; detection count, not exposure-corrected rate",
        start_time=info["start_time"], fps=24, start_frame=41520, bin_frames=7200,
        run_id=run_id, cache=str(cache),
        cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        validated_sources=[dict(path=str(p), size=p.stat().st_size,
                                mtime_ns=p.stat().st_mtime_ns) for p in sources],
        missing_bins=int(np.isnan(counts).sum()),
        total_ant_onsets=int(np.nansum(counts)),
    ), indent=2) + "\n")
    print(f"Interaction counts: {counts.shape}, {np.isnan(counts).sum()} missing bins; saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.block, args.source, args.output)
