"""Single-ant identity adapter for the production colony block stitchers."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tracking.colony.block_caches import cache_segments, merge_dense
from tracking.colony.combine_blocks import fingerprint, check_fingerprint
from tracking.stitch_tracks import concatenate_shifted_parquets
from tracking.single_ant.full_cup_tracking import manifest, root, save_parquet
from tracking.single_ant.manual_cup_tracking import write_json


def offsets(blocks):
    """The colony rule: elapsed timestamp * FPS, with genuine unknown gaps."""
    if len({b["fps"] for b in blocks}) != 1:
        raise ValueError("Cannot stitch different frame rates without resampling")
    origin = datetime.fromisoformat(blocks[0]["start_datetime"])
    dated = len({Path(b["source_block"]).parent.name for b in blocks}) > 1
    previous = 0
    result = []
    for b in blocks:
        offset = round((datetime.fromisoformat(b["start_datetime"])-origin).total_seconds()*b["fps"])
        if offset < previous:
            raise ValueError("Overlapping or reversed recording blocks")
        source = Path(b["source_block"])
        label = source.parent.name+"_"+source.name if dated else source.name
        result.append(dict(block=label, source_block=str(source), start_datetime=b["start_datetime"], frame_offset=offset,
                           num_frames=b["n_frames"], gap_before_frames=offset-previous))
        previous = offset+b["n_frames"]
    if len({s["block"] for s in result}) != len(result):
        raise ValueError("Ambiguous recording block labels")
    return result, previous


def prepare(block_manifests, output, published_output):
    loaded = sorted(((p, manifest(p)) for p in block_manifests), key=lambda pair: datetime.fromisoformat(pair[1]["start_datetime"]))
    block_manifests, blocks = zip(*loaded)
    if len({b["source_block"] for b in blocks}) != len(blocks):
        raise ValueError("The same recording block was supplied more than once")
    if any(b["cups"] != blocks[0]["cups"] for b in blocks[1:]):
        raise ValueError("Cup labels/geometry changed; cannot assume the same cup identities")
    segments, total = offsets(blocks)
    output = Path(output)
    pairs, residents, changes = {}, {}, []
    for b, segment in zip(blocks, segments):
        q = pd.read_csv(root(b)/"cup_identities.csv")
        for row in q.loc[q.Eligible].itertuples(index=False):
            cid, tag = int(row.CupID), int(row.ArucoID)
            if cid in residents and residents[cid] != tag:
                changes.append(dict(block=segment["block"], cup=cid, previous_ID=residents[cid], new_ID=tag))
            residents[cid] = tag
            name = f"TrackID_{row.CupID:04d}_all_single.parquet"
            track = root(b)/"per_track"/name
            speed = root(b)/"analysis/speed_vectors/per_track"/Path(name).stem
            pf = pq.ParquetFile(track)
            if int(pf.schema_arrow.metadata[b"num_frames"]) != b["n_frames"]:
                raise ValueError(f"Incorrect source recording span: {track}")
            # Every source row must retain the dedicated ID represented by its
            # registry entry. Statistics avoid decoding the complete pose table.
            col = pf.schema_arrow.get_field_index("ArucoID")
            for i in range(pf.num_row_groups):
                stats = pf.metadata.row_group(i).column(col).statistics
                if stats is None or not stats.has_min_max or stats.min != row.ArucoID or stats.max != row.ArucoID:
                    raise ValueError(f"Ambiguous source tag identity: {track}")
            pairs.setdefault((int(row.CupID), int(row.ArucoID)), []).append(dict(
                track=fingerprint(track), block=segment["block"], frame_offset=segment["frame_offset"],
                num_frames=segment["num_frames"], caches={"speed_vectors": dict(
                    metadata=fingerprint(speed/"speed_metadata.json"),
                    files=[fingerprint(speed/n) for n in ["speed_mm_s.npy", "speed_unclipped_mm_s.npy", "speed_metadata.json"]])}))
    ants, cups, identities = [], [], []
    used, extra = set(), max(c["cup_id"] for c in blocks[0]["cups"])+1
    for (cid, tag), sources in sorted(pairs.items(), key=lambda p: (p[0][0], p[1][0]["frame_offset"])):
        tid = cid if cid not in used else extra
        if tid == extra:
            extra += 1
        used.add(cid)
        cup = dict(next(c for c in blocks[0]["cups"] if c["cup_id"] == cid), track_id=tid)
        if sum(k[0] == cid for k in pairs) > 1:
            cup["name"] += f"_ID{tag}"
        name = f"TrackID_{tid:04d}_all_single.parquet"
        ants.append(dict(name=name, cup_id=cid, raw_id=tag, track_id=tid, side="single", sources=sources))
        cups.append(cup)
        identities.append(dict(CupID=cid, TrackID=tid, Cup=cup["name"], ArucoID=tag, Eligible=True))
    m = dict(protocol="manual_cups_colony_block_stitch_v1", output=str(output), cups=cups,
             physical_cups=blocks[0]["cups"], fps=blocks[0]["fps"], n_frames=total,
             recorded_frames=sum(b["n_frames"] for b in blocks), recording_segments=segments,
             start_datetime=blocks[0]["start_datetime"], tracking=blocks[0]["tracking"],
             title=(segments[0]["block"]+"–"+segments[-1]["block"] if "_" in segments[0]["block"] else
                    Path(blocks[0]["source_block"]).parent.name+" "+segments[0]["block"]+"–"+segments[-1]["block"]),
             preserve_block_speeds=True, permanent_censoring=False, ants=ants,
             identity_policy="cup plus raw ArUco ID; changed IDs form separate ant records",
             source_manifests=[fingerprint(p) for p in block_manifests], published_output=str(published_output))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(identities).to_csv(output/"cup_identities.csv", index=False)
    write_json(output/"identity_changes.json", changes)
    for name in ["manual_regions.csv", "panorama_metadata.json"]:
        shutil.copy2(root(blocks[0])/name, output/name)
    for name in ["cup_order.csv", "cup_name_mapping.csv", "cup_order_source.json"]:
        if (root(blocks[0])/name).is_file():
            shutil.copy2(root(blocks[0])/name, output/name)
    if (root(blocks[0])/"panorama.png").is_file():
        shutil.copy2(root(blocks[0])/"panorama.png", output/"panorama.png")
    write_json(output/"manifest.json", m)
    return output/"manifest.json"


def worker(m, index):
    if index >= len(m["ants"]):
        return
    ant = m["ants"][index]
    for source in ant["sources"]:
        for record in [source["track"], *source["caches"]["speed_vectors"]["files"]]:
            check_fingerprint(record)
    path = root(m)/"per_track"/ant["name"]
    result = concatenate_shifted_parquets(
        [(Path(s["track"]["path"]), s["frame_offset"], s["block"]) for s in ant["sources"]], path,
        metadata=dict(num_frames=m["n_frames"], fps=m["fps"], start_datetime=m["start_datetime"]),
        expected_track_id=ant["cup_id"], keep_block_frame=True,
        frame_limits={s["block"]: s["num_frames"] for s in ant["sources"]})
    if ant["track_id"] != ant["cup_id"]:
        from tracking.core.track_parquet import AntParquetWriter
        pf = pq.ParquetFile(path)
        with AntParquetWriter(path, pf.schema_arrow) as writer:
            for batch in pf.iter_batches():
                table = pa.Table.from_batches([batch])
                table = table.set_column(table.schema.get_field_index("TrackID"), "TrackID",
                                         pa.array(np.full(len(table), ant["track_id"], dtype=np.int64)))
                writer.write_table(table)
    group = dict(num_frames=m["n_frames"], fps=m["fps"], start_datetime=m["start_datetime"])
    cache_root = root(m)/"analysis"
    merge_dense(cache_segments(ant, "speed_vectors"), "speed_vectors", ant, group, cache_root, cache_root)
    directory = cache_root/"speed_vectors/per_track"/Path(ant["name"]).stem
    metadata = json.loads((directory/"speed_metadata.json").read_text())
    metadata.update(track_path=str(path), aruco_id=ant["raw_id"], cup_id=ant["cup_id"])
    write_json(directory/"speed_metadata.json", metadata)
    # The colony dense-cache combiner handles the standard vector. Preserve
    # its optional unclipped companion with exactly the same offsets and gaps.
    raw = np.lib.format.open_memmap(directory/"speed_unclipped_mm_s.npy", mode="w+", dtype="float32", shape=(m["n_frames"],))
    raw[:] = np.nan
    for s in ant["sources"]:
        source = Path(s["caches"]["speed_vectors"]["metadata"]["path"]).parent/"speed_unclipped_mm_s.npy"
        values = np.load(source, mmap_mode="r")
        if values.shape != (s["num_frames"],):
            raise ValueError(f"Unclipped speed span mismatch: {source}")
        start = s["frame_offset"]
        raw[start:start+len(values)] = values
    raw.flush(); del raw
    write_json(root(m)/"combination_checks"/(Path(ant["name"]).stem+".json"), result)
    from tracking.single_ant.full_cup_report import diagnose_cup
    diagnose_cup(m, index)
