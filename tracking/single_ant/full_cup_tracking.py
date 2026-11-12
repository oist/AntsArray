"""Full manual-cup tracking with the unchanged colony tracking engine.

Each cup retains its dedicated raw ArUco ID. Other IDs compete for skeletons
under the colony rules; they never censor a cup. Tracks remain within the
manual cup, and repeated raw IDs in different cups remain separate ants.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tracking.single_ant.manual_cup_tracking import (
    Settings, frame_evidence, load_cups, read_window, sha256, tag_has_skeleton,
    validate_timing, write_json,
)
from tracking.single_ant.cup_order import row_major

PROTOCOL = "manual_cup_native_colony_v2"
MM_PER_PX = .016
TRACKING = dict(max_distance=90., lost_track_max_frames=120,
                lost_track_max_distance=90., lost_track_aruco_max_distance=90.,
                aruco_sleap_max_distance=90.)
# The colony mapper retains finite coordinates without a node-score or
# skeleton-completeness threshold. Keep its candidates, including partial poses.
MAPPING = Settings(min_anchor_score=0., min_pose_nodes=1)
TRACK_SCHEMA = pa.schema([
    ("Frame", pa.int64()), ("TrackID", pa.int64()), ("Bodypoint", pa.int64()),
    *[(name, pa.float64()) for name in ["X", "Y", "TrackX", "TrackY", "ArucoX", "ArucoY", "SleapAnchorX", "SleapAnchorY"]],
    *[(name, pa.int64()) for name in ["CameraID", "ArucoCam", "SleapCam", "CupID", "ArucoID"]],
    ("PositionSource", pa.string()),
])


def manifest(path):
    return json.loads(Path(path).read_text())


def root(m):
    return Path(m["output"])


def chunk_folder(m, index):
    return root(m) / "mapped" / f"chunk{index:03d}"


def save_parquet(path, data, metadata=None, schema=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(data, preserve_index=False, schema=schema)
    if metadata:
        md = dict(table.schema.metadata or {})
        md.update({str(k).encode(): str(v).encode() for k, v in metadata.items()})
        table = table.replace_schema_metadata(md)
    temp = path.with_suffix(".partial.parquet")
    pq.write_table(table, temp, compression="zstd", row_group_size=100_000)
    temp.replace(path)


def resident_registry(path, cups):
    """Read independently documented resident IDs; never infer an ID for an untagged cup."""
    table = pd.read_csv(path, sep="\t")
    if not {"cup", "tag_id"}.issubset(table.columns):
        raise ValueError("Resident registry requires cup and tag_id columns")
    ids = pd.to_numeric(table.cup, errors="raise")
    if not np.isfinite(ids).all() or not ids.eq(np.floor(ids)).all() or ids.duplicated().any():
        raise ValueError("Resident registry cup numbers must be unique integers")
    if set(ids.astype(int)) != {c["cup_id"] for c in cups}:
        raise ValueError("Resident registry must describe every row-major cup exactly once")
    tags = pd.to_numeric(table.tag_id, errors="raise")
    valid = tags.dropna()
    if not np.isfinite(valid).all() or not valid.ge(0).all() or not valid.eq(np.floor(valid)).all():
        raise ValueError("Resident registry tag IDs must be nonnegative integers or blank")
    return {str(int(cid)): int(tag) if pd.notna(tag) else None for cid, tag in zip(ids, tags)}


def prepare(args):
    output = args.output
    if (output / "manifest.json").exists():
        raise FileExistsError(output / "manifest.json")
    m = manifest(args.detections)
    validate_timing(m)
    meta = manifest(args.metadata)
    if not np.isclose(meta["mm_per_pixel"], 1 / m["board_px_per_mm"]):
        raise ValueError("Manual cups and input calibration have different scales")
    if sha256(args.homography) != m["homography_sha256"]:
        raise ValueError("Input homography changed")
    if Path(meta["homographies"]).parts[-4:] != Path(m["homography_source"]).parts[-4:]:
        raise ValueError("Cup labels use another calibration")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.detections, output / "detections_manifest.json")
    pilot = (pd.read_csv(args.pilot_identities) if args.pilot_identities else
             pd.DataFrame(columns=["CupID", "ArucoID", "Eligible", "SingleIDFraction"]))
    known = {int(r.CupID): int(r.ArucoID) for r in pilot.itertuples(index=False)
             if r.Eligible and r.ArucoID >= 0 and r.SingleIDFraction > .6}
    cups, order = row_major(load_cups(args.regions))
    registry = getattr(args, "resident_registry", None)
    documented = resident_registry(registry, cups) if registry is not None else None
    known = {r["CupID"]: known[r["PreviousCupID"]] for r in order if r["PreviousCupID"] in known}
    # Retain established identities even if a tag later becomes occluded.
    # The other cups can qualify using the complete recording.
    contract = dict(protocol=PROTOCOL, output=str(output), detections=str(output / "detections_manifest.json"),
                    source_detections_manifest=str(args.detections),
                    detections_sha256=sha256(args.detections), cups=cups,
                    cup_order="row major: rows top to bottom; columns left to right",
                    initial_identities={str(k): v for k, v in known.items()},
                    initial_identity_source=str(args.pilot_identities) if args.pilot_identities else None,
                    initial_identity_sha256=sha256(args.pilot_identities) if args.pilot_identities else None,
                    resident_registry_source=str(registry) if registry is not None else None,
                    resident_registry_sha256=sha256(registry) if registry is not None else None,
                    documented_identities=documented,
                    start_datetime=m.get("origin"), regions_source=str(args.regions),
                    regions_sha256=sha256(args.regions), metadata_sha256=sha256(args.metadata),
                    fps=m["fps"], n_frames=m["n_frames"], chunks=m["chunks"],
                    synchronization=m["synchronization"], source_block=m["block"],
                    tracking=TRACKING, mapping=asdict(MAPPING), output_mm_per_px=MM_PER_PX,
                    min_competitor_frame_fraction=.25,
                    tracking_engine="tracking.core.tracking_utils.get_complete_tracks",
                    native_engine_sha256=sha256(Path(__file__).parents[1] / "core" / "tracking_utils.py"),
                    script_sha256=sha256(__file__), permanent_censoring=False,
                    chunk_policy="Same independent chunks and native tracker as colony pipeline; true global frame offsets",
                    identity_policy="Keep initially qualified cup IDs; qualify previously unassigned cups over full recording")
    write_json(output / "manifest.json", contract)
    pd.DataFrame(order).to_csv(output / "cup_order.csv", index=False)
    for source, name in [(args.regions, "manual_regions.csv"), (args.metadata, "panorama_metadata.json"),
                         (args.pilot_identities, "initial_identities.csv"), (registry, "resident_registry.tsv")]:
        if source is not None:
            shutil.copy2(source, output / name)
    for name in ["cup_name_mapping.csv", "cup_order_source.json"]:
        source = args.regions.with_name(name)
        if source.is_file():
            shutil.copy2(source, output / name)
    if args.panorama is not None:
        shutil.copy2(args.panorama, output / "panorama.png")
    print(json.dumps(dict(frames=m["n_frames"], hours=m["n_frames"] / m["fps"] / 3600,
                          cups=len(contract["cups"]), established_identities=len(known)), indent=2))


def map_chunk(m, index):
    output = chunk_folder(m, index)
    done = output / "COMPLETE.json"
    if done.exists():
        return
    source = manifest(m["detections"])
    chunk = m["chunks"][index]
    start, stop = chunk["start"], chunk["stop"]
    a, p, nodes, sources = read_window(source, m["cups"], start, stop, MAPPING)
    output.mkdir(parents=True, exist_ok=True)
    frequency = a.groupby("Instance").Frame.nunique() / (stop - start)
    retained = [int(x) for x in frequency.index[frequency.ge(m["min_competitor_frame_fraction"])]]
    write_json(output / "competitor_ids.json", dict(retained_ids=retained,
               frequency={str(int(k)): float(v) for k, v in frequency.items()}))
    statistics = []
    for cup in m["cups"]:
        cid = cup["cup_id"]
        ca, cp, cn = [d.loc[d.CupID.eq(cid)].copy() for d in (a, p, nodes)]
        target = output / f"cup{cid:03d}"
        for name, d in [("aruco", ca), ("poses", cp), ("nodes", cn)]:
            save_parquet(target / f"{name}.parquet", d)
        support = tag_has_skeleton(ca, cp, MAPPING)
        supported = ca.loc[support].groupby("Instance").Frame.nunique()
        for raw_id, group in ca.groupby("Instance", sort=False):
            counts, _ = frame_evidence(group, start, stop, MAPPING)
            statistics.append(dict(CupID=cid, ArucoID=int(raw_id),
                                   ObservedFrames=int(group.Frame.nunique()),
                                   SingleInstanceFrames=int((counts == 1).sum()),
                                   SupportedFrames=int(supported.get(raw_id, 0)),
                                   MultipleInstanceFrames=int((counts > 1).sum())))
    pd.DataFrame(statistics, columns=["CupID", "ArucoID", "ObservedFrames", "SingleInstanceFrames",
                                      "SupportedFrames", "MultipleInstanceFrames"]).to_csv(output / "id_counts.csv", index=False)
    write_json(done, dict(index=index, frame_start=start, frame_stop=stop,
                         aruco_rows=len(a), pose_instances=len(p), node_rows=len(nodes), sources=sources))


def identify(m):
    parts = []
    for chunk in m["chunks"]:
        folder = chunk_folder(m, chunk["index"])
        if not (folder / "COMPLETE.json").exists():
            raise ValueError(f"Missing mapped chunk: {folder}")
        parts.append(pd.read_csv(folder / "id_counts.csv"))
    counts = pd.concat(parts, ignore_index=True).groupby(["CupID", "ArucoID"], as_index=False).sum(numeric_only=True)
    counts["SingleInstanceFraction"] = counts.SingleInstanceFrames / m["n_frames"]
    counts["SupportedFraction"] = counts.SupportedFrames / m["n_frames"]
    counts.to_csv(root(m) / "full_recording_id_counts.csv", index=False)
    rows = []
    for cup in m["cups"]:
        cid = cup["cup_id"]
        c = counts.loc[counts.CupID.eq(cid)]
        old = m["initial_identities"].get(str(cid))
        source = "initial_10_minute_identity" if old is not None else "full_recording_gt60_percent"
        reason = "dedicated_ID_retained_no_censoring"
        registry = m.get("documented_identities")
        if registry is not None:
            expected = registry[str(cid)]
            chosen = c.loc[c.ArucoID.eq(expected)] if expected is not None else c.iloc[:0]
            # External identity replaces only the discovery threshold. Actual tag/pose
            # support is still required, and all native frame-level guards remain intact.
            supported = len(chosen) and chosen.SupportedFrames.iloc[0] > 0 and chosen.SingleInstanceFrames.iloc[0] > 0
            raw_id = int(expected) if supported else -1
            source = "documented_resident_registry"
            reason = "documented_ID_with_observed_tag_pose_support" if supported else (
                "documented_untagged_cup" if expected is None else "documented_ID_without_tag_pose_support")
            conflicts = c.loc[c.SingleInstanceFraction.gt(.6) & c.SupportedFrames.gt(0)]
            if expected is not None and len(conflicts.loc[~conflicts.ArucoID.eq(expected)]):
                raise ValueError(f"Cup {cid}: high-coverage observed ID conflicts with resident registry")
        elif old is not None:
            chosen = c.loc[c.ArucoID.eq(old)]
            raw_id = int(old)
        else:
            candidates = c.loc[c.SingleInstanceFraction.gt(.6) & c.SupportedFrames.gt(0)]
            chosen = candidates.sort_values(["SupportedFrames", "SingleInstanceFrames", "ArucoID"],
                                             ascending=[False, False, True]).head(1)
            raw_id = int(chosen.ArucoID.iloc[0]) if len(chosen) else -1
        row = dict(CupID=cid, Cup=cup["name"], ArucoID=raw_id, Eligible=raw_id >= 0,
                   AssignmentSource=source if raw_id >= 0 else "unassigned",
                   SingleIDFractionFull=float(chosen.SingleInstanceFraction.iloc[0]) if len(chosen) else 0.,
                   TagAndPoseFractionFull=float(chosen.SupportedFraction.iloc[0]) if len(chosen) else 0.,
                   Reason=reason if registry is not None or raw_id >= 0 else "no_ID_over_60_percent")
        rows.append(row)
    q = pd.DataFrame(rows)
    q.to_csv(root(m) / "cup_identities.csv", index=False)
    write_json(root(m) / "IDENTITIES_COMPLETE.json", dict(eligible=int(q.Eligible.sum()), cups=len(q),
               permanent_censoring=False, frames=m["n_frames"]))
    print(q.to_string(index=False), flush=True)


def native_cup_track(aruco, nodes, *, cup_id, dedicated_id, frame_start, frame_stop,
                     output, tracking=None, competitor_ids=None):
    """Invoke the actual colony engine, then export only this cup's resident."""
    from tracking.core.tracking_utils import get_complete_tracks
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    a, s = aruco.copy(), nodes.copy()
    if competitor_ids is not None:
        a = a.loc[a.Instance.isin(set(competitor_ids) | {int(dedicated_id)})].copy()
    for data in (a, s):
        data["Frame"] = data.Frame.astype(np.int64) - frame_start
        data[["X", "Y"]] = data[["X", "Y"]] / MM_PER_PX
    native = output.with_name(output.stem + "_native_all_ids.parquet")
    get_complete_tracks(native, a, s, num_frames=frame_stop - frame_start,
                        stream_output=True, **(tracking or TRACKING))
    d = pd.read_parquet(native, filters=[("TrackID", "==", int(dedicated_id))])
    d["Frame"] = d.Frame.astype(np.int64) + int(frame_start)
    d["TrackID"] = np.full(len(d), int(cup_id), dtype=np.int64)
    d["CupID"] = np.full(len(d), int(cup_id), dtype=np.int64)
    d["ArucoID"] = np.full(len(d), int(dedicated_id), dtype=np.int64)
    d["PositionSource"] = np.where(d.ArucoX.notna(),
        np.where(d.SleapAnchorX.notna(), "aruco_sleap", "aruco_only"), "sleap_bridge")
    save_parquet(output, d, dict(num_frames=frame_stop-frame_start, frame_start=frame_start,
                                coordinate_mm_per_px=MM_PER_PX, permanent_censoring=False), schema=TRACK_SCHEMA)
    return d


def track_task(m, index):
    # One independent chunk/cup task, exactly as colony combine-and-stitch.
    count = len(m["cups"])
    chunk_index, cup_index = divmod(index, count)
    chunk, cup = m["chunks"][chunk_index], m["cups"][cup_index]
    cid = cup["cup_id"]
    output = root(m) / "tracked_chunks" / f"cup{cid:03d}" / f"chunk{chunk_index:03d}.parquet"
    done = output.with_suffix(".json")
    if done.exists():
        return
    q = pd.read_csv(root(m) / "cup_identities.csv")
    identity = q.loc[q.CupID.eq(cid)].iloc[0]
    if not identity.Eligible:
        write_json(done, dict(skipped=True, cup=cid, reason=identity.Reason))
        return
    folder = chunk_folder(m, chunk_index)
    a = pd.read_parquet(folder / f"cup{cid:03d}" / "aruco.parquet")
    nodes = pd.read_parquet(folder / f"cup{cid:03d}" / "nodes.parquet")
    competitors = manifest(folder / "competitor_ids.json")["retained_ids"]
    d = native_cup_track(a, nodes, cup_id=cid, dedicated_id=int(identity.ArucoID),
                         frame_start=chunk["start"], frame_stop=chunk["stop"], output=output,
                         tracking=m["tracking"], competitor_ids=competitors)
    xy = d.drop_duplicates("Frame")[["TrackX", "TrackY"]].to_numpy() * MM_PER_PX
    if len(xy) and np.any(np.linalg.norm(xy - cup["center_mm"], axis=1) > cup["radius_mm"] + 1e-6):
        raise AssertionError("Native tracker left its manual cup")
    write_json(done, dict(cup=cid, raw_id=int(identity.ArucoID), chunk=chunk_index,
                         frames=int(d.Frame.nunique()), rows=len(d), permanent_censoring=False))


def stitch_cup(m, index):
    from tracking.core.track_parquet import AntParquetWriter
    cid = m["cups"][index]["cup_id"]
    q = pd.read_csv(root(m) / "cup_identities.csv")
    if not q.loc[q.CupID.eq(cid), "Eligible"].iloc[0]:
        return
    dest = root(m) / "per_track" / f"TrackID_{cid:04d}_all_single.parquet"
    marker = root(m) / "state" / f"stitched_cup{cid:03d}.json"
    if marker.exists():
        return
    paths = [root(m) / "tracked_chunks" / f"cup{cid:03d}" / f"chunk{c['index']:03d}.parquet" for c in m["chunks"]]
    for path in paths:
        if not path.with_suffix(".json").exists():
            raise ValueError(f"Tracking chunk incomplete: {path}")
    schema = pq.ParquetFile(paths[0]).schema_arrow
    md = dict(schema.metadata or {})
    md.update({b"num_frames": str(m["n_frames"]).encode(), b"frame_start": b"0",
               b"protocol": PROTOCOL.encode(), b"permanent_censoring": b"False"})
    schema = schema.with_metadata(md)
    n_rows = 0
    with AntParquetWriter(dest, schema) as writer:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000):
                table = pa.Table.from_batches([batch]).cast(schema)
                writer.write_table(table)
                n_rows += table.num_rows
    write_json(marker, dict(cup=cid, track=str(dest), rows=n_rows, num_frames=m["n_frames"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ["detections", "regions", "metadata", "homography", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--pilot-identities", type=Path)
    p.add_argument("--panorama", type=Path)
    for command in ["map", "identify", "track", "stitch", "diagnose", "report", "publish"]:
        p = sub.add_parser(command)
        p.add_argument("--manifest", type=Path, required=True)
        if command in ["map", "track", "stitch", "diagnose"]:
            p.add_argument("--index", type=int, required=True)
        if command == "publish":
            p.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    m = manifest(args.manifest)
    if args.command == "map":
        map_chunk(m, args.index)
    elif args.command == "identify":
        identify(m)
    elif args.command == "track":
        track_task(m, args.index)
    elif args.command == "stitch":
        stitch_cup(m, args.index)
    else:
        from tracking.single_ant.full_cup_report import diagnose_cup, aggregate_report, publish
        if args.command == "diagnose":
            diagnose_cup(m, args.index)
        elif args.command == "report":
            aggregate_report(m)
        else:
            publish(m, args.destination)


if __name__ == "__main__":
    main()
