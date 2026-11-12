#!/usr/bin/env python3
"""Reusable manual-cup workflow. Run on a Deigo login node.

python -m tracking.single_ant.run /bucket/.../20261027 --blocks block01 block02 --wait
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from tracking.single_ant import full_cup_tracking as tracking
from tracking.single_ant.inputs import detection_manifest
from tracking.single_ant.manual_cup_tracking import write_json
from tracking.colony.combine_blocks import check_fingerprint, consecutive_groups


def freeze_code(args, label):
    base = (args.work_root or Path("/flash/ReiterU/ant_tmp")/getpass.getuser()/"single_ant").resolve()
    run = base/(label+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    code = run/"code"
    for folder in ["tracking", "analysis", "camera_cal"]:
        for source in (REPO/folder).rglob("*"):
            if source.is_file() and source.suffix in {".py", ".md", ".sh"}:
                target = code/source.relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    return run, code


def completed_manifests(paths):
    paths = [str(p.resolve()) for p in paths]
    for path in paths:
        m = tracking.manifest(path)
        if m.get("preserve_block_speeds") or not (tracking.root(m)/"FULL_RUN_COMPLETE.json").is_file():
            raise ValueError(f"Completed individual-block manifests are required: {path}")
    return paths


def annotation_panorama(metadata):
    """Use the image belonging to the selected annotations, including alternate calibrations."""
    sibling = metadata.with_name("panorama_from_hmats.png")
    if sibling.is_file():
        return sibling
    recorded = Path(tracking.manifest(metadata)["panorama_image"])
    return recorded if recorded.is_file() else None


def initialize_stitch(args):
    if not args.output or len(args.stitch_only) < 2:
        raise ValueError("--stitch-only requires at least two manifests and --output DIR")
    prior = completed_manifests(args.stitch_only)
    destination = args.output.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Existing combined results: {destination}")
    from tracking.single_ant.combine import offsets
    records = sorted((tracking.manifest(p) for p in prior), key=lambda m: datetime.fromisoformat(m["start_datetime"]))
    offsets(records)
    if any(m["cups"] != records[0]["cups"] for m in records[1:]):
        raise ValueError("Cup labels/geometry differ between selected blocks")
    run, code = freeze_code(args, "stitch_"+destination.name)
    plan = dict(run=str(run), code=str(code), blocks=[], prior_manifests=prior,
                combined=str(destination), combined_stage=str(run/"combined"),
                python=sys.executable, exclude_nodes=args.exclude_nodes)
    write_json(run/"plan.json", plan)
    return run/"plan.json"


def initialize(args):
    dataset = args.dataset.resolve()
    names = args.blocks or sorted(p.name for p in dataset.glob("block[0-9]*") if p.is_dir())
    if not names or any(Path(n).name != n or not n.removeprefix("block").isdigit() for n in names):
        raise ValueError("Select numbered blockNN folders")
    names = sorted(set(names), key=lambda n: int(n.removeprefix("block")))
    prior = completed_manifests(args.stitch_with or [])
    run, code = freeze_code(args, dataset.name)
    blocks = []
    for name in names:
        block, work = dataset/name, run/name
        destination = block/args.analysis_dir
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Existing results: {destination}; resume the original run or choose --analysis-dir")
        print(f"Checking {block}", flush=True)
        source, (regions, metadata, homography) = detection_manifest(
            block, args.regions, args.homography, interrupted_cameras=getattr(args, "interrupted_cameras", ()))
        work.mkdir(parents=True, exist_ok=True)
        write_json(work/"source.json", source)
        panorama = annotation_panorama(metadata)
        tracking.prepare(SimpleNamespace(output=work, detections=work/"source.json", regions=regions,
                         metadata=metadata, homography=homography, pilot_identities=None, panorama=panorama,
                         resident_registry=getattr(args, "resident_registry", None)))
        m = tracking.manifest(work/"manifest.json")
        m["title"] = dataset.name+" "+name
        write_json(work/"manifest.json", m)
        blocks.append(dict(name=name, number=int(name.removeprefix("block")),
                           manifest=str(work/"manifest.json"), destination=str(destination),
                           chunks=len(m["chunks"]), cups=len(m["cups"])))
        print(f"{name}: {source['n_frames']} frames; {len(source['chunks'])} chunks; labels {regions}", flush=True)
    if prior:
        from tracking.single_ant.combine import offsets
        records = [tracking.manifest(p) for p in prior+[b["manifest"] for b in blocks]]
        if any(b["cups"] != records[0]["cups"] for b in records[1:]):
            raise ValueError("Earlier block cup labels/geometry differ; review their correspondence")
        offsets(sorted(records, key=lambda b: datetime.fromisoformat(b["start_datetime"])))
    combined = dataset/args.combined_dir
    if len(blocks)+len(prior) > 1 and combined.exists() and any(combined.iterdir()):
        raise FileExistsError(f"Existing combined results: {combined}")
    plan = dict(run=str(run), code=str(code), dataset=str(dataset), blocks=blocks,
                combined=str(combined), combined_stage=str(run/"combined"), python=sys.executable,
                track_task_batch_size=getattr(args, "track_task_batch_size", 1),
                exclude_nodes=args.exclude_nodes, prior_manifests=prior)
    write_json(run/"plan.json", plan)
    return run/"plan.json"


def submit(plan_path):
    plan = tracking.manifest(plan_path)
    batch_size = int(plan.get("track_task_batch_size", 1))
    if batch_size < 1:
        raise ValueError("track_task_batch_size must be positive")
    run, code = Path(plan["run"]), Path(plan["code"])
    jobs_path = run/"jobs.json"
    jobs = tracking.manifest(jobs_path) if jobs_path.exists() else {}
    attempts_path = run/"job_attempts.json"
    attempts = tracking.manifest(attempts_path) if attempts_path.exists() else {name: [jid] for name, jid in jobs.items()}
    states, records, replaced = {}, {}, set()
    if jobs:
        rows = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()),
                                        "--format=JobID,State"], text=True)
        for row in rows.splitlines():
            jid, state, *_ = row.split("|")
            states.setdefault(jid.split("_")[0], set()).add(state.split()[0])
            records.setdefault(jid.split("_")[0], []).append((jid, state.split()[0]))
    def job(name, command, dependencies=(), array=None, cpus=1, mem="8G", partition="short", wall="0-01:00:00"):
        old = jobs.get(name)
        if old:
            observed = states.get(old)
            if not observed:
                raise RuntimeError(f"Cannot verify state of previous job {old}; check Slurm accounting before resuming")
            failed = any(s.startswith(("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "PREEMPTED", "BOOT_FAIL", "DEADLINE")) for s in observed)
            if not failed and not replaced.intersection(dependencies):
                return old
            if observed & {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
                subprocess.run(["scancel", old], check=True)
            elif array and failed:
                slots = [jid.split("_", 1)[-1] for jid, state in records[old] if state != "COMPLETED"]
                if slots and all(slot.isdigit() for slot in slots):
                    array = ",".join(sorted(set(slots), key=int))
        script = run/"jobs"/(name+".sbatch")
        script.parent.mkdir(exist_ok=True)
        (run/"logs").mkdir(exist_ok=True)
        script.write_text(f"""#!/usr/bin/env bash
#SBATCH -J singleant_{name}
#SBATCH -p {partition}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {wall}
{('#SBATCH --exclude='+plan['exclude_nodes']) if plan.get('exclude_nodes') else ''}
#SBATCH -o {run}/logs/{name}_%A_%a.out
#SBATCH -e {run}/logs/{name}_%A_%a.err
set -euo pipefail
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS={cpus} OMP_NUM_THREADS={cpus}
cd {shlex.quote(str(code))}
{command}
""")
        args = ["sbatch", "--parsable"]
        needed = [dep for dep in dependencies if states.get(dep) != {"COMPLETED"}]
        if needed:
            args += ["--dependency=afterok:"+":".join(needed)]
        if array:
            args += ["--array="+array]
        jobs[name] = subprocess.check_output(args+[str(script)], text=True).strip().split(";")[0]
        replaced.add(jobs[name])
        attempts.setdefault(name, []).append(jobs[name])
        write_json(attempts_path, attempts)
        write_json(jobs_path, jobs)  # Retain submitted IDs even if a later submission fails.
        return jobs[name]
    def internal(stage, block=None, array=False):
        cmd = [plan["python"], "-u", "-m", "tracking.single_ant.run", "_worker", stage, "--plan", str(plan_path)]
        if block is not None:
            cmd += ["--block", str(block)]
        return shlex.join(cmd)+(' --index "$SLURM_ARRAY_TASK_ID"' if array else '')
    def native(stage, b, array=False):
        cmd = [plan["python"], "-u", "-m", "tracking.single_ant.full_cup_tracking", stage, "--manifest", b["manifest"]]
        return shlex.join(cmd)+(' --index "$SLURM_ARRAY_TASK_ID"' if array else '')
    maps = [job(b["name"]+"_map", internal("map", i, True), array=f"0-{b['chunks']-1}",
                cpus=4, mem="32G", partition="compute", wall="0-02:00:00") for i, b in enumerate(plan["blocks"])]
    ids = job("identities", internal("identify"), maps) if maps else None
    reports = []
    for i, b in enumerate(plan["blocks"]):
        prefix = b["name"]
        count = (b['chunks']*b['cups'] + batch_size - 1)//batch_size
        command = internal("track", i, True) if batch_size > 1 else native("track", b, True)
        track = job(prefix+"_track", command, [ids], array=f"0-{count-1}")
        stitch = job(prefix+"_stitch", native("stitch", b, True), [track], array=f"0-{b['cups']-1}")
        diag = job(prefix+"_diagnose", native("diagnose", b, True), [stitch], array=f"0-{b['cups']-1}",
                   cpus=2, mem=plan.get("diagnose_mem", "16G"))
        reports.append(job(prefix+"_report", native("report", b), [diag], cpus=4, mem="16G"))
    if len(plan["blocks"])+len(plan.get("prior_manifests", [])) > 1:
        cp = job("combine_plan", internal("combine-plan"), reports)
        count = sum(b["cups"] for b in plan["blocks"])+sum(len(tracking.manifest(p)["cups"]) for p in plan.get("prior_manifests", []))
        cw = job("combine_ants", internal("combine", array=True), [cp], array=f"0-{count-1}",
                 cpus=2, mem=plan.get("combine_mem", "32G"), wall="0-02:00:00")
        job("combined_report", internal("combined-report"), [cw], cpus=4, mem="16G")
    print(json.dumps(dict(run=str(run), jobs=jobs, block_outputs=[b["destination"] for b in plan["blocks"]], combined=plan["combined"]), indent=2), flush=True)


def work(stage, plan, index=0, block=0):
    manifests = [tracking.manifest(b["manifest"]) for b in plan["blocks"]]
    if stage == "map":
        m = manifests[block]
        chunk = m["chunks"][index]
        source = tracking.manifest(m["detections"])
        for f in source["files"]:
            if f["frame_start"] < chunk["stop"] and f["frame_start"]+f["frames"] > chunk["start"]:
                for record in f["fingerprints"]:
                    check_fingerprint(record)
        tracking.map_chunk(m, index)
    elif stage == "track":
        m = manifests[block]
        batch_size = int(plan.get("track_task_batch_size", 1))
        for slot in range(index*batch_size, min((index+1)*batch_size, len(m["chunks"])*len(m["cups"]))):
            tracking.track_task(m, slot)
    elif stage == "identify":
        previous, changes = {}, []
        for path in sorted(plan.get("prior_manifests", []), key=lambda p: tracking.manifest(p)["start_datetime"]):
            old = tracking.manifest(path)
            q = pd.read_csv(tracking.root(old)/"cup_identities.csv")
            if old["cups"] != manifests[0]["cups"]:
                raise ValueError("Earlier block cup labels/geometry differ; review their correspondence")
            previous.update({int(r.CupID): int(r.ArucoID) for r in q.loc[q.Eligible].itertuples(index=False)})
        for b, m in zip(plan["blocks"], manifests):
            tracking.identify(m)
            q = pd.read_csv(tracking.root(m)/"cup_identities.csv")
            counts = pd.read_csv(tracking.root(m)/"full_recording_id_counts.csv")
            for i, row in q.iterrows():
                old = previous.get(int(row.CupID))
                if old is not None and not row.Eligible:
                    observed = counts.loc[counts.CupID.eq(row.CupID)&counts.ArucoID.eq(old)]
                    q.loc[i, ["ArucoID", "Eligible", "AssignmentSource", "Reason"]] = [old, True, "previous_block", "keep_trying_established_cup_ID"]
                    if len(observed):
                        q.loc[i, "SingleIDFractionFull"] = float(observed.SingleInstanceFraction.iloc[0])
                        q.loc[i, "TagAndPoseFractionFull"] = float(observed.SupportedFraction.iloc[0])
                elif old is not None and row.Eligible and int(row.ArucoID) != old:
                    changes.append(dict(block=b["name"], cup=int(row.CupID), previous_ID=old, new_ID=int(row.ArucoID)))
            q.to_csv(tracking.root(m)/"cup_identities.csv", index=False)
            previous.update({int(r.CupID): int(r.ArucoID) for r in q.loc[q.Eligible].itertuples(index=False)})
            write_json(tracking.root(m)/"IDENTITIES_COMPLETE.json", dict(eligible=int(q.Eligible.sum()), cups=len(q), permanent_censoring=False))
        write_json(Path(plan["run"])/"identity_changes.json", changes)
    elif stage == "combine-plan":
        from tracking.single_ant.combine import prepare
        prior = plan.get("prior_manifests", [])
        groups = [plan["blocks"]] if prior else [g for g in consecutive_groups(plan["blocks"]) if len(g)>1]
        outputs = []
        for g in groups:
            suffix = g[0]["name"]+"_"+g[-1]["name"] if len(groups)>1 else ""
            path = prepare(prior+[b["manifest"] for b in g], Path(plan["combined_stage"])/suffix, Path(plan["combined"])/suffix)
            outputs.append(str(path))
        write_json(Path(plan["run"])/"combined_manifests.json", outputs)
    else:
        outputs = tracking.manifest(Path(plan["run"])/"combined_manifests.json")
        for path in outputs:
            m = tracking.manifest(path)
            if stage == "combine":
                if index < len(m["ants"]):
                    from tracking.single_ant.combine import worker
                    worker(m, index)
                    return
                index -= len(m["ants"])
            elif stage == "combined-report":
                from tracking.single_ant.full_cup_report import aggregate_report
                aggregate_report(m)


def watch(plan):
    with (Path(plan["run"])/"publisher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("The login-node publisher is already running for this run.", flush=True)
            return
        _watch(plan)


def _watch(plan):
    """Transfer from the login node: compute-node bucket mounts are read-only."""
    from tracking.single_ant.full_cup_report import publish
    run = Path(plan["run"])
    published = set()
    while True:
        jobs = tracking.manifest(run/"jobs.json")
        accounting = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()), "--format=JobID,State,ExitCode"], text=True)
        failures = [line for line in accounting.splitlines() if any(state in line for state in ["FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED"])]
        if failures:
            raise RuntimeError("Cluster jobs failed: "+"; ".join(failures[:8]))
        targets = [(b["manifest"], b["destination"]) for b in plan["blocks"]]
        combined = run/"combined_manifests.json"
        ready = len(plan["blocks"])+len(plan.get("prior_manifests", [])) == 1 or combined.exists()
        if combined.exists():
            targets += [(p, tracking.manifest(p)["published_output"]) for p in tracking.manifest(combined)]
        for source, target in targets:
            m = tracking.manifest(source)
            if source not in published and (tracking.root(m)/"FULL_RUN_COMPLETE.json").exists():
                m["publication_relocations"] = {
                    str(tracking.root(tracking.manifest(b["manifest"]))): b["destination"] for b in plan["blocks"]}
                publish(m, Path(target))
                published.add(source)
                print("Published "+target, flush=True)
        status = dict(published=len(published), targets=len(targets), combination_planned=ready)
        write_json(run/"progress.json", status)
        print(json.dumps(status), flush=True)
        if ready and len(published) == len(targets):
            write_json(run/"PUBLISHED.json", dict(outputs=[t for _, t in targets], complete=True))
            if len(targets)>len(plan["blocks"]):
                write_json(Path(plan["combined"])/"_SUCCESS.json", dict(run=str(run), groups=[t for _, t in targets[len(plan["blocks"]):]]))
            return
        time.sleep(30)


def main():
    if len(sys.argv)>1 and sys.argv[1] == "_worker":
        p = argparse.ArgumentParser(); p.add_argument("stage"); p.add_argument("--plan", type=Path, required=True)
        p.add_argument("--index", type=int, default=0); p.add_argument("--block", type=int, default=0)
        a = p.parse_args(sys.argv[2:]); plan = tracking.manifest(a.plan)
        watch(plan) if a.stage == "watch" else work(a.stage, plan, a.index, a.block)
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path, nargs="?"); p.add_argument("--blocks", nargs="+")
    p.add_argument("--regions", type=Path); p.add_argument("--homography", type=Path)
    p.add_argument("--interrupted-cameras", nargs="+", default=[],
                   help="Explicitly acknowledge synthesized crash sidecars and assumed common-trigger alignment for these cameras")
    p.add_argument("--track-task-batch-size", type=int, default=1,
                   help="Consecutive chunk/cup tasks per Slurm slot, to stay within array and submit limits")
    p.add_argument("--resident-registry", type=Path, help="Curated TSV with row-major cup and tag_id columns; blank tags remain unassigned")
    p.add_argument("--stitch-with", type=Path, nargs="+", help="Completed individual-block manifests to include chronologically, including earlier dates")
    p.add_argument("--stitch-only", type=Path, nargs="+", help="Stitch completed blocks without tracking")
    p.add_argument("--output", type=Path, help="Destination for --stitch-only")
    p.add_argument("--analysis-dir", default="analysis_outputs/single_ant")
    p.add_argument("--combined-dir", default="continous_stitched")
    p.add_argument("--work-root", type=Path); p.add_argument("--resume", type=Path); p.add_argument("--wait", action="store_true")
    p.add_argument("--exclude-nodes", help="Slurm node exclusion, for unavailable compute-node environments")
    a = p.parse_args()
    if a.dataset is None and a.resume is None and not a.stitch_only:
        p.error("Provide a dataset, --stitch-only, or --resume RUN/plan.json")
    if a.stitch_only and (a.dataset or a.resume or a.stitch_with):
        p.error("Use --stitch-only on its own with --output")
    path = a.resume.resolve() if a.resume else initialize_stitch(a) if a.stitch_only else initialize(a)
    plan = tracking.manifest(path)
    if a.exclude_nodes:
        plan["exclude_nodes"] = a.exclude_nodes
        write_json(path, plan)
    submit(path)
    if a.wait:
        watch(plan)
    else:
        log = (Path(plan["run"])/"publish.log").open("a")
        subprocess.Popen([plan["python"], str(Path(plan["code"])/"tracking/single_ant/run.py"), "_worker", "watch", "--plan", str(path)],
                         cwd=plan["code"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                         env=dict(os.environ, PYTHONNOUSERSITE="1", MPLBACKEND="Agg"))
    print("Run manifest: "+str(path), flush=True)


if __name__ == "__main__":
    main()
