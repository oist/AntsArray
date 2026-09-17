"""Prepare deterministic per-ant fanout tasks from a longitudinal inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def prepare(source):
    source=Path(source)
    manifest=json.loads((source/"run_manifest.json").read_text())
    inventory=pd.read_parquet(source/"inventory.parquet")
    if inventory.duplicated(["source_block","side","track_id"]).any():
        raise ValueError("Duplicate source/ant identity in inventory")
    inventory["ant"]=inventory.side+":"+inventory.track_id.astype(int).astype(str).str.zfill(3)
    lookup={w["block"]:(i,w,manifest["sources"][i]["info"]) for i,w in enumerate(manifest["windows"])}
    tasks=[]
    for ant,part in inventory.groupby("ant",sort=True):
        entries=[]
        for row in part.itertuples():
            index,window,info=lookup[row.source_block]
            entries.append(dict(ant=ant,track_name=row.track_name,block_index=index,block=row.source_block,
                                start=window["start"],stop=window["stop"],fps=info["fps"],frame_start=info["frame_start"]))
        tasks.append(dict(ant=ant,entries=sorted(entries,key=lambda e:e["block_index"])))
    return tasks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-folder",type=Path,required=True)
    parser.add_argument("--tasks",type=Path,required=True)
    args=parser.parse_args()
    tasks=prepare(args.analysis_folder)
    args.tasks.parent.mkdir(parents=True,exist_ok=True)
    args.tasks.write_text(json.dumps(tasks,indent=2)+"\n")
    print(json.dumps(dict(ants=len(tasks),tracks=sum(len(t["entries"]) for t in tasks),array=f"0-{len(tasks)-1}%8")))


if __name__=="__main__":main()
