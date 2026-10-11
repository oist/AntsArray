"""Compare contact-history support across sleep definitions, for diagnosis only.

Historical motion labels are read solely as a reference here. They never enter
state fitting, current sleep assignment, or the main behavioral report pipeline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import sleep_motion_analysis_utils as sma
from analysis.full_behavior_sleep import contact_history_audit


def run(block,source,output,previous_labels=None):
    output.mkdir(parents=True,exist_ok=True)
    ants=pd.read_csv(source/'roles/ant_classes.csv')
    ants['cluster_id']='right_behavior_'+ants.ant_class.astype(str)
    paths=list((source/'return_context_cache').glob('pair_contact_bouts_*.parquet'))
    if len(paths)!=1:raise ValueError('Expected one current contact-bout cache; disambiguate before auditing')
    bouts=pd.read_parquet(paths[0],columns=['side','ant_a','ant_b','start_frame','is_new_onset'])
    bouts=bouts[bouts.side.eq('right')]
    definitions=[('legacy_motion_reference',block/'stitched/sleep_motion_labels')]
    if previous_labels is not None:definitions.append(('previous_state_labels',previous_labels))
    definitions.append(('current_state_labels',source/'sleep_from_states'))
    audits=[];coverage=[];inputs=[]
    for name,path in definitions:
        tracks=sma.load_sleep_label_tracks(path,ants)
        audit=contact_history_audit(tracks,bouts)
        audit.insert(0,'definition',name);audits.append(audit)
        for row in tracks.itertuples():
            state=np.load(row.state_path,mmap_mode='r')
            known=int((state>=0).sum());quiet=int((state==1).sum())
            total=int(ants.loc[ants.track_id.eq(row.track_id),'n_frames'].iloc[0])
            coverage.append(dict(definition=name,track_id=row.track_id,
                classified_fraction=known/total,sleep_fraction=quiet/known if known else np.nan))
            meta=Path(row.metadata_path)
            inputs.append(dict(definition=name,metadata_path=str(meta),
                metadata=json.loads(meta.read_text()),state_bytes=Path(row.state_path).stat().st_size))
        print(name,audit.select_dtypes('number').drop(columns='track_id').sum().to_dict(),flush=True)
    audit=pd.concat(audits,ignore_index=True)
    audit.to_csv(output/'contact_label_history_audit.csv',index=False)
    audit.groupby('definition').sum(numeric_only=True).drop(columns='track_id').to_csv(output/'contact_label_history_totals.csv')
    pd.DataFrame(coverage).to_csv(output/'contact_label_coverage_comparison.csv',index=False)
    (output/'contact_label_audit_inputs.json').write_text(json.dumps(dict(
        note='Historical labels used for this diagnostic only; primary analysis uses the selected state labels',
        contact_bouts=str(paths[0]),prior_seconds=10,guard_seconds=1,inputs=inputs),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous-labels',type=Path)
    args=parser.parse_args()
    run(args.block,args.source,args.output,args.previous_labels)
