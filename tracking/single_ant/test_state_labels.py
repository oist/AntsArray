import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracking.single_ant.state_labels import export_labels


def test_every_cup_has_direct_labels_with_gap_and_identity_exclusions(tmp_path,monkeypatch):
    dataset=tmp_path/'dataset';packet=tmp_path/'packet'
    dataset.mkdir();(packet/'models').mkdir(parents=True)
    manifest=dict(n_frames=24,fps=24,cups=[dict(cup_id=i) for i in [1,2,3]],
        recording_segments=[dict(frame_offset=0,num_frames=8),dict(frame_offset=12,num_frames=12)])
    (dataset/'manifest.json').write_text(json.dumps(manifest))
    (packet/'models/dictionary.npz').write_bytes(b'test dictionary fingerprint')
    (packet/'models/selection.json').write_text(json.dumps(dict(n_states=3,supported=False)))
    pd.DataFrame([dict(track_id=1,ant='single:001',selected=False),
                  dict(track_id=2,ant='single:002',selected=True)]).to_csv(packet/'inventory_all.csv',index=False)
    metadata=dataset/'analysis/speed_by_metadata';metadata.mkdir(parents=True)
    pd.DataFrame(dict(CupID=[1,2,3],TrackID=[1,2,np.nan],IdentityStatus=['matched','ID mismatch','no track'])).to_csv(metadata/'metadata_join.csv',index=False)
    feature=packet/'features/single_001';feature.mkdir(parents=True)
    assignment=packet/'assignments/single_001';assignment.mkdir(parents=True)
    np.save(feature/'start.npy',[0,6]);np.save(feature/'length.npy',[6,6]);np.save(assignment/'state.npy',[1,2])
    speed=dataset/'analysis/speed_vectors/per_track/TrackID_0001_all_single';speed.mkdir(parents=True)
    np.save(speed/'speed_mm_s.npy',np.full(24,.5))
    monkeypatch.setattr('tracking.single_ant.state_labels.render',lambda *args:None)
    result=export_labels(packet,dataset)
    assert result['cups']==3 and result['predicted_ants']==1 and result['discovery_ants']==0
    assert not result['model_supported']
    definition=json.loads((packet/'s01_labels/definition.json').read_text())
    assert not definition['model_supported'] and definition['model_selection']['n_states']==3
    first=np.load(packet/'s01_labels/per_track/TrackID_0001_all_single/labels.npz')
    np.testing.assert_array_equal(first['category'],[1]*8+[-1]*4+[0]*12)
    assert set(np.unique(first['evidence']))=={0,1}
    for cid in [2,3]:
        labels=np.load(packet/f's01_labels/per_track/TrackID_{cid:04d}_all_single/labels.npz')
        assert (labels['observed']==-1).all() and (labels['category']==-1).all()
    summary=pd.read_csv(packet/'s01_labels/per_ant_summary.csv')
    np.testing.assert_allclose(summary.SleepPercent+summary.NotS01Percent+summary.AmbiguousPercent,100)
    assert summary.loc[0,'ValidSpeedFrames']==20
    assert summary.loc[0,'ValidSpeedFraction']==summary.loc[0,'First6hCoverage']==1
    assert summary.loc[0,'PredictionEligible'] and not summary.loc[0,'DiscoverySelected']


def test_label_export_rejects_a_different_dictionary(tmp_path,monkeypatch):
    # Reuse the full setup above, then verify the provenance guard before any
    # existing labels can be overwritten by a different fitted model.
    test_every_cup_has_direct_labels_with_gap_and_identity_exclusions(tmp_path,monkeypatch)
    (tmp_path/'packet/models/dictionary.npz').write_bytes(b'other model')
    import pytest
    with pytest.raises(ValueError,match='different inputs'):
        export_labels(tmp_path/'packet',tmp_path/'dataset')
