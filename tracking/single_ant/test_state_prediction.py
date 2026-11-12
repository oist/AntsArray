"""Single-ant local prediction must not change discovery/colony eligibility."""
import json
import numpy as np
import pandas as pd

from tracking.single_ant.sleep_states import observed_states
from tracking.single_ant.state_prediction import predict_additional,identity_ok


def test_non_discovery_ant_can_reconstruct_existing_local_predictions(tmp_path):
    name='single_001';f=tmp_path/'features'/name;a=tmp_path/'assignments'/name
    f.mkdir(parents=True);a.mkdir(parents=True)
    np.save(f/'start.npy',np.array([2,8]));np.save(f/'length.npy',np.array([3,4]))
    np.save(a/'state.npy',np.array([0,2]))
    actual=observed_states(tmp_path,'single:001',24,False)
    expected=np.full(24,-1);expected[4:10]=0;expected[16:24]=2
    np.testing.assert_array_equal(actual,expected)
    assert (observed_states(tmp_path,'single:002',24,False)==-1).all()


def test_additional_prediction_preserves_discovery_inventory(tmp_path,monkeypatch):
    output=tmp_path/'packet';output.mkdir();source=tmp_path/'dataset';source.mkdir()
    (source/'manifest.json').write_text(json.dumps(dict(n_frames=100,fps=24)))
    table=pd.DataFrame([dict(ant='single:001',track_id=1,selected=True,original_source_path='one.parquet'),
                        dict(ant='single:002',track_id=2,selected=False,original_source_path='two.parquet')])
    table.to_csv(output/'inventory.csv',index=False);table.to_csv(output/'inventory_all.csv',index=False)
    before=(output/'inventory.csv').read_bytes();calls=[]
    def prediction(*args,**kwargs):
        calls.append(args[4]);return dict(prediction_eligible=True)
    monkeypatch.setattr('tracking.single_ant.state_prediction.predict_track',prediction)
    predict_additional(dict(output=str(output),source=str(source),block=str(tmp_path/'block')))
    assert calls==[2]
    assert (output/'inventory.csv').read_bytes()==before
    after=pd.read_csv(output/'inventory_all.csv')
    assert after.selected.tolist()==[True,False]
    assert after.prediction_eligible.tolist()==[True,True]


def test_identity_audit_returns_json_serializable_boolean(tmp_path):
    folder=tmp_path/'analysis/speed_by_metadata';folder.mkdir(parents=True)
    pd.DataFrame(dict(TrackID=[1,2],IdentityStatus=['matched','mismatch'])).to_csv(folder/'metadata_join.csv',index=False)
    results=[identity_ok(tmp_path,i) for i in [1,2,3]]
    assert json.dumps(results)=='[true, false, true]'
