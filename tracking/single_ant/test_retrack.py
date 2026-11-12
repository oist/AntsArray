"""Reruns must retain their actual recording date, block and input detections."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tracking.single_ant import retrack


@pytest.mark.parametrize('date,block',[('20261027','block02'),('20261028','block01')])
def test_tracking_only_rerun_retains_recording_identity(tmp_path,monkeypatch,date,block):
    baseline=tmp_path/'baseline';baseline.mkdir();(baseline/'mapped').mkdir()
    source=baseline/'manifest.json'
    source.write_text(json.dumps(dict(output=str(baseline),source_block=f'/bucket/ants/{date}/{block}',
        start_datetime='2026-10-28T10:40:00+09:00',n_frames=100,fps=24,chunks=[dict(index=0)],cups=[dict(cup_id=1)])))
    original=source.read_bytes();(baseline/'FULL_RUN_COMPLETE.json').write_text('{}')
    run=tmp_path/'run';code=run/'code';(code/'tracking/core').mkdir(parents=True)
    for name in ['tracking_utils.py','pose_candidates.py']:
        shutil.copy2(Path(retrack.__file__).parents[1]/'core'/name,code/'tracking/core'/name)
    monkeypatch.setattr(retrack,'freeze_code',lambda args,label:(run,code))
    args=SimpleNamespace(source=source,reference=None,destination=tmp_path/'published',work_root=tmp_path,
                         tracking_only=True,exclude_nodes='slow_node')
    path=retrack.initialize(args);plan=json.loads(path.read_text())
    result=json.loads((Path(plan['output'])/'manifest.json').read_text())
    assert Path(plan['output']).name==block and plan['tracking_only']
    assert result['recording_segments'][0]['block']==f'{date}_{block}'
    assert result['recording_segments'][0]['source_block']==f'/bucket/ants/{date}/{block}'
    assert (Path(plan['output'])/'mapped').resolve()==baseline/'mapped'
    assert not (Path(plan['output'])/'analysis/eigenposture').exists()
    assert source.read_bytes()==original
