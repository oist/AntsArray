"""A published packet and both direct-label exports must advertise completion."""
import json
from pathlib import Path
import shutil

from tracking.single_ant.eigenposture import publish


def test_publish_preserves_label_completion_markers(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();packet=run/'packet';labels=packet/'s01_labels'
    labels.mkdir(parents=True)
    (packet/'COMPLETE.json').write_text(json.dumps(dict(complete=True)))
    (labels/'COMPLETE.json').write_text(json.dumps(dict(cups=41,predicted_ants=38)))
    (labels/'per_ant_summary.csv').write_text('CupID,AmbiguousPercent\n1,20\n')
    for name in ['posture_interpolated','sleep_motion']:
        (run/'block/stitched'/name).mkdir(parents=True)
    def transfer(command,**kwargs):
        assert command[0]=='rsync'
        # Match the real transfer's exclusion of completion markers. The
        # publisher must release them only after all files have been copied.
        shutil.copytree(Path(command[-2]),Path(command[-1]),dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('COMPLETE.json'))
    monkeypatch.setattr('tracking.single_ant.eigenposture.subprocess.run',transfer)
    destination=tmp_path/'published/analysis/eigenposture'
    publish(dict(run=str(run),output=str(packet),block=str(run/'block'),destination=str(destination)))
    for target in [destination/'s01_labels',destination.parent/'sleep_s01_exclusion']:
        assert json.loads((target/'COMPLETE.json').read_text())['predicted_ants']==38
    assert (destination/'COMPLETE.json').exists() and (run/'PUBLISHED.json').exists()
