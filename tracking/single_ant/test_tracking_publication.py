import json
import shutil

from tracking.colony.combine_blocks import fingerprint, check_fingerprint
from tracking.single_ant.full_cup_report import relocate_publication_metadata


def test_stitched_publication_references_published_source_blocks(tmp_path):
    block = tmp_path/'scratch/block'; block.mkdir(parents=True)
    published_block = tmp_path/'bucket/longer_block_destination'; published_block.mkdir(parents=True)
    (block/'manifest.json').write_text(json.dumps(dict(output=str(block))))
    (published_block/'manifest.json').write_text(json.dumps(dict(output=str(published_block))))
    source = tmp_path/'scratch/combined'; source.mkdir()
    destination = tmp_path/'bucket/combined'; destination.mkdir()
    original = dict(output=str(source), source_manifests=[fingerprint(block/'manifest.json')],
                    source_track=str(block/'per_track/ant.parquet'))
    (source/'manifest.json').write_text(json.dumps(original))
    shutil.copy2(source/'manifest.json', destination/'manifest.json')
    relocate_publication_metadata(dict(output=str(source), publication_relocations={str(block):str(published_block)}), destination)
    result = json.loads((destination/'manifest.json').read_text())
    assert result['output']==str(destination)
    assert result['source_track']==str(published_block/'per_track/ant.parquet')
    assert result['source_manifests']==[fingerprint(published_block/'manifest.json')]
    check_fingerprint(result['source_manifests'][0])
    assert json.loads((source/'manifest.json').read_text())==original
