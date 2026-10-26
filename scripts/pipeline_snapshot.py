#!/usr/bin/env python3
"""Copy pipeline source at submission so queued stages cannot mix code versions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

DIRECTORIES = ('analysis', 'scripts', 'camera_cal', 'tracking', 'detection_pipeline', 'lib')
SUFFIXES = {'.py', '.sh', '.sbatch', '.md', '.txt', '.json', '.csv', '.yaml', '.yml', '.toml', '.cfg'}


def sources(root):
    files = [p for name in DIRECTORIES for p in (root / name).rglob('*')
             if p.is_file() and p.suffix in SUFFIXES and not any(part.startswith('.') for part in p.relative_to(root).parts)]
    files += [p for p in root.iterdir() if p.is_file() and p.suffix in SUFFIXES and p.name != 'PIPELINE_SOURCE.json']
    if (root / 'DEPLOY_INFO').is_file():
        files.append(root / 'DEPLOY_INFO')
    return sorted(files)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(source, parent):
    source, parent = Path(source).resolve(), Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix='code_', dir=parent))
    try:
        files = sources(source)
        hashes = {}
        for path in files:
            relative = path.relative_to(source)
            copy = target / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            copy.write_bytes(data)
            copy.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
            hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        if files != sources(source) or any(digest(source / name) != value for name, value in hashes.items()):
            raise RuntimeError('Pipeline source changed while taking the snapshot; submit again.')
        manifest = target / 'PIPELINE_SOURCE.json'
        manifest.write_text(json.dumps(dict(source=str(source), created_epoch=time.time(), files=hashes), indent=2) + '\n')
        manifest.chmod(0o444)
    except Exception:
        shutil.rmtree(target)
        raise
    return target


def verify(root):
    root = Path(root)
    manifest = root / 'PIPELINE_SOURCE.json'
    # Direct interactive use and older deployments need not be snapshots.
    if not manifest.is_file():
        return
    hashes = json.loads(manifest.read_text())['files']
    changed = [name for name, value in hashes.items()
               if not (root / name).is_file() or digest(root / name) != value]
    if changed:
        raise RuntimeError('Pipeline snapshot was modified: ' + ', '.join(changed[:10]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        if not args.source or not args.parent:
            parser.error('--source and --parent are required')
        print(snapshot(args.source, args.parent))
