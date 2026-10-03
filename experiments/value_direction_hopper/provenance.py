"""Immutable source snapshots let completed runs survive later code iterations."""
import hashlib
import json
from pathlib import Path
import shutil


def snapshot(root):
    root=Path(root);contract=json.loads((root/'configuration.json').read_text());records={}
    destination=root/'sources';destination.mkdir(exist_ok=True)
    for source,digest in contract['source_hashes'].items():
        source=Path(source);out=destination/(digest+'_'+source.name)
        if not out.exists():
            if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise ValueError('Cannot snapshot changed source: '+str(source))
            shutil.copyfile(source,out)
        if hashlib.sha256(out.read_bytes()).hexdigest()!=digest:raise ValueError('Corrupt archived source')
        records[str(source)]=str(out.resolve())
    path=root/'source_snapshots.json'
    if path.exists() and json.loads(path.read_text())!=records:raise ValueError('Snapshot index changed')
    if not path.exists():path.write_text(json.dumps(records,indent=2)+'\n')
    return records
