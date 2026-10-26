"""Hash actual installed/model/scorer inputs, never publish their private paths."""
from pathlib import Path
import json,hashlib,sys
R=Path(__file__).resolve().parent;REPO=R.parents[2]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()
paths=json.loads((R.parent/'private/runtime_paths.json').read_text());protocol=json.loads((R/'protocol.json').read_text())
checks=[]
for p,h in {**protocol['sources'],**protocol['scorer_sources']}.items():
    assert sha(REPO/p)==h,p;checks.append(dict(role='source',file=p,sha256=h))
package=Path(paths['python']).parents[1]/'lib/python3.12/site-packages/vllm'
for p,h in protocol['runtime_source_pins'].items():
    assert sha(package/p)==h,p;checks.append(dict(role='runtime',file=p,sha256=h))
model=[]
for row in json.loads((R/'model_fingerprints.json').read_text())['files']:
    actual=sha(Path(paths['model_path'])/row['file']);assert actual==row['sha256'];model.append(dict(file=row['file'],sha256=actual,verified=True))
receipt=dict(passed=True,source_checks=len(checks),checks=checks,model_files=model,interpreter_sha256=sha(Path(paths['python'])),official_gold_sha256=sha(Path(paths['gold_path'])),private_manifest_sha256=sha(R.parent/'private/manifest.json'),protocol_sha256=sha(R/'protocol.json'),model_forwards=0)
with (R/'environment_qualification.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True)
print(json.dumps(dict(passed=True,source_checks=len(checks),model_files=len(model))))
