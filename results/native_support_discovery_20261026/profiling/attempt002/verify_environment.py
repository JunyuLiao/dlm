"""Read actual model/runtime/scorer files, write only a new verification receipt."""
from pathlib import Path
import hashlib,json,time
R=Path(__file__).resolve().parent;S=R.parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    paths=json.loads((S/'private/runtime_paths.json').read_text());p=json.loads((R/'protocol.json').read_text());m=json.loads((R/'model_fingerprints.json').read_text());files=[]
    for entry in m['files']:
        x=Path(paths['model_path'])/entry['file'];assert x.stat().st_size==entry['bytes'] and sha(x)==entry['sha256'];files.append({k:entry[k] for k in ['file','sha256','bytes']})
    package=Path(paths['python']).parents[1]/'lib/python3.12/site-packages/vllm'
    for name,h in p['runtime_source_pins'].items():assert sha(package/name)==h,name
    gold=sha(Path(paths['gold_path']));expected=json.loads((S.parent/'commit_boundary_single_pulse_20261026/scorer_verification001.json').read_text())['files'][0]['sha256'];assert gold==expected
    with (R/'model_runtime_verification001.json').open('x') as f:json.dump(dict(passed=True,checked_unix=time.time(),files=files,runtime_source_hashes=p['runtime_source_pins'],official_gold_sha256=gold,model_forwards=0),f,indent=2,sort_keys=True)
    print('Verified model16/runtime7/official gold; no model calls.')
if __name__=='__main__':main()
