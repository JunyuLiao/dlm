"""Verify actual model/runtime/scorer bytes without any model invocation."""
from pathlib import Path
import hashlib,json,time,sys,importlib.metadata
R=Path(__file__).resolve().parent;S=R.parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    paths=json.loads((S/'private/runtime_paths.json').read_text());p=json.loads((R/'protocol.json').read_text());m=json.loads((R/'model_fingerprints.json').read_text());files=[]
    for e in m['files']:
        x=Path(paths['model_path'])/e['file'];assert x.stat().st_size==e['bytes'] and sha(x)==e['sha256'];files.append({k:e[k] for k in ['file','bytes','sha256']})
    package=Path(paths['python']).parents[1]/'lib/python3.12/site-packages/vllm'
    for name,h in p['runtime_source_pins'].items():assert sha(package/name)==h
    gold=sha(Path(paths['gold_path']));profile=json.loads((S/'profiling/attempt002/model_runtime_verification001.json').read_text());assert gold==profile['official_gold_sha256']
    import torch
    versions=json.loads((S/'profiling/attempt002/runtime_versions_verification001.json').read_text());assert torch.__version__==versions['torch'] and torch.version.cuda==versions['torch_cuda'] and sys.version==versions['python']
    for name,version in versions['packages'].items():assert importlib.metadata.version(name)==version
    with (R/'model_runtime_verification001.json').open('x') as f:json.dump(dict(passed=True,files=files,runtime_source_hashes=p['runtime_source_pins'],official_gold_sha256=gold,versions=versions,verified_unix=time.time(),model_forwards=0),f,indent=2,sort_keys=True)
if __name__=='__main__':main()
