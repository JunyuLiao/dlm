"""Read/hash actual model and installed runtime before GPU generation."""
import hashlib
import json
from pathlib import Path
import time


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(16<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];repo=root.parents[3]
    old=repo/'results/denoising_e2e_research_20261024/attempt003'
    paths=json.loads((root/'private/runtime_paths.json').read_text());model=Path(paths['model_path'])
    expected=json.loads((old/'model_fingerprints.json').read_text());files=[]
    for item in expected['files']:
        digest=sha(model/item['file']);assert digest==item['sha256'],item['file']
        files.append(dict(file=item['file'],sha256=digest,bytes=(model/item['file']).stat().st_size))
    import vllm
    package=Path(vllm.__file__).parent
    runtime=json.loads((old/'runtime_source_audit001.json').read_text())['files']
    assert all(sha(package/n)==v for n,v in runtime.items())
    with (root/'model_runtime_verification001.json').open('x') as f:json.dump(dict(passed=True,checked_unix=time.time(),model_revision=expected['model_revision'],files=files,runtime_source_hashes=runtime),f,indent=2)
    print(json.dumps(dict(passed=True,model_files=len(files),runtime_sources=len(runtime))))


if __name__=='__main__':main()
