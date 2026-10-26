"""Freeze discovery sources before any candidate/reference generation."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;REPO=R.parents[3];P=R.parents[1]/'profiling/attempt002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def main():
    old=json.loads((P/'protocol.json').read_text());sources=dict(old['sources']);files=list((R/'code').glob('*.py'))+list((R/'tests').glob('*.py'))+[R/'SPEC.md',R/'DEFINITIONS.md',R/'freeze.py',R/'launch.py',R/'verify_environment.py']
    for p in files:sources[str(p.relative_to(REPO))]=sha(p)
    for name in ['model_fingerprints.json','runtime_fingerprints.json']:save(R/name,json.loads((P/name).read_text()))
    save(R/'protocol.json',dict(old,schema='native_support_discovery_single_pulse_v1',sources=sources,profiling_protocol_sha256=sha(P/'protocol.json'),prospective_spec_sha256=sha(R/'SPEC.md'),definitions_sha256=sha(R/'DEFINITIONS.md'),no_campaign_resource_ceilings=True,scope='finite development dataset and conditional confirmation only',seeds=[42,43],uncommitted_tail='retained in native N; excluded from committed-canvas endpoints',operator_work='explicit diagnostic work, never hidden model forwards'))
    print(sha(R/'protocol.json'))
if __name__=='__main__':main()
