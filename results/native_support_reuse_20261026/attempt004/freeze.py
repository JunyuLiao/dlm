from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;REPO=R.parents[2];P=REPO/'results/native_support_discovery_20261026/profiling/attempt002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True)
old=json.loads((P/'protocol.json').read_text());sources=dict(old['sources'])
for p in list((R/'code').glob('*.py'))+list((R/'tests').glob('*.py'))+[R/'SPEC.md',R/'METHOD_REGISTRY.json',R/'freeze.py',R/'launch.py',R/'verify_environment.py']:
    sources[str(p.relative_to(REPO))]=sha(p)
for n in ['model_fingerprints.json','runtime_fingerprints.json']:save(R/n,json.loads((P/n).read_text()))
save(R/'protocol.json',dict(old,schema='controlled_native_reuse_v1',sources=sources,prospective_spec_sha256=sha(R/'SPEC.md'),manifest_identity_sha256=sha(R.parent/'manifest_identity.json'),private_manifest_sha256=sha(R.parent/'private/manifest.json'),no_campaign_resource_ceilings=True,seeds=[42,43],old_pulse_protocol_sha256='10398a912c355dabf3d6511475a7bdf0fb3c3434c0e7055e439a724fd5bc0d70'))
print(sha(R/'protocol.json'))
