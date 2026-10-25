"""Bind all consequential sources and identities before any native generation."""
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent;repo=root.parents[2]
    prior=repo/'results/denoising_e2e_research_20261024/attempt003'
    old=json.loads((prior/'e2e_protocol.json').read_text())
    sources={str(Path(p).relative_to(repo)):h for p,h in old['sources'].items() if Path(p).is_relative_to(repo)}
    external={p:h for p,h in old['sources'].items() if not Path(p).is_relative_to(repo)}
    for path,h in external.items():assert sha(path)==h
    (root/'private/external_source_paths.json').write_text(json.dumps(external))
    for path,h in sources.items():assert sha(repo/path)==h
    own=list(root.glob('*.py'))+list((root/'code').glob('*.py'))+list((root/'tests').glob('test*.py'))+[root/'README.md']
    sources.update({str(p.relative_to(repo)):sha(p) for p in own})
    (root/'model_fingerprints.json').write_bytes((prior/'model_fingerprints.json').read_bytes())
    (root/'runtime_fingerprints.json').write_bytes((prior/'runtime_fingerprints.json').read_bytes())
    public_model=json.loads((root/'model_fingerprints.json').read_text())
    private_scorer={str(Path(p).relative_to(repo)):h for p,h in old['scorer_sources'].items()}
    paths=dict(model_path=old['model_path'],python='/home/exouser/dyh/v31_env_20261005/bin/python',gold_path=str(repo/'results/humaneval_value_observe_no_overlap_k256_20261019/attempt001/private/pools/thinking_on/gold.json'))
    (root/'private/runtime_paths.json').write_text(json.dumps(paths))
    result=dict(schema='stage_a_mechanism_v1',starting_commit='7f04e2be0348c341b2bc9a7deae83c157efa45d0',
        spec_sha256=sha(root/'README.md'),manifest_sha256=sha(root/'manifest_identity.json'),private_manifest_sha256=sha(root/'private/manifest.json'),
        model_revision=public_model['model_revision'],model_fingerprint_sha256=sha(root/'model_fingerprints.json'),
        runtime_fingerprint_sha256=sha(root/'runtime_fingerprints.json'),runtime_source_pins=json.loads((prior/'runtime_source_audit001.json').read_text())['files'],
        sources=sources,external_sources={p.split('site-packages/')[-1]:h for p,h in external.items()},scorer_sources=private_scorer,native_sampler=old['native_sampler'],
        layers=[5,17,29],heads=[0,8],queryblocks=[0,1],geometry=[128,64],seeds=[42,43],
        physical_deletion='max(1,floor(0.05B)) per unit, same absolute count at all three locations',
        native_output_caps={'aime26':8192,'humaneval':8192,'longbench_v2':16384},
        current_error_guard=.1,deleted_mass_guard=.2,minimum_feasible=4,error_match=.05,mass_match=.01,response_separation=.25,
        bank_max_candidates=48,diagnostic_hash_seed=19331,campaign_GPU_seconds_ceiling=None,campaign_model_forward_ceiling=None,
        stage_scope='CPU addendum, engineering qualification, A1 all24cells; A2 conditional on frozen exact gate; Stage B/C excluded',
        advance_gate_source='README.md and code/analyze.py pinned above',fresh_requests_only=True)
    with (root/'protocol.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(protocol_sha256=sha(root/'protocol.json'),source_count=len(sources),manifest=sha(root/'manifest_identity.json'))))


if __name__=='__main__':main()
