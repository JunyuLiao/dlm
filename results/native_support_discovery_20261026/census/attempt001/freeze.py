"""Freeze CPU leaf sources and exact immutable input hashes before readout."""
from pathlib import Path
import hashlib,json
L=Path(__file__).resolve().parent;REPO=L.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((L/'input_manifest.json').read_text());assert manifest['total']==368 and manifest['counts']==dict(R=280,C=72,H=16)
    inputs={str((L/'input_manifest.json').relative_to(REPO)):sha(L/'input_manifest.json')}
    for x in manifest['records']:
        assert sha(REPO/x['path'])==x['sha256'];inputs[x['path']]=x['sha256']
        if 'quality_source' in x:
            assert sha(REPO/x['quality_source'])==x['quality_source_sha256'];inputs[x['quality_source']]=x['quality_source_sha256']
    paths=[
        'results/response_behavior_and_stopping_headroom_20261025/track_r/attempt001/protocol.json',
        'results/response_behavior_and_stopping_headroom_20261025/track_h/attempt001/protocol.json',
        'results/commit_boundary_single_pulse_20261026/panel/attempt004/protocol.json',
        'results/response_behavior_and_stopping_headroom_20261025/track_r/attempt001/manifest_identity.json',
        'results/response_behavior_and_stopping_headroom_20261025/track_h/attempt001/manifest_identity.json',
        'results/commit_boundary_single_pulse_20261026/panel/attempt004/roster_identity.json',
        'results/commit_boundary_single_pulse_20261026/panel/attempt004/manifest_identity.json',
        'results/response_behavior_and_stopping_headroom_20261025/track_r/attempt001/code/worker.py',
        'results/response_behavior_and_stopping_headroom_20261025/track_h/attempt001/code/worker.py',
        'results/commit_boundary_single_pulse_20261026/panel/attempt004/code/worker.py',
    ]
    review=REPO/'results/commit_boundary_single_pulse_review_20261026/review001'
    paths += [str(p.relative_to(REPO)) for p in review.glob('*.json') if p.name in ['claim_audit.json','engineering_and_measurement_audit.json','recomputed_summary.json','preliminary_C_opportunities.json','next_cpu_input_manifest.json','review_receipt.json']]
    paths += [str((review/p).relative_to(REPO)) for p in ['REVIEW.md','NEXT_PROTOCOL.md']]
    for p in paths:inputs[p]=sha(REPO/p)
    sources={str(p.relative_to(REPO)):sha(p) for p in [L/'SPEC.md',L/'freeze.py',L/'code/census.py',L/'tests/test_census.py',L.parents[1]/'README.md']}
    v=dict(schema='saved_native_control_census_v1',inputs=inputs,sources=sources,records=368,request_counts=manifest['counts'],bootstrap_draws=20000,bootstrap_seed=20261026,no_new_GPU_or_operator_calls=True,no_campaign_resource_ceilings=True)
    with (L/'protocol.json').open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
    print(sha(L/'protocol.json'))
if __name__=='__main__':main()
