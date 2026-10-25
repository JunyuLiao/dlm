"""Freeze immutable R method, manifest, runtime and analysis contract."""
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent;repo=root.parents[3]
    old=repo/'results/denoising_e2e_research_20261024/attempt003'
    prior=json.loads((old/'e2e_protocol.json').read_text())
    manifest=json.loads((root/'manifest_identity.json').read_text())
    sources={str(Path(p).relative_to(repo)):v for p,v in prior['sources'].items() if Path(p).is_relative_to(repo)}
    assert all(sha(repo/p)==v for p,v in sources.items())
    external={p:v for p,v in prior['sources'].items() if not Path(p).is_relative_to(repo)}
    assert all(sha(Path(p))==v for p,v in external.items())
    for name in ['selector.py','online_runtime.py']:assert sha(root/'code'/name)==sha(old/'code'/name)
    (root/'private/external_sources.json').write_text(json.dumps(external))
    private_scorer={p:v for p,v in prior['scorer_sources'].items() if '/private/' in p}
    (root/'private/scorer_sources.json').write_text(json.dumps(private_scorer))
    assert all(sha(Path(p))==v for p,v in private_scorer.items())
    own=list(root.glob('*.py'))+list((root/'code').glob('*.py'))+list((root/'tests').glob('*.py'))+[root/'README.md']
    sources.update({str(p.relative_to(repo)):sha(p) for p in own})
    for name in ['model_fingerprints.json','runtime_fingerprints.json']:(root/name).write_bytes((old/name).read_bytes())
    arms=['dense_full_fix51994','allkept_layer29','current_error_fixed4','sham_fixed4','response_guard_qdirection']
    result=dict(schema='unchanged_response_behavior_replication_v1',starting_commit='3a1e6e9c903b3152012e84ac105854fbb09451ee',
        sources=sources,external_sources={p.split('site-packages/')[-1]:v for p,v in external.items()},
        runtime_source_pins=json.loads((old/'runtime_source_audit001.json').read_text())['files'],
        scorer_sources={str(Path(p).relative_to(repo)):v for p,v in prior['scorer_sources'].items() if '/private/' not in p},
        private_scorer_hashes=list(private_scorer.values()),model_revision=prior['model_revision'],
        model_fingerprints_sha256=sha(root/'model_fingerprints.json'),runtime_fingerprints_sha256=sha(root/'runtime_fingerprints.json'),
        method=prior['method'],native_sampler=prior['native_sampler'],runtime_settings={'short':prior['runtime_settings']['short']},
        arms=arms,legacy_arms=['response_guard_qdirection','dense_full_fix51994','allkept_layer29','current_error_fixed4','sham_fixed4','dense_piecewise'],
        legacy_reproduction=manifest['legacy_reproduction'],panel=manifest['panel'],manifest_sha256=sha(root/'manifest_identity.json'),
        private_manifest_sha256=sha(root/'private/manifest.json'),spec_sha256=sha(root/'README.md'),exposure_registry_sha256=sha(root/'exposure_registry.json'),
        legacy_method_sha256={n:sha(old/'code'/n) for n in ['selector.py','online_runtime.py']},
        primary_family_comparisons=8,bootstrap_draws=50000,bootstrap_seed=20261025,simultaneous_interval_confidence=.99375,
        resource_ceilings={'GPU_hours':None,'elapsed_seconds':None,'model_forwards':None,'cost':None,'hang_timeout':None},
        legacy_protocol_sha256=sha(old/'e2e_protocol.json'),legacy_source_checkpoint='4ac2b53f',stage_BC=False,
        source_freeze_order='spec+source+protocol checkpoint before any GPU; binding and tests before legacy; legacy gate before panel')
    with (root/'protocol.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
    print(json.dumps(dict(protocol_sha256=sha(root/'protocol.json'),source_pins=len(sources))))


if __name__=='__main__':main()
