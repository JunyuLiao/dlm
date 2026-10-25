"""Separate H source/protocol binding, only after frozen R readout."""
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent;repo=root.parents[3];study=root.parents[1]
    r=study/'track_r/attempt001';assert (r/'readout_frozen.json').exists()
    rp=json.loads((r/'protocol.json').read_text());sources=dict(rp['sources'])
    assert all(sha(repo/p)==h for p,h in sources.items())
    own=list(root.glob('*.py'))+list((root/'code').glob('*.py'))+list((root/'tests').glob('*.py'))+[root/'README.md',repo/'results/denoising_mechanism_stage_a_20261025/attempt002/code/operators.py']
    sources.update({str(p.relative_to(repo)):sha(p) for p in own})
    closure=json.loads((r/'scorer_source_closure001.json').read_text())
    for entry in [closure['final_response_source'],closure['numeric_score_source']]:
        assert sha(repo/entry['file'])==entry['sha256'];sources[entry['file']]=entry['sha256']
    for n in ['runtime_paths.json','external_sources.json','scorer_sources.json']:(root/'private'/n).write_bytes((r/'private'/n).read_bytes())
    for n in ['model_fingerprints.json','runtime_fingerprints.json']:(root/n).write_bytes((r/n).read_bytes())
    result=dict(schema='later_canvas_signed_bias_headroom_v1',sources=sources,external_sources=rp['external_sources'],scorer_sources=rp['scorer_sources'],
        model_revision=rp['model_revision'],runtime_source_pins=rp['runtime_source_pins'],native_sampler=rp['native_sampler'],
        R_readout_sha256=sha(r/'readout_frozen.json'),R_protocol_sha256=sha(r/'protocol.json'),R_outcomes_used_for_H_selection=False,
        spec_sha256=sha(root/'README.md'),manifest_sha256=sha(root/'manifest_identity.json'),private_manifest_sha256=sha(root/'private/manifest.json'),
        seeds=[42,43],layer=29,heads=[0,8],queryblocks=[0,1],geometry=[128,64],physical_delete='max(1,floor(.05B)) per unit',
        anchor='first post-first canvas after2consecutive ready/stable/nonconfident/nonconverged calls, step<=46; next ordinary call',
        error_guard=.1,mass_guard=.2,error_matching=.05,mass_matching=.01,relative_error_floor=1e-6,minimum_bias_Frobenius=1e-6,
        maximum_varied_unit_bias_cosine=.5,full_response_norm_separation_not_required=True,native_deleting_consumer_error_bound=.03,
        current_only_direction_rules=['direction_high','direction_low'],native_output_cap=16384,one_pulse=True,
        resource_ceilings=None,Stage_BC=False,production_optimization=False)
    with (root/'protocol.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
    print(json.dumps(dict(protocol_sha256=sha(root/'protocol.json'),source_pins=len(sources))))


if __name__=='__main__':main()
