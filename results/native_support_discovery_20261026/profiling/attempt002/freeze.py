"""Source freeze for baseline profiling only; discovery has its own later freeze."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;REPO=R.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def main():
    old=REPO/'results/commit_boundary_single_pulse_20261026/panel/attempt004'
    reference=json.loads((old/'protocol.json').read_text());tracked=subprocess.check_output(['git','ls-files','--','experiments','scripts'],cwd=REPO,text=True).splitlines()
    files=[REPO/p for p in tracked if Path(p).suffix in ['.py','.cu','.cuh','.cpp','.h']]
    files += list((R/'code').glob('*.py'))+list((R/'tests').glob('*.py'))+[R/'SPEC.md',R/'roster_identity.json',R/'freeze.py',R/'launch.py',R.parents[1]/'experiment/attempt001/SPEC.md']
    source={str(p.relative_to(REPO)):sha(p) for p in files}
    save(R/'model_fingerprints.json',json.loads((old/'model_fingerprints.json').read_text()))
    save(R/'runtime_fingerprints.json',json.loads((old/'runtime_fingerprints.json').read_text()))
    save(R/'protocol.json',dict(schema='native_support_discovery_profiling_v1',sources=source,roster_sha256=sha(R/'roster_identity.json'),private_roster_sha256=sha(R/'private/roster.json'),runtime_source_pins=reference['runtime_source_pins'],scorer_sources=reference['scorer_sources'],model_revision=reference['model_revision'],native_sampler=reference['native_sampler'],native_output_caps=reference['native_output_caps'],arms=['corrected_dense_full','dense_piecewise','matched_allkept_l5_l17_l29'],seeds=[42,43],common_consumer=dict(layers=[5,17,29],bootstrap_calls=1,every_canvas=True,selector=False),no_campaign_resource_ceilings=True,model_fingerprint_sha256=sha(R/'model_fingerprints.json'),spec_sha256=sha(R/'SPEC.md')))
    print(sha(R/'protocol.json'))
if __name__=='__main__':main()
