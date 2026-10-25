from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[3];OLD=REPO/'results/response_behavior_and_stopping_headroom_20261025/track_h/attempt001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((OLD/'protocol.json').read_text());sources=dict(p['sources']);assert all(sha(REPO/n)==h for n,h in sources.items())
 own=list((ROOT/'code').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+[ROOT/'freeze.py',ROOT/'launch.py',ROOT.parents[1]/'README.md',ROOT/'roster_identity.json']
 sources.update({str(x.relative_to(REPO)):sha(x) for x in own})
 p.update(schema='single_pulse_independent_eligibility_online_v2',sources=sources,roster_sha256=sha(ROOT/'roster_identity.json'),private_roster_sha256=sha(ROOT/'private/roster.json'),private_manifest_sha256=None,
  resource_ceilings=None,online_selection=True,CE_sham_requires_directional_pair=False,clean_optional_measurements=False,selector_policies_fixed=True,
  arms=['corrected_dense_full','dense_piecewise','matched_allkept_layer29','current_error_single_pulse','support_sham_single_pulse','direction_high_single_pulse','direction_low_single_pulse'],
  scoring='Original bound official AIME, HumanEval harness and LongBench extraction; private outputs only',bootstrap=dict(draws=20000,seed=20261026,confidence=.95,clusters='question retaining paired seeds'),
  historical_H_protocol_sha256=sha(OLD/'protocol.json'))
 with (ROOT/'protocol.json').open('x') as f:json.dump(p,f,indent=2,sort_keys=True)
 print(json.dumps(dict(protocol_sha256=sha(ROOT/'protocol.json'),sources=len(sources))))
if __name__=='__main__':main()
