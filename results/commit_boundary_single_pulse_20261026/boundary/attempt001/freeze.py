from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[3];OLD=REPO/'results/response_behavior_and_stopping_headroom_20261025/track_h/attempt001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((OLD/'protocol.json').read_text());sources=dict(p['sources'])
 assert all(sha(REPO/n)==h for n,h in sources.items())
 own=list((ROOT/'code').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+[ROOT/'README.md',ROOT/'freeze.py',ROOT/'launch.py',ROOT.parents[1]/'README.md']
 sources.update({str(x.relative_to(REPO)):sha(x) for x in own})
 p.update(schema='commit_boundary_observation_v1',sources=sources,historical_H_protocol_sha256=sha(OLD/'protocol.json'),
  private_manifest_sha256=sha(ROOT/'private/manifest.json'),resource_ceilings=None,exact_old_supports=True,observations_only=True,
  specimen_cells=[dict(id='longbench_v2/66ec2755821e116aacb1b6f2',seed=42),dict(id='longbench_v2/66ec2755821e116aacb1b6f2',seed=43),dict(id='longbench_v2/6724645dbb02136c067d73ad',seed=43)])
 with (ROOT/'protocol.json').open('x') as f:json.dump(p,f,indent=2,sort_keys=True)
 print(json.dumps(dict(protocol_sha256=sha(ROOT/'protocol.json'),sources=len(sources))))
if __name__=='__main__':main()
