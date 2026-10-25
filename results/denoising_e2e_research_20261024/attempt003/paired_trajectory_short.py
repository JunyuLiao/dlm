"""Supplement paired N/C, output-length and amortized-cost uncertainty."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();out=root/'paired_trajectory_short001';out.mkdir(exist_ok=False)
    protocol=json.loads((root/'e2e_protocol.json').read_text())
    scores=json.loads((root/'scoring_short002/official_scores.json').read_text())
    quality={(x['arm'],x['id'],x['seed']):x['correct'] for x in scores['outcomes']}
    data=defaultdict(dict);inputs={}
    for p in (root/'runs/panel/short').glob('*/clean[0-9][0-9][0-9].json'):
        v=json.loads(p.read_text());audit=json.loads(p.with_name(p.name.replace('clean','audit')).read_text());tail=0
        for event in audit['native_trace']:
            if event['commit']:tail=0
            else:tail+=1
        v['native_uncommitted_tail_calls']=tail
        v['committed_N_per_C']=(v['N']-tail)/v['C']
        data[v['arm']][v['id'],v['seed']]=v
        inputs[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    result=[];rng=np.random.default_rng(20261024)
    for task in ['aime26','humaneval','longbench_v2']:
        for reference in ['dense_full_fix51994','dense_piecewise','allkept_layer29','v31_optimized','current_error_fixed4','sham_fixed4']:
            for arm in protocol['arms']:
                common=sorted(set(data[arm])&set(data[reference]))
                common=[k for k in common if data[arm][k]['benchmark']==task]
                ids=sorted({k[0] for k in common})
                if not ids:continue
                draws=rng.integers(0,len(ids),size=(10000,len(ids)))
                row=dict(task=task,arm=arm,reference=reference,cells=len(common),item_clusters=len(ids),metrics={},items=[])
                for field in ['N','C','N_per_C','committed_N_per_C','T','W','S_per_N']:
                    values=[];deltas=[]
                    for identity in ids:
                        keys=[k for k in common if k[0]==identity]
                        values.append(statistics.mean(math.log(data[arm][k][field]/data[reference][k][field]) for k in keys))
                        deltas.append(statistics.mean(data[arm][k][field]-data[reference][k][field] for k in keys))
                    v=np.array(values);d=np.array(deltas)
                    row['metrics'][field]=dict(geometric_ratio=float(np.exp(v.mean())),
                        ratio_ci95=np.exp(np.percentile(v[draws].mean(1),[2.5,97.5])).tolist(),
                        arithmetic_paired_difference=float(d.mean()),difference_ci95=np.percentile(d[draws].mean(1),[2.5,97.5]).tolist())
                row['timing_qualified']=all(data[a][k]['timing_valid'] for a in [arm,reference] for k in common)
                for identity in ids:
                    keys=[k for k in common if k[0]==identity]
                    row['items'].append(dict(id=identity,seeds=[k[1] for k in keys],
                        both_seeds_fewer_calls=all(data[arm][k]['N']<data[reference][k]['N'] for k in keys),
                        both_seeds_lower_N_per_C=all(data[arm][k]['N_per_C']<data[reference][k]['N_per_C'] for k in keys),
                        reference_correct_candidate_wrong=sum(quality[reference,*k] and not quality[arm,*k] for k in keys),
                        reference_wrong_candidate_correct=sum(not quality[reference,*k] and quality[arm,*k] for k in keys),
                        candidate_scores=[bool(quality[arm,*k]) for k in keys],reference_scores=[bool(quality[reference,*k]) for k in keys]))
                result.append(row)
    value=dict(scope='Completed short leaf; long pending',whole_panel_complete=False,comparisons=result,bootstrap='10000 paired question-cluster draws retaining both seeds; no population noninferiority claim',
        purpose='Report requested normalized native trajectory and output-length endpoints; no score tuning or changed promotion thresholds',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    (out/'input_hashes.json').write_text(json.dumps(inputs,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(comparisons=len(result))))


if __name__=='__main__':main()
