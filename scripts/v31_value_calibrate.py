"""Development generation calibration, with immutable trial directories.

Only achieved GLOBAL physical sparsity chooses a uniform threshold. No gold,
scorer result or target generation is read. No quota is added to the selector.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


ARMS = ('value_v1_online_discard_mass','value_v2_online_preserve_mass')


def main():
    p=argparse.ArgumentParser()
    for key in ('python','model','manifests','cells','reference','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--initial-threshold',type=float,default=.001)
    p.add_argument('--tolerance',type=float,default=.003)
    p.add_argument('--max-trials',type=int,default=12)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=Path(args.output)
    if output.exists():
        raise ValueError('new calibration directory required')
    output.mkdir(parents=True)
    reference=json.loads(Path(args.reference).read_text())
    target=reference['global_sparsity']
    if reference['items']!=len(json.loads(Path(args.cells).read_text())):
        raise ValueError('reference and development schedule size mismatch')
    config=dict(kind='development generation only',target_global_sparsity=target,
        tolerance_absolute=args.tolerance,max_trials_per_arm=args.max_trials,
        initial_threshold=args.initial_threshold,
        cells_sha256=hashlib.sha256(Path(args.cells).read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
        selection='first trial within tolerance; otherwise no freeze',
        threshold_scope='one uniform GLOBAL threshold per arm, no quota')
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    history={arm:[] for arm in ARMS}
    frozen={}
    start=time.perf_counter()
    for arm in ARMS:
        threshold=args.initial_threshold
        for trial in range(1,args.max_trials+1):
            attempt=output/arm/f'trial{trial:03d}'
            cmd=[args.python,str(root/'scripts/v31_value_campaign.py'),'--python',args.python,
                '--model',args.model,'--manifests',args.manifests,'--cells',args.cells,
                '--attempt',str(attempt),'--arm',arm,'--threshold',str(threshold),'--purpose','development']
            subprocess.run(cmd,cwd=root,check=True)
            rows=[json.loads(x) for x in (attempt/'records.jsonl').read_text().splitlines()]
            eligible=sum(r['receipts']['adapter']['global_eligible_tiles'] for r in rows)
            kept=sum(r['receipts']['adapter']['global_kept_tiles'] for r in rows)
            achieved=1-kept/eligible
            entry=dict(trial=trial,threshold=threshold,global_sparsity=achieved,
                delta=achieved-target,eligible_tiles=eligible,kept_tiles=kept,
                source_commit=json.loads((attempt/'config.json').read_text())['source_commit'])
            history[arm].append(entry)
            (output/'progress.json').write_text(json.dumps(history,indent=2)+'\n')
            print(json.dumps(dict(arm=arm,**entry)),flush=True)
            if abs(entry['delta'])<=args.tolerance:
                frozen[arm]=threshold
                break
            # Real generation may be nonmonotonic. Brackets are observations,
            # not a monotonicity claim; the measured tolerance remains the gate.
            below=[x for x in history[arm] if x['delta']<0]
            above=[x for x in history[arm] if x['delta']>0]
            if below and above:
                pairs=[(lo,hi) for lo in below for hi in above if lo['threshold']<hi['threshold']]
                if pairs:
                    lo,hi=min(pairs,key=lambda pair:math.log(pair[1]['threshold']/pair[0]['threshold']))
                    threshold=math.sqrt(lo['threshold']*hi['threshold'])
                else:
                    threshold *= math.exp(-entry['delta'])
            else:
                threshold *= 3 if entry['delta']<0 else 1/3
        if arm not in frozen:
            (output/'search_incomplete.json').write_text(json.dumps(dict(arm=arm,history=history),indent=2)+'\n')
            raise RuntimeError('measured match not reached; target launch remains gated')
    result=dict(status='complete',thresholds=frozen,history=history,
        config_sha256=hashlib.sha256((output/'config.json').read_bytes()).hexdigest(),
        elapsed_s=time.perf_counter()-start,
        label='development-only calibration; historically examined official pool')
    (output/'calibration_complete.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
