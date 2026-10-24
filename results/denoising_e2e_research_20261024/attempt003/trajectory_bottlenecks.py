"""Supplement native confidence/stability/cap accounting; no tuning or text."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();out=root/'trajectory_analysis001';out.mkdir(exist_ok=False)
    rows=[];totals=defaultdict(Counter);inputs={}
    for file in sorted((root/'runs/panel').glob('*/*/audit[0-9][0-9][0-9].json')):
        value=json.loads(file.read_text());counts=Counter();canvas=[];canvases=[]
        for call in value['native_trace']:
            if call['commit']:
                if canvas:
                    terminal=canvas[-1]
                    canvases.append(dict(calls=len(canvas),cap=terminal['cap_reached'],
                        terminal_entropy=terminal['entropy_mean'],terminal_confident=terminal['confidence'],
                        terminal_stable=terminal['stable'],terminal_history_ready=terminal['native_history_ready']))
                canvas=[]
                continue
            canvas.append(call);counts['denoise_calls']+=1
            ready=call['native_history_ready'];stable=call['stable'];confident=call['confidence']
            if not ready:counts['history_unready']+=1
            elif stable and not confident:counts['confidence_only_bottleneck']+=1
            elif confident and not stable:counts['stability_only_bottleneck']+=1
            elif not stable and not confident:counts['both_bottlenecks']+=1
            else:counts['ready_stable_confident']+=1
            if call['cap_reached']:counts['cap_events']+=1
        uncommitted_tail_calls=len(canvas)
        assert len(canvases)==value['C'] and sum(x['calls'] for x in canvases)+uncommitted_tail_calls==value['N']
        intervals=value['native_interval_steps']['native_interval_steps']
        emission_intervals_equal=([x['calls'] for x in canvases]==intervals and not uncommitted_tail_calls)
        key=value['benchmark']+'|'+value['arm'];totals[key].update(counts)
        rows.append(dict(id=value['id'],seed=value['seed'],benchmark=value['benchmark'],arm=value['arm'],
            counts=dict(counts),canvases=canvases,uncommitted_tail_calls=uncommitted_tail_calls,emission_intervals_equal=emission_intervals_equal,cap_frequency=sum(x['cap'] for x in canvases)/len(canvases),
            first_canvas=canvases[0],terminal_canvas=canvases[-1]))
        inputs[str(file.relative_to(root))]=hashlib.sha256(file.read_bytes()).hexdigest()
    result=dict(rows=rows,totals={k:dict(v) for k,v in totals.items()},
        definition='Confidence-only requires native history readiness and stability; stability-only requires confidence. Both/unready remain separate.',
        scope='Native flags read after unchanged compiled sampler; no reconstructed acceptance claims.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'input_hashes.json').write_text(json.dumps(inputs,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(cells=len(rows),groups=len(totals))))


if __name__=='__main__':main()
