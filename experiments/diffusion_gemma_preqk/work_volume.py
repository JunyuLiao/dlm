"""Compare total decoder work across paired trajectories, not sparsity alone."""
from collections import Counter
import argparse
import json
from pathlib import Path
from .config import ROOT
from .report import csv_write
from .run import digest


def baseline_work(calls):
    counts=Counter(c['layer'] for c in calls)
    if not counts or len(set(counts.values()))!=1:raise ValueError('unequal or missing dense decoder layer coverage')
    return dict(steps=next(iter(counts.values())),eligible=sum(c['eligible_tiles'] for c in calls),
        # Shapes/dimensions are for the pinned DiffusionGemma revision only.
        qk_flops=sum(2*len(c['head_ids'])*c['query_length']*c['sequence_length']*
            (512 if c['attention_type']=='global' else 256) for c in calls))


def analyze(stage):
    setup=json.loads((stage/'freeze.json').read_text());dense={};rows=[]
    if 'dense_cache_manifest' in setup:setup['cached_dense']=json.loads(Path(setup['dense_cache_manifest']).read_text())
    for ident,cache in setup['cached_dense'].items():
        assert digest(cache['path'])==cache['sha256']
        data=json.loads(Path(cache['path']).read_text())
        if 'calls' in data:dense[ident]=baseline_work(data['calls'])
        else:
            layers={r['layer'] for r in data['records']}
            dense[ident]=dict(steps=data['attention_calls']/len(layers),eligible=data['costs']['eligible_tiles'],qk_flops=data['costs']['generation_qk_flops'])
    for name,entry in setup['conditions'].items():
        paired=[]
        for path in (stage/'conditions'/name/'shards').glob('*.json'):
            data=json.loads(path.read_text());baseline=dense[data['id']]
            layers={r['layer'] for r in data['records']}
            paired.append(dict(baseline,prompt_id=data.get('prompt_id',data['id']),router_steps=data['attention_calls']/len(layers),
                router_eligible=data['costs']['eligible_tiles'],router_retained=data['costs']['retained_tiles'],
                router_qk_flops=data['costs']['generation_qk_flops']))
        if not paired:continue
        total=lambda name:sum(r[name] for r in paired)
        rows.append(dict(condition=name,benchmark=entry['benchmark'],prompts=len({r['prompt_id'] for r in paired}),generations=len(paired),
            dense_steps=total('steps'),router_steps=total('router_steps'),
            denoising_step_ratio=total('router_steps')/total('steps'),
            dense_eligible_tiles=total('eligible'),router_eligible_tiles=total('router_eligible'),
            router_retained_tiles=total('router_retained'),
            retained_tile_volume_ratio=total('router_retained')/total('eligible'),
            dense_decoder_qk_flops=total('qk_flops'),router_decoder_qk_flops=total('router_qk_flops'),
            dense_emulation_decoder_qk_work_ratio=total('router_qk_flops')/total('qk_flops')))
    csv_write(stage/'trajectory_work.csv',rows)
    (stage/'trajectory_work_notes.md').write_text(
        '# Trajectory work accounting\n\n'
        'The retained-tile volume ratio compares sum(router retained tile instances) '
        'with sum(dense eligible tile instances) on paired prompts, including changes '
        'in sequence length and denoising duration. It is NOT executed sparse work.\n\n'
        'The decoder QK FLOP ratio counts actual generation matrix shapes in the '
        'dense-mask backend. Tile masking avoids no executed QK/PV; duration and '
        'sequence changes can still change total matmul work. PV has the same '
        'generation shape count for this pinned model. Additional diagnostic PV '
        'work is separately recorded in cost_breakdown.csv. Encoder/prefill, '
        'projections, MLP/MoE, metadata, and wall time are excluded here, so neither '
        'ratio is an end-to-end speedup.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,default=ROOT/'online_v1')
    analyze(parser.parse_args().stage)
