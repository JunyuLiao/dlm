"""Whole-layer replay after the bounded single-head screen.

Sum signed head errors through the actual O projection before taking norms.
This catches cross-head cancellation that headwise scores cannot represent.
The failed historical-query extrapolation is not promoted into this stage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import time

import torch

from scripts import v31_value_decision_replay as base
from experiments.numerical_qk_reuse.v31_value_decision import (
    OmittedCache, bounded_proposal, fixed_topk, mass_score, moments,
)


def prepare(study):
    parent = study.parent
    source = base.verify(parent)
    if not (parent / 'operator_complete.json').exists():
        raise RuntimeError('single-head screen is incomplete')
    protocol = dict(source, schema='value_decision_reuse_whole_layer_v1', heads=list(range(16)), query_tiles=[0, 1],
        stage='all_head_both_query_tiles_cross_head_projection',
        parent_complete_sha256=base.sha(parent / 'operator_complete.json'),
        directions=['full_swap8', 'online_swap8', 'correction_residual', 'correction_moments'],
        direction2='past-query extrapolation rejected: age-1 prediction worse on every nontrivial screened unit',
        C_gate='neutral; historical snapshots have no sampler state', timing='paused')
    protocol['source_sha256'] = dict(source['source_sha256'], **{str(Path(__file__).relative_to(base.ROOT)): base.sha(Path(__file__))})
    protocol['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    base.write(study / 'operator_protocol.json', protocol, True)


@torch.inference_mode()
def run(study):
    protocol = base.verify(study)
    if not torch.cuda.is_available():
        raise RuntimeError('pinned H100 required')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    for path, digest in base.read(base.SOURCE / 'specs/real_snapshot_inventory.json')['sha256'].items():
        assert base.sha(base.SOURCE / path) == digest
    for path, digest in protocol['model_shards_sha256'].items():
        assert base.sha(base.MODEL / path) == digest
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(8)
    from safetensors import safe_open
    index = base.read(base.MODEL / 'model.safetensors.index.json')['weight_map']
    matrices, grams = {}, {}
    for layer in (5, 11, 17, 23, 29):
        name = f'model.decoder.layers.{layer}.self_attn.o_proj.weight'
        with safe_open(str(base.MODEL / index[name]), framework='pt', device='cpu') as f:
            w = f.get_tensor(name).float().cuda()
        for h in range(16):
            matrices[layer, h] = w[:, h * 512:(h + 1) * 512].contiguous()
            grams[layer, h] = matrices[layer, h].T @ matrices[layer, h]
    captures = dict(count=0)
    capture = torch.cuda.CUDAGraph.capture_begin
    def counted(graph, *a, **kw):
        captures['count'] += 1
        return capture(graph, *a, **kw)
    torch.cuda.CUDAGraph.capture_begin = counted
    monitor = time.monotonic()
    for group, files in sorted(base.group_paths().items()):
        name = '_'.join(group); family = study / 'operator' / name
        done = sorted(family.glob('attempt*/complete.json'))
        if done:
            marker = base.read(done[-1])
            assert marker['protocol_sha256'] == base.sha(study / 'operator_protocol.json')
            assert marker['records_sha256'] == base.sha(done[-1].parent / 'records.json')
            continue
        used = [int(p.name[7:]) for p in family.glob('attempt*') if p.is_dir()]
        leaf = family / f'attempt{max(used, default=0) + 1:03d}'
        leaf.mkdir(parents=True); start = time.monotonic()
        base.write(study / 'state.json', dict(stage='running', group=name, timing='paused'))
        states, records, prefix = {}, [], None
        for path in files:
            data = torch.load(path, weights_only=True, map_location='cpu')
            if prefix is None:
                prefix = torch.load(path.parent / data['prefix_file'], weights_only=True, map_location='cuda')
            keys = torch.cat((prefix['k'], data['tail_k'].cuda()), 2)[0]
            values = torch.cat((prefix['v'], data['tail_v'].cuda()), 2)[0]
            signed = {}; swaps = defaultdict(int); masses = defaultdict(list); head_errors = defaultdict(list)
            counters = defaultdict(int)
            for h in range(16):
                for qt in (0, 1):
                    unit = (h, qt); anchor_event = data['selection_event']
                    q = data['q'][0, h, qt * 128:min((qt + 1) * 128, data['n'])].cuda().float()
                    if len(q) == 0 or (unit not in states and not anchor_event):
                        continue
                    kh = h // (16 // len(keys)); layer = data['layer']
                    m = moments(q, keys[kh], values[kh], data['prefix'], data['scale'])
                    k = min(len(m.mass), 27 if data['dataset'] == 'aime26' else 128)
                    if anchor_event:
                        if data['dataset'] == 'aime26':
                            score = mass_score(m).clamp_min(1e-30).log()
                            if unit in states:
                                score += 1.386 * states[unit]['masks']['mass_anchor']
                            anchor = fixed_topk(score, k)
                        else:
                            anchor = data['next'][0, h, qt, :len(m.mass)].cuda().bool()
                        assert int(anchor.sum()) == k
                        masks = {'mass_anchor': anchor}; infos = {'mass_anchor': {'swaps': 0}}
                        for label, online in [('full_swap8', False), ('online_swap8', True)]:
                            masks[label], infos[label] = bounded_proposal(m, anchor, swaps=8, online=online, gram=grams[layer, h])
                        states[unit] = dict(masks=masks, infos=infos, cache=OmittedCache.build(m, anchor), age=0)
                    else:
                        states[unit]['age'] += 1
                    st = states[unit]; dense = m.dense(); anchor = st['masks']['mass_anchor']
                    outputs = {label: m.output(mask) for label, mask in st['masks'].items()}
                    for kind in ('residual', 'moments'):
                        outputs['correction_' + kind] = st['cache'].reconstruct(m, anchor, kind)
                    for label, output in outputs.items():
                        mask = st['masks'].get(label, anchor)
                        assert int(mask.sum()) == k
                        projected_error = (output - dense) @ matrices[layer, h].T
                        if label not in signed:
                            signed[label] = torch.zeros((data['n'], matrices[layer, h].shape[0]), device=q.device)
                        signed[label][qt * 128:qt * 128 + len(q)] += projected_error
                        swaps[label] += st['infos'].get(label, {'swaps': 0})['swaps']
                        masses[label].append(m.selected(mask)[1])
                        head_errors[label].append((output - dense).norm(dim=-1) / m.value_rms)
                        counters[label] += 1
                    del m, outputs
            for label, errors in signed.items():
                e = errors.norm(dim=-1); pre = torch.cat(head_errors[label]); mass = torch.cat(masses[label])
                records.append(dict(request=group[0], dataset=data['dataset'], layer=data['layer'], canvas=data['canvas'],
                    call=data['call'], age=states[0, 0]['age'], arm=label, units=counters[label], rows=len(e),
                    budget=k, prefix_tiles=data['prefix'] // 64, mean_post_o_error=float(e.mean()),
                    p95_post_o_error=float(e.quantile(.95)), max_post_o_error=float(e.max()),
                    mean_full_v_error=float(pre.mean()), max_full_v_error=float(pre.max()),
                    mean_retained_mass=float(mass.mean()), mean_swaps=swaps[label]/counters[label]))
            if time.monotonic() - monitor >= 1800:
                print(json.dumps(dict(event='progress', group=name, call=data['call'])), flush=True)
                monitor = time.monotonic()
        base.write(leaf / 'records.json', records, True)
        base.write(leaf / 'complete.json', dict(status='complete', records=len(records), input_frames=len(files),
            records_sha256=base.sha(leaf / 'records.json'), protocol_sha256=base.sha(study / 'operator_protocol.json'),
            budget_errors=0, cuda_captures=captures['count'], gpu_reservation_s=time.monotonic()-start), True)
        print(json.dumps(dict(event='group_complete', group=name, records=len(records))), flush=True)
        del states, prefix
        torch.cuda.empty_cache()
    summarize(study)


def summarize(study):
    protocol = base.verify(study); allrows = []; complete = 0
    for group in protocol['groups']:
        done = sorted((study / 'operator' / group).glob('attempt*/complete.json'))
        if not done:
            continue
        mark = base.read(done[-1]); assert mark['records_sha256'] == base.sha(done[-1].parent / 'records.json')
        assert mark['protocol_sha256'] == base.sha(study / 'operator_protocol.json')
        allrows += base.read(done[-1].parent / 'records.json'); complete += 1
    buckets = defaultdict(list)
    anchors = {(r['request'], r['canvas'], r['call'], r['layer']): r for r in allrows if r['arm']=='mass_anchor'}
    for r in allrows:
        if r['age'] == 0 or r['budget'] == r['prefix_tiles']:
            continue
        buckets[r['dataset'], r['request'], r['arm']].append(r)
    metrics = []
    for (dataset, request, arm), rows in sorted(buckets.items()):
        refs = [anchors[r['request'], r['canvas'], r['call'], r['layer']] for r in rows]
        metrics.append(dict(dataset=dataset, request=request, arm=arm, frames=len(rows),
            mean_post_o_error=sum(r['mean_post_o_error'] for r in rows)/len(rows),
            relative_mean_error=sum(r['mean_post_o_error'] for r in rows)/sum(r['mean_post_o_error'] for r in refs),
            max_post_o_error=max(r['max_post_o_error'] for r in rows),
            frames_worse=sum(r['mean_post_o_error']>a['mean_post_o_error']+1e-6 for r,a in zip(rows,refs)),
            frames_max_worse_2pct=sum(r['max_post_o_error']>a['max_post_o_error']*1.02+1e-6 for r,a in zip(rows,refs))))
    base.write(study / 'summary.json', dict(status='complete' if complete==len(protocol['groups']) else 'partial',
        groups_complete=complete, groups_expected=len(protocol['groups']), records=len(allrows), sparse_reuse_only=True, metrics=metrics,
        decision_gate='not evaluated', native_accuracy_launched=False, timing='paused'))
    if complete == len(protocol['groups']):
        base.write(study / 'operator_complete.json', dict(status='complete', scope=protocol['stage'],
            protocol_sha256=base.sha(study/'operator_protocol.json'), summary_sha256=base.sha(study/'summary.json'),
            model_decision_gate='pending', whole_study_complete=False), True)
    base.write(study / 'state.json', dict(stage='complete' if complete==len(protocol['groups']) else 'partial',
        groups_complete=complete, timing='paused'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['prepare','run','summary'],required=True)
    p.add_argument('--study',type=Path,default=base.DEFAULT/'whole_layer')
    a=p.parse_args(); {'prepare':prepare,'run':run,'summary':summarize}[a.stage](a.study)
