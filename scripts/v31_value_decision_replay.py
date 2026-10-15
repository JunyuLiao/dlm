"""Frozen, resumable same-state screen of three independent value directions.

No native generation, model decisions, or timing benchmark is performed here.
Saved trajectories are immutable and no future state is passed to selection.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from experiments.numerical_qk_reuse.v31_value_decision import (
    OmittedCache, bounded_proposal, fixed_topk, mass_score, moments, squared_rows,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'results/v31_value_decision_reuse_20261015'
SOURCE = ROOT / 'results/v31_cached_refresh_stable_20261013'
MODEL = Path('/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b')
CODE = ['experiments/numerical_qk_reuse/v31_value_decision.py',
        'scripts/v31_value_decision_replay.py', 'tests/test_v31_value_decision.py',
        'docs/V31_VALUE_DECISION_REUSE_PLAN_20261015.md']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists() and immutable:
        if path.read_text() != payload:
            raise RuntimeError(f'preserve immutable result: {path.name}')
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(payload); temporary.replace(path)


def prepare(study):
    if (study / 'operator_protocol.json').exists():
        verify(study)
        return
    study.mkdir(parents=True, exist_ok=True)
    (study / 'private').mkdir(exist_ok=True, mode=0o700)
    binding = read(SOURCE / 'operator/trajectory_binding.json')
    trajectory = SOURCE / binding['completed_leaf'] / 'private/trajectories'
    sources = sorted(trajectory.glob('request*/canvas*_call*_layer*.pt'))
    groups = sorted({(f.parent.name, f.name.split('_')[0], f.name.split('_')[-1][:-3]) for f in sources})
    model_index = read(MODEL / 'model.safetensors.index.json')['weight_map']
    model_files = {model_index[f'model.decoder.layers.{layer}.self_attn.o_proj.weight'] for layer in (5, 11, 17, 23, 29)}
    protocol = dict(schema='value_decision_reuse_operator_v1', source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={p: sha(ROOT / p) for p in CODE},
        snapshot_inventory_sha256=sha(SOURCE / 'specs/real_snapshot_inventory.json'),
        trajectory_binding_sha256=sha(SOURCE / 'operator/trajectory_binding.json'),
        model_shards_sha256={p: sha(MODEL / p) for p in sorted(model_files)},
        heads=[0, 8], query_tiles=[0], query_tile_size=128, kv_tile_size=64,
        long_budget=128, short_budget=27, extra_mass_neighbors=32,
        shortlist=16, swap_caps=[2, 8], online_threshold=.021,
        mass_guard=.025, max_error_ratio=1.02,
        groups=['_'.join(g) for g in groups],
        design_requests=['request001', 'request003'], evaluation_requests=['request002', 'request004'],
        split='Frozen by request parity before new measurements; four previously examined trajectories, not held out benchmark data',
        stage='single_head_single_query_tile_screen', C_gate='unavailable in historical QKV; neutral weights only; no decision gate',
        score='exact headwise O-projection Gram; cross-head cancellation deferred to whole-layer diagnostic',
        timing='paused; wall reservation accounting only; no CUDA timing events',
        native_promotion='requires all-head and same-state model decision gates; this replay alone cannot launch accuracy')
    write(study / 'operator_protocol.json', protocol, True)
    write(study / 'state.json', dict(stage='operator_prepared', timing='paused', native_accuracy_launched=False))


def verify(study):
    p = read(study / 'operator_protocol.json')
    for path, digest in p['source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError(f'frozen implementation changed: {path}')
    if sha(SOURCE / 'specs/real_snapshot_inventory.json') != p['snapshot_inventory_sha256']:
        raise RuntimeError('snapshot inventory changed')
    if sha(SOURCE / 'operator/trajectory_binding.json') != p['trajectory_binding_sha256']:
        raise RuntimeError('trajectory binding changed')
    return p


def group_paths():
    binding = read(SOURCE / 'operator/trajectory_binding.json')
    leaf = SOURCE / binding['completed_leaf']
    for name in ('config', 'receipt'):
        assert sha(leaf / f'{name}.json') == binding[f'{name}_sha256']
    groups = collections.defaultdict(list)
    for f in sorted((leaf / 'private/trajectories').glob('request*/canvas*_call*_layer*.pt')):
        groups[(f.parent.name, f.name.split('_')[0], f.name.split('_')[-1][:-3])].append(f)
    return groups


def metric(output, dense, gram, nu):
    err = output - dense
    pre = err.norm(dim=-1) / nu.clamp_min(1e-30)
    post = squared_rows(err, gram).sqrt()
    return dict(full_v_mean=float(pre.mean()), full_v_p95=float(pre.quantile(.95)), full_v_max=float(pre.max()),
                post_o_mean=float(post.mean()), post_o_p95=float(post.quantile(.95)), post_o_max=float(post.max()))


def replay(study, limit_groups=None):
    protocol = verify(study)
    if not torch.cuda.is_available():
        raise RuntimeError('pinned H100 required')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if gpu:
        raise RuntimeError('GPU occupied; do not share a worker')
    # Full file verification is separate from replay, once per invocation.
    inventory = read(SOURCE / 'specs/real_snapshot_inventory.json')
    for path, digest in inventory['sha256'].items():
        if sha(SOURCE / path) != digest:
            raise RuntimeError('private snapshot content changed')
    for path, digest in protocol['model_shards_sha256'].items():
        if sha(MODEL / path) != digest:
            raise RuntimeError('model weight source changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(8)
    from safetensors import safe_open
    model_index = read(MODEL / 'model.safetensors.index.json')['weight_map']
    grams = {}
    for layer in (5, 11, 17, 23, 29):
        name = f'model.decoder.layers.{layer}.self_attn.o_proj.weight'
        with safe_open(str(MODEL / model_index[name]), framework='pt', device='cpu') as f:
            w = f.get_tensor(name).float()
        for h in protocol['heads']:
            part = w[:, h * 512:(h + 1) * 512].cuda()
            grams[layer, h] = part.T @ part
    captures = dict(count=0)
    original = torch.cuda.CUDAGraph.capture_begin
    def count_capture(graph, *args, **kwargs):
        captures['count'] += 1
        return original(graph, *args, **kwargs)
    torch.cuda.CUDAGraph.capture_begin = count_capture
    started = time.monotonic(); last_monitor = started
    groups = group_paths()
    completed = 0
    for group, files in sorted(groups.items()):
        group_name = '_'.join(group)
        family = study / 'operator' / group_name
        done = sorted(family.glob('attempt*/complete.json'))
        if done:
            marker = read(done[-1])
            assert marker['protocol_sha256'] == sha(study / 'operator_protocol.json')
            assert marker['records_sha256'] == sha(done[-1].parent / 'records.json')
            continue
        if limit_groups is not None and completed >= limit_groups:
            break
        attempts = [int(f.name[7:]) for f in family.glob('attempt*') if f.is_dir()]
        leaf = family / f'attempt{max(attempts, default=0) + 1:03d}'
        leaf.mkdir(parents=True)
        write(study / 'state.json', dict(stage='operator_running', group=group_name, attempt=str(leaf.relative_to(study)),
                                       native_accuracy_launched=False, timing='paused'))
        prefix = None; records = []; states = {}; previous_q = {}; selection_counts = collections.Counter()
        group_start = time.monotonic()
        for path in files:
            data = torch.load(path, map_location='cpu', weights_only=True)
            if prefix is None:
                prefix = torch.load(path.parent / data['prefix_file'], map_location='cuda', weights_only=True)
            layer = int(data['layer'])
            keys = torch.cat((prefix['k'], data['tail_k'].cuda()), 2)[0]
            values = torch.cat((prefix['v'], data['tail_v'].cuda()), 2)[0]
            for h in protocol['heads']:
                for qt in protocol['query_tiles']:
                    unit = (h, qt)
                    q = data['q'][0, h, qt * 128:min((qt + 1) * 128, data['n'])].cuda().float()
                    if len(q) == 0:
                        continue
                    dq = q - previous_q[unit] if unit in previous_q else torch.zeros_like(q)
                    previous_q[unit] = q.clone()
                    anchor_event = bool(data['selection_event'])
                    if unit not in states and not anchor_event:
                        continue
                    kvh = h // (data['q'].shape[1] // keys.shape[0])
                    m = moments(q, keys[kvh], values[kvh], data['prefix'], data['scale'], dq if anchor_event else None)
                    pt = len(m.mass)
                    budget = min(pt, protocol['short_budget'] if data['dataset'] == 'aime26' else protocol['long_budget'])
                    gram = grams[layer, h]
                    if anchor_event:
                        if data['dataset'] != 'aime26':
                            anchor = data['next'][0, h, qt, :pt].cuda().bool()
                            assert int(anchor.sum()) == budget
                        else:
                            score = mass_score(m).clamp_min(1e-30).log()
                            if unit in states:
                                score = score + 1.386 * states[unit]['masks']['mass_anchor']
                            anchor = fixed_topk(score, budget)
                        masks = {'mass_anchor': anchor}
                        infos = {'mass_anchor': dict(swaps=0, pairs=0)}
                        for cap in protocol['swap_caps']:
                            for label, life, online in [('full', False, False), ('online', False, True), ('lifetime', True, False)]:
                                name = f'{label}_swap{cap}'
                                masks[name], infos[name] = bounded_proposal(m, anchor, swaps=cap, gram=gram,
                                    extra=protocol['extra_mass_neighbors'], shortlist=protocol['shortlist'], lifetime=life, online=online,
                                    mass_guard=protocol['mass_guard'], max_error_ratio=protocol['max_error_ratio'])
                        dense = m.dense()
                        states[unit] = dict(masks=masks, infos=infos, age=0, q=q.clone(),
                            errors={name: m.output(mask) - dense for name, mask in masks.items()},
                            derivatives={name: m.tangent(mask) - m.tangent(torch.ones_like(mask)) for name, mask in masks.items()},
                            correction=OmittedCache.build(m, anchor))
                        selection_counts[unit] += 1
                    else:
                        states[unit]['age'] += 1
                    st = states[unit]; dense = m.dense(); anchor = st['masks']['mass_anchor']
                    outputs = {name: m.output(mask) for name, mask in st['masks'].items()}
                    for kind in ('residual', 'moments', 'tangent'):
                        outputs[f'correction_{kind}'] = st['correction'].reconstruct(m, anchor, kind, min(st['age'], 1))
                    for name, out in outputs.items():
                        mask = st['masks'].get(name, anchor)
                        assert int(mask.sum()) == budget
                        info = st['infos'].get(name, st['infos']['mass_anchor'])
                        metrics = metric(out, dense, gram, m.value_rms)
                        row = dict(arm=name, request=group[0], dataset=data['dataset'], layer=layer, head=h, query_tile=qt,
                            canvas=int(data['canvas']), call=int(data['call']), age=st['age'], selection_event=anchor_event,
                            budget=budget, prefix_tiles=pt, retained_mass=float(m.selected(mask)[1].mean()),
                            iou_anchor=float((mask & anchor).sum() / (mask | anchor).sum().clamp_min(1)),
                            swaps=info['swaps'], pairs_scored=info['pairs'] if anchor_event else 0,
                            query_relative_change=float((q - st['q']).norm() / st['q'].norm().clamp_min(1e-30)), **metrics)
                        if name in st['errors']:
                            error = out - dense
                            # Prediction evaluation uses actual current error only after
                            # masks and causal estimates have already been frozen.
                            predicted = st['errors'][name] + min(st['age'], 1) * st['derivatives'][name]
                            row['static_prediction_error'] = float(squared_rows(error - st['errors'][name], gram).mean().sqrt())
                            row['tangent_prediction_error'] = float(squared_rows(error - predicted, gram).mean().sqrt())
                        records.append(row)
                    del m, outputs
            if time.monotonic() - last_monitor >= 1800:
                write(study / 'progress.json', dict(group=group_name, call=int(data['call']), completed_groups=completed,
                                                   elapsed_reservation_s=time.monotonic() - started, timing='paused'))
                print(json.dumps(dict(event='progress', group=group_name, call=int(data['call']))), flush=True)
                last_monitor = time.monotonic()
        write(leaf / 'records.json', records, True)
        write(leaf / 'complete.json', dict(status='complete', protocol_sha256=sha(study / 'operator_protocol.json'),
            records_sha256=sha(leaf / 'records.json'), records=len(records), input_frames=len(files),
            cuda_captures=captures['count'], gpu_reservation_s=time.monotonic() - group_start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), budget_errors=0), True)
        completed += 1
        print(json.dumps(dict(event='group_complete', group=group_name, records=len(records))), flush=True)
        del prefix, states, previous_q
        torch.cuda.empty_cache()
    summarize(study)


def summarize(study):
    protocol = verify(study)
    rows = []; groups = []
    for group in protocol['groups']:
        done = sorted((study / 'operator' / group).glob('attempt*/complete.json'))
        if done:
            m = read(done[-1]); assert m['records_sha256'] == sha(done[-1].parent / 'records.json')
            assert m['protocol_sha256'] == sha(study / 'operator_protocol.json')
            rows += read(done[-1].parent / 'records.json'); groups.append(group)
    buckets = collections.defaultdict(list)
    for r in rows:
        split = 'design' if r['request'] in protocol['design_requests'] else 'evaluation'
        age = 'anchor' if r['age'] == 0 else 'reuse'
        buckets[r['dataset'], split, age, r['arm']].append(r)
    summary = []
    for (dataset, split, age, arm), vals in sorted(buckets.items()):
        summary.append(dict(dataset=dataset, split=split, phase=age, arm=arm, states=len(vals),
            mean_full_v_error=sum(r['full_v_mean'] for r in vals)/len(vals),
            mean_post_o_error=sum(r['post_o_mean'] for r in vals)/len(vals),
            max_post_o_error=max(r['post_o_max'] for r in vals),
            mean_swaps=sum(r['swaps'] for r in vals)/len(vals)))
    complete = len(groups) == len(protocol['groups'])
    write(study / 'operator_summary.json', dict(status='complete' if complete else 'partial',
        groups_complete=len(groups), groups_expected=len(protocol['groups']), rows=len(rows), metrics=summary,
        model_decision_gate='not yet evaluated', native_accuracy='not launched', timing='paused'))
    write(study / 'state.json', dict(stage='operator_complete' if complete else 'operator_partial',
        groups_complete=len(groups), groups_expected=len(protocol['groups']), native_accuracy_launched=False, timing='paused'))
    if complete:
        write(study / 'operator_complete.json', dict(status='complete', scope=protocol['stage'], groups=len(groups),
            protocol_sha256=sha(study / 'operator_protocol.json'), summary_sha256=sha(study / 'operator_summary.json'),
            timing='paused', whole_study_complete=False), True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=DEFAULT)
    parser.add_argument('--stage', choices=['prepare', 'run', 'summary'], required=True)
    parser.add_argument('--limit-groups', type=int)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.study)
    elif args.stage == 'run':
        with torch.inference_mode():
            replay(args.study, args.limit_groups)
    else:
        summarize(args.study)


if __name__ == '__main__':
    main()
