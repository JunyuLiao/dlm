"""Read-only native LLaDA trajectory tracing and same-state attention probes.

Previously examined tasks only. No EOS override, threshold search, or new
benchmark claims. Tensor snapshots and generations are private.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

import torch

V2 = "value_v2_gaussian32_reuse_r4"
V31 = "mass_topk_reuse_r4"
V3A = "value_v3a_singleton_reuse_r4"
TAU = .8944271909999159
PIN = "224f8f11173bd16e5cf10054a963c74233758cf2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def immutable(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def distribution(x):
    x = x.detach().float().flatten()
    return {"mean": x.mean().item(), "min": x.min().item(),
            "max": x.max().item(), "q05": torch.quantile(x, .05).item(),
            "q95": torch.quantile(x, .95).item()}


def eos_stats(logits, eos):
    x = logits.float()
    prob = torch.softmax(x, -1)
    other = x.clone()
    other[..., eos] = -torch.inf
    return {"eos_probability": prob[..., eos].tolist(),
            "eos_margin": (x[..., eos] - other.max(-1).values).tolist(),
            "eos_rank": (1 + (x > x[..., eos, None]).sum(-1)).tolist(),
            "eos_argmax": (x.argmax(-1) == eos).tolist(),
            "entropy": (-(prob * torch.log(prob.clamp_min(1e-30))).sum(-1)).tolist()}


def output_metrics(q, k, v, keep, scale):
    """FP32 mathematical full/sparse attention on the same GQA inputs."""
    h, n = q.shape[1:3]
    kh = torch.arange(h, device=q.device) // (h // k.shape[1])
    keys, values = k[0, kh].float(), v[0, kh].float()
    scores = torch.matmul(q[0].float(), keys.transpose(-1, -2)) * scale
    probs = scores.softmax(-1)
    full = torch.matmul(probs, values)
    # One Q128 block per LLaDA32 canvas; keep has H x KV64 geometry.
    token_keep = keep.reshape(h, -1).repeat_interleave(64, -1)[:, :k.shape[2]]
    retained = (probs * token_keep[:, None]).sum(-1)
    sparse = torch.matmul(probs * token_keep[:, None], values) / retained[..., None]
    nu = values.square().sum(-1).mean(-1).sqrt().clamp_min(1e-12)
    err = (sparse - full).norm(dim=-1)
    return {"retained_mass": distribution(retained),
            "error_over_value_rms": distribution(err / nu[:, None]),
            "relative_error_over_dense_output": distribution(err / full.norm(dim=-1).clamp_min(1e-12)),
            "kept_per_head": keep.reshape(h, -1).sum(-1).tolist()}, full, sparse


def pinned_modules(base):
    snapshot = base / "private/execution" / PIN
    package = types.ModuleType("experiments")
    package.__path__ = [str(snapshot / "experiments")]
    sys.modules["experiments"] = package
    sys.path[:0] = [str(snapshot), str(snapshot / "scripts")]
    attention = importlib.import_module("experiments.numerical_qk_reuse.multi_model_attention")
    runtime = importlib.import_module("experiments.numerical_qk_reuse.multi_model_runtime")
    for mod in (attention, runtime):
        if not Path(mod.__file__).resolve().is_relative_to(snapshot.resolve()):
            raise RuntimeError("module escaped frozen numerical snapshot")
    return attention, runtime


class Observer:
    def __init__(self, harness, private, item, arm):
        self.h = harness
        self.private, self.item, self.arm = private, item, arm
        self.steps, self.blocks, self.snapshots = [], [], []
        self.latest, self.first_write = {}, None
        self.previous_predictions = None
        self.eos = harness.tokenizer.eos_token_id
        self.original_attention = harness.runtime.attention
        self.original_joint = harness.model._joint_decode_block
        target = inspect.unwrap(harness.native_joint)
        self.target = target.__code__
        lines, first_line = inspect.getsourcelines(target)
        def number(fragment):
            return first_line + next(i for i, line in enumerate(lines) if fragment in line)
        self.sample_line = number('neg_inf = torch.full_like')
        self.write_line = number('is_original_mask_snapshot = list(')
        self.after_line = number('if x[0, block_start:block_end].tolist() == list(input_key)')

    def attention(self, q, k, v, layer, scale, **kwargs):
        rt = self.h.runtime
        active = rt.phase == "denoise" and self.arm == V2
        key = (layer, q.shape[2], k.shape[2])
        refresh = key not in rt.maps or (rt.call_in_block - 1) % 4 == 0
        result = self.original_attention(q, k, v, layer, scale, **kwargs)
        if active:
            if layer == 0:
                self.latest = {}
            self.latest[layer] = dict(q=q.detach().clone(), k=k.detach().clone(),
                                      v=v.detach().clone(), keep=rt.maps[key][0].detach().clone(),
                                      output=result.detach().clone(), scale=scale,
                                      refresh=refresh)
        return result

    def snapshot(self, label, loc):
        h, rt = self.h, self.h.runtime
        entry = dict(label=label, block_start=rt.block_start, block_end=rt.block_end,
                     call_in_block=rt.call_in_block, step_id=loc['step_id'],
                     logits=loc['block_logits'].detach().cpu().clone(),
                     before=loc['old_block'].detach().cpu().clone(),
                     input_ids=h.current_x.detach().clone(),
                     cache={layer: (k.detach().clone(), v.detach().clone())
                            for layer, (k, v) in h.cache.items()},
                     maps={key: (keep.clone(), lists) for key, (keep, lists) in rt.maps.items()},
                     layers=self.latest.copy())
        self.snapshots.append(entry)
        return entry

    def trace(self, frame, event, arg):
        if frame.f_code != self.target:
            return None
        if event != "line":
            return self.trace
        loc = frame.f_locals
        if frame.f_lineno == self.sample_line:
            pred, conf = loc['x0'], loc['p0']
            old, mask = loc['old_block'], loc['mask_index']
            es = eos_stats(loc['block_logits'], self.eos)
            self.step = dict(block_start=int(loc['block_start']), step_id=int(loc['step_id']),
                             call_in_block=self.h.runtime.call_in_block,
                             mask_count=int(mask.sum()), original_masks=int(loc['original_mask_count']),
                             post_steps=int(loc['post_steps']),
                             mean_entropy=sum(es['entropy']) / len(es['entropy']),
                             mean_confidence=float(conf.mean()),
                             eos_predictions=int((pred == self.eos).sum()),
                             high_conf_eos=int(((pred == self.eos) & (conf > .5)).sum()),
                             proposed_eos_on_masks=int(((pred == self.eos) & mask).sum()),
                             proposed_eos_on_tokens=int(((pred == self.eos) & ~mask & ~loc['prompt_mask_block']).sum()),
                             predicted_token_changes=int((pred != old).sum()),
                             mean_eos_probability=sum(es['eos_probability'])/len(es['eos_probability']))
            if self.arm == V2 and not self.snapshots:
                self.snapshot("first_forward", loc)
        elif frame.f_lineno == self.write_line:
            mask, pred, old = loc['mask_index'], loc['x0'], loc['old_block']
            selected = loc['mt2_index'] | loc['t2t_index']
            proposed_eos = selected & (pred == self.eos) & (old != self.eos)
            self.step.update(m2t=int(loc['mt2_index'].sum()), t2t=int(loc['t2t_index'].sum()),
                             eos_writes=int(proposed_eos.sum()),
                             eos_write_m2t=int((proposed_eos & mask).sum()),
                             eos_write_t2t=int((proposed_eos & ~mask).sum()),
                             final_round=bool(loc['final_round']))
            if self.arm == V2 and self.first_write is None and bool(proposed_eos.any()):
                self.first_write = self.snapshot("first_eos_write", loc)
                self.first_write['eos_positions'] = proposed_eos.nonzero().flatten().tolist()
                self.first_write['pre_write_eos'] = eos_stats(loc['block_logits'][proposed_eos], self.eos)
        elif frame.f_lineno == self.after_line:
            after = loc['x'][0, loc['block_start']:loc['block_end']]
            stable = torch.equal(after, loc['old_block']) and not bool((after == loc['mask_id']).any())
            self.step.update(after_masks=int((after == loc['mask_id']).sum()),
                             after_eos=int((after == self.eos).sum()), stable_stop=stable,
                             delete_proposals=int((loc['x0'] == loc['delete_token_id']).sum()),
                             split_proposals=int((loc['x0'] == loc['split_token_id']).sum()))
            self.steps.append(self.step)
        return self.trace

    def joint(self, *args, **kwargs):
        x, start, end = args[:3]
        previous = len(self.steps)
        result = self.original_joint(*args, **kwargs)
        self.blocks.append(dict(block_start=start, steps=len(self.steps)-previous,
                                eos_after=int((x[0, start:end] == self.eos).sum()),
                                mask_after=int((x[0, start:end] == 156895).sum()),
                                stop_reason='stable' if self.steps[-1]['stable_stop'] else 'budget_or_other'))
        return result

    def __enter__(self):
        self.h.runtime.attention = self.attention
        self.h.model._joint_decode_block = self.joint
        self.previous_trace = sys.gettrace()
        sys.settrace(self.trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous_trace)
        self.h.runtime.attention = self.original_attention
        self.h.model._joint_decode_block = self.original_joint


@torch.inference_mode()
def operator_probe(attention, snapshot):
    rows = []
    for layer, s in snapshot['layers'].items():
        q, k, v = (s[name] for name in ['q', 'k', 'v'])
        rt = attention.AttentionRuntime(V2, TAU)
        rt.block(snapshot['block_start'], snapshot['block_end'])
        rt.call_in_block = 1
        rt.attention(q, k, v, layer, s['scale'], block_size=32)
        fresh = rt.maps[(layer, q.shape[2], k.shape[2])][0]
        maps = {'actual_v2': s['keep'], 'fresh_v2': fresh}
        for name, arm, threshold in [('v31_half', V31, TAU), ('v3a_half', V3A, TAU),
                                     ('v2_aime_tau', V2, .012489663701836472)]:
            other = attention.AttentionRuntime(arm, threshold)
            other.block(snapshot['block_start'], snapshot['block_end'])
            other.call_in_block = 1
            other.attention(q, k, v, layer, s['scale'], block_size=32)
            maps[name] = other.maps[(layer, q.shape[2], k.shape[2])][0]
        for name, keep in maps.items():
            met, full, sparse = output_metrics(q, k, v, keep, s['scale'])
            actual = s['keep']
            met.update(label=snapshot['label'], layer=layer, method=name,
                       block_start=snapshot['block_start'], call=snapshot['call_in_block'],
                       eligible=keep.numel(), kept=int(keep.sum()), refresh=s['refresh'],
                       first_tile_keep_fraction=float(keep[..., 0].float().mean()),
                       iou_actual=float((actual & keep).sum() / (actual | keep).sum()))
            if name == 'actual_v2':
                consumer = s['output'][0].transpose(0, 1).float()
                met['consumer_vs_masked_fp32_relative_l2'] = float((consumer-sparse).norm()/sparse.norm())
            rows.append(met)
    return rows


@torch.inference_mode()
def same_state(h, snapshot):
    # All probes occur after native generation; restore the complete runtime.
    old_runtime, old_cache = h.runtime, h.cache
    old_flags = (h.in_joint, h.query_offset, h.store_cache, h.current_x)
    attention = importlib.import_module('experiments.numerical_qk_reuse.multi_model_attention')
    rows = []
    positions = snapshot['eos_positions']
    reference = snapshot['logits'][positions]
    for label, arm, tau, held in [
        ('same_v2_actual_map', V2, TAU, True),
        ('v2_fresh', V2, TAU, False), ('allkept_fa4', 'allkept_fa4', TAU, False),
        ('v31_half', V31, TAU, False), ('v3a_half', V3A, TAU, False),
        ('v2_aime_tau', V2, .012489663701836472, False)]:
        rt = attention.AttentionRuntime(arm, tau)
        rt.block(snapshot['block_start'], snapshot['block_end'])
        if held:
            rt.maps = snapshot['maps'].copy()
            rt.call_in_block = snapshot['call_in_block'] - 1
        h.runtime, h.cache = rt, snapshot['cache']
        h.in_joint, h.query_offset, h.store_cache = True, snapshot['block_start'], False
        try:
            logits = h._llada_forward(snapshot['input_ids'][:, :snapshot['block_end']]).logits.tensor[0]
            observed = logits[positions].float().cpu()
            delta = observed-reference
            rows.append(dict(route=label, positions=positions,
                             **eos_stats(observed, h.tokenizer.eos_token_id),
                             max_abs_logit_delta=float(delta.abs().max()),
                             counts=dict(rt.counts)))
        finally:
            h.runtime, h.cache = old_runtime, old_cache
            h.in_joint, h.query_offset, h.store_cache, h.current_x = old_flags
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('base', type=Path)
    p.add_argument('out', type=Path)
    args = p.parse_args()
    base, out = args.base.resolve(), args.out.resolve()
    protocol = json.loads((out/'protocol.json').read_text())
    if (out/'receipt.json').exists() or sha(__file__) != protocol['runner_sha256']:
        raise RuntimeError('fresh source-bound attempt required')
    os.environ['HF_MODULES_CACHE'] = str(base/'private/hf_modules')
    os.environ['TRITON_CACHE_DIR'] = str(base/'private/triton_cache')
    attention, runtime = pinned_modules(base)
    if sha(attention.__file__) != protocol['attention_sha256'] or sha(runtime.__file__) != protocol['runtime_sha256']:
        raise RuntimeError('frozen source mismatch')
    data = json.loads((base/'private/data/llada22/humaneval.json').read_text())
    if sha(base/'private/data/llada22/humaneval.json') != protocol['manifest_sha256']:
        raise RuntimeError('data mismatch')
    private = out/'private'
    private.mkdir(exist_ok=True)
    start = time.monotonic()
    h = runtime.ModelHarness(base, 'llada22').load()
    results, probes = [], []
    for item_id in protocol['cohort']:
        task = next(d for d in data if d['id'] == item_id)
        for arm in protocol['arms']:
            h.runtime.arm, h.runtime.threshold = arm, TAU
            obs = Observer(h, private, item_id, arm)
            with obs:
                result = h.generate(task['prompt'], 1024, 42, answer_prefix=task.get('answer_prefix',''))
            original = []
            for path in (base/'runs/llada22/humaneval'/arm).glob('shard*/attempt001/records.jsonl'):
                original += [json.loads(line) for line in path.read_text().splitlines()]
            saved = next(d for d in original if d['id'] == item_id and d['seed'] == 42)
            public = dict(id=item_id, arm=arm, output_tokens=len(result['tokens']),
                          counts=result['counts'], output_sha256=token_hash(result['tokens']),
                          matches_saved_output=token_hash(result['tokens']) == saved['output_sha256'],
                          matches_saved_counts=result['counts'] == saved['counts'],
                          steps=obs.steps, blocks=obs.blocks)
            immutable(private/f"{item_id.replace('/','_')}_{arm}.json",result)
            if not public['matches_saved_output'] or not public['matches_saved_counts']:
                immutable(out/'parity_failure.json',public)
                raise RuntimeError('observation changed saved native output/counts')
            if arm == V2:
                if obs.first_write is None:
                    raise RuntimeError('no actual EOS write observed')
                public['same_state'] = same_state(h, obs.first_write)
                for snap in obs.snapshots:
                    probes += [dict(id=item_id,**r) for r in operator_probe(attention,snap)]
                # QKV and tokens are private even though scalar diagnostics are public.
                torch.save(obs.snapshots, private/f"{item_id.replace('/','_')}_snapshots.pt")
            results.append(public)
            immutable(out/f"{item_id.replace('/','_')}_{arm}_receipt.json",public)
            print(json.dumps(dict(event='request_complete', id=item_id,arm=arm,
                                  output_tokens=public['output_tokens'],parity=True)),flush=True)
    immutable(out/'operator_metrics.json',probes)
    immutable(out/'receipt.json',dict(status='complete',protocol_sha256=sha(out/'protocol.json'),
                                      source_sha256=sha(__file__),results=results,
                                      gpu_worker_seconds=time.monotonic()-start,
                                      observer_output_and_count_parity=True,
                                      no_sampler_modifications=True,
                                      official_benchmark=False))
    immutable(out/'final_complete.json',dict(status='complete',requests=len(results),
                                            receipt_sha256=sha(out/'receipt.json')))


if __name__ == '__main__':
    main()
