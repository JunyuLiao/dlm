"""Read-only diagnostics around the unchanged vLLM native sampler.

No model forward, sampler draw, sampler state write or stopping override occurs
here. The dense output/state parity gate in worker.py is mandatory.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import torch


def digest(tensor):
    host = tensor.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()


def rng_state():
    return (torch.get_rng_state().clone(),
            torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None)


def assert_rng_equal(before):
    after = rng_state()
    if not torch.equal(before[0], after[0]) or (before[1] is not None and
            not torch.equal(before[1], after[1])):
        raise RuntimeError('observer changed a native RNG state')


def state_snapshot(args):
    slots = args[1]
    result = {}
    for name, index in [('canvas', 5), ('argmax', 6), ('step', 7),
                        ('encoder_phase', 8), ('confidence', 9),
                        ('self_condition', 10), ('history', 13), ('history_len', 14)]:
        value = args[index][slots].detach().cpu().contiguous()
        if name == 'history':
            # Ignore inactive storage left by a preceding request.
            length = min(int(args[14][slots].item()), value.shape[1])
            value = value[:, :length].contiguous()
        result[name + '_sha256'] = digest(value)
        if name in ('step', 'encoder_phase', 'confidence', 'history_len'):
            result[name] = value.reshape(-1).tolist()
    result['cpu_rng_sha256'] = digest(torch.get_rng_state())
    if args[0].is_cuda:
        result['cuda_rng_sha256'] = digest(torch.cuda.get_rng_state())
    result['valid_canvas_len'] = args[4].detach().cpu().tolist()
    return result


def distribution_stats(scaled, entropy_bound):
    """Native entropy formula, bounded vocabulary workspace (16 rows)."""
    entropies, margins, top_probabilities = [], [], []
    values = scaled.detach().reshape(-1, scaled.shape[-1])
    for chunk in values.split(16):
        logp = chunk.float().log_softmax(-1)
        probability = logp.exp()
        entropies.append(-(probability * logp).sum(-1).cpu())
        top = chunk.float().topk(2, dim=-1).values
        margins.append((top[:, 0] - top[:, 1]).cpu())
        top_probabilities.append(logp.amax(-1).exp().cpu())
    entropy = torch.cat(entropies)
    margin = torch.cat(margins)
    top_probability = torch.cat(top_probabilities)
    ordered, indices = entropy.sort()
    accepted = torch.zeros_like(entropy, dtype=torch.bool)
    accepted[indices] = ordered.cumsum(0) - ordered.cummax(0).values <= entropy_bound
    return entropy, margin, top_probability, accepted


class ProductionObserver:
    def __init__(self):
        self.begin('off')

    def begin(self, mode):
        if mode not in ('off', 'minimal', 'rich'):
            raise ValueError(mode)
        self.mode = mode
        self.sampler_calls = self.denoise_calls = self.commit_calls = 0
        self.canvas = 0
        self.records, self.fingerprints = [], []
        self.private_argmax, self.private_entropy = [], []
        self.stable_dwell = self.confident_dwell = None
        self.previous_sc = None
        self.rng_checks = 0
        self.native_predicate_mismatches = 0

    def sample(self, original, *args, **kwargs):
        if self.mode == 'off':
            self.sampler_calls += 1
            return original(*args, **kwargs)
        if args[1].numel() != 1:
            raise RuntimeError('diagnostic requires exactly one native decode slot')
        rng_before_observer = rng_state()
        before = state_snapshot(args)
        slots = args[1]
        committing = bool(args[8][slots].item())
        previous = args[6][slots].detach().cpu().clone().reshape(-1)
        assert_rng_equal(rng_before_observer)
        self.rng_checks += 1
        result = original(*args, **kwargs)
        rng_after_native = rng_state()
        after = state_snapshot(args)
        self.fingerprints.append({'before': before, 'after': after})
        self.sampler_calls += 1
        if committing:
            self.commit_calls += 1
            self.canvas += 1
            self.stable_dwell = self.confident_dwell = None
            self.previous_sc = None
        else:
            self.denoise_calls += 1
            if self.mode == 'rich':
                self.record(args, kwargs, result, previous, before, after)
        assert_rng_equal(rng_after_native)
        self.rng_checks += 1
        return result

    def record(self, args, config, scaled, previous, before, after):
        entropy, margin, top_probability, accepted = distribution_stats(scaled, config['entropy_bound'])
        slots = args[1]
        current = args[6][slots].detach().cpu().clone().reshape(-1)
        stable_positions = (current == previous) if self.stable_dwell is not None else torch.zeros_like(current, dtype=torch.bool)
        confident_positions = entropy < config['confidence_threshold']
        if self.stable_dwell is None:
            self.stable_dwell = torch.zeros_like(current, dtype=torch.int64)
            self.confident_dwell = torch.zeros_like(current, dtype=torch.int64)
        self.stable_dwell = torch.where(stable_positions, self.stable_dwell + 1, 0)
        self.confident_dwell = torch.where(confident_positions, self.confident_dwell + 1, 0)
        # Read the native circular history AFTER its write; do not duplicate its
        # update and do not replace its convergence flag. ST includes current.
        history = args[13][slots].detach().cpu()
        native_stable = bool((history == history[:, :1]).all())
        enough_history = int(args[14][slots].item()) >= int(config['ST'])
        native_confident = bool(args[9][slots].item())
        step_after = int(args[7][slots].item())
        cap = step_after >= int(config['max_denoising_steps'])
        converged = bool(args[8][slots].item())
        expected = (native_stable and enough_history and native_confident) or cap
        self.native_predicate_mismatches += int(expected != converged)
        sc = args[10][slots].detach().float().cpu()
        sc_drift = None if self.previous_sc is None else float((sc - self.previous_sc).norm())
        self.previous_sc = sc
        bottleneck = ('joint_satisfied' if native_stable and enough_history and native_confident else
                      'confidence' if native_stable and enough_history else
                      'stability' if native_confident else 'both')
        row = dict(canvas_id=self.canvas, call_index=self.denoise_calls - 1,
            call_in_canvas=step_after, temperature=float(config['t_min'] +
                (config['t_max']-config['t_min']) * max(1, config['max_denoising_steps']-step_after+1)/config['max_denoising_steps']),
            entropy_mean=float(entropy.mean()), entropy_p50=float(entropy.quantile(.5)),
            entropy_p90=float(entropy.quantile(.9)), entropy_max=float(entropy.max()),
            margin_p10=float(margin.quantile(.1)), margin_p50=float(margin.quantile(.5)),
            margin_p90=float(margin.quantile(.9)), top1_prob_mean=float(top_probability.mean()),
            accepted_count=int(accepted.sum()), renoised_count=int((~accepted).sum()),
            accepted_fraction=float(accepted.float().mean()),
            stable_position_fraction=float(stable_positions.float().mean()),
            confident_position_fraction=float(confident_positions.float().mean()),
            stable_and_confident_fraction=float((stable_positions & confident_positions).float().mean()),
            argmax_flip_count=None if step_after == 1 else int((~stable_positions).sum()),
            stable_dwell_p50=float(self.stable_dwell.float().median()),
            confident_dwell_p50=float(self.confident_dwell.float().median()),
            self_conditioning_norm=float(sc.norm()), self_conditioning_drift_l2=sc_drift,
            native_stable=native_stable, native_history_ready=enough_history,
            native_confident=native_confident, diagnostic_confident=bool(entropy.mean()<config['confidence_threshold']),
            native_converged=converged, cap_reached=cap, stopping_bottleneck=bottleneck,
            stop_reason=('cap_and_joint' if cap and bottleneck=='joint_satisfied' else
                         'cap' if cap else 'stable_and_confident') if converged else None,
            valid_canvas_tokens=int(args[4].item()),
            stable_dwell_fractions={str(d): float((self.stable_dwell>=d).float().mean()) for d in (1,2,4,8)},
            confident_dwell_fractions={str(d): float((self.confident_dwell>=d).float().mean()) for d in (1,2,4,8)})
        self.records.append(row)
        # Token-level arrays remain in memory; only sanitized attribution leaves.
        self.private_argmax.append((self.canvas, current))
        self.private_entropy.append((self.canvas, entropy))

    def summary(self):
        calls = Counter(r['canvas_id'] for r in self.records)
        canvases = []
        for canvas, count in sorted(calls.items()):
            records = [r for r in self.records if r['canvas_id'] == canvas]
            a = torch.stack([a for c,a in self.private_argmax if c == canvas])
            e = torch.stack([e for c,e in self.private_entropy if c == canvas])
            terminal = a[-1]
            attribution = {}
            for dwell in (1,2,4,8):
                index = max(0,len(a)-1-dwell)
                attribution[str(dwell)] = dict(
                    terminal_argmax_already_present_fraction=float((a[index]==terminal).float().mean()),
                    terminal_argmax_persistent_fraction=float((a[index:]==terminal).all(0).float().mean()),
                    terminal_confident_persistent_fraction=float((e[index:]<.005).all(0).float().mean()),
                    available=len(a)>dwell)
            canvases.append(dict(canvas_id=canvas,calls=count,terminal=records[-1],
                terminal_attribution=attribution))
        return dict(sampler_calls=self.sampler_calls,denoising_calls=self.denoise_calls,
            commit_calls=self.commit_calls,canvas_count=len(calls),calls_per_canvas=dict(calls),
            N_over_C=self.denoise_calls/len(calls) if calls else None,canvases=canvases,
            stopping_bottlenecks=dict(Counter(r['stopping_bottleneck'] for r in self.records)),
            max_call_canvases=sum(r['terminal']['cap_reached'] for r in canvases),
            observer_rng_checks=self.rng_checks,native_predicate_mismatches=self.native_predicate_mismatches)
