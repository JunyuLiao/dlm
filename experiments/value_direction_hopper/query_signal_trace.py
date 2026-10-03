"""Causal per-query observations; no mutation of sampler outputs or logits.

The saved features are outcomes of a completed call. An analysis must shift
them by one call before predicting the next outcome and must not cross canvases.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import aime_temporal_sweep as base
from .aime_query_sensitivity_gated import model_config, PRIOR, CONTROLS
from .experiment import atomic, fingerprint, sha
from .query_sensitivity_uniform import UniformThresholdState
from .query_adaptive import observe
from .integration import install

ROOT = Path('/home/exouser/aime_query_signal_audit_v1')
GATED = Path('/home/exouser/aime_query_sensitivity_gated_v2/arms/C_gate_tau2p5')
CONDITIONS = ('dense', 'C_gate', 'T_prior')
SEEDS = (42, 43, 44)


class TraceState(UniformThresholdState):
    def __init__(self, method, router, **kwargs):
        # This is an offline signal-audit collector, not a production timing
        # path. Request the complete historical statistics explicitly because
        # the trace stores confidence, margin, and entropy for every method.
        super().__init__(method, router, diagnostics=False,
                         collect_all_stats=True, **kwargs)
        self.trace = []

    def observe_logits(self, logits, accepted, cur_step):
        previous_top = self.previous_top
        x = logits.float()
        temperature = .4 + .4 * cur_step/48
        raw = x * temperature
        raw_logz = torch.logsumexp(raw, -1)
        entropy = torch.distributions.Categorical(logits=x).entropy()
        previous_probability = (None if previous_top is None else
            (x.gather(-1, previous_top.unsqueeze(-1)).squeeze(-1) -
             torch.logsumexp(x, -1)).exp())
        super().observe_logits(logits, accepted, cur_step)
        current = dict(canvas=self.canvas, call=self.iteration, temperature=temperature,
            confidence=self.confidence, margin=self.margin, entropy=entropy,
            raw_confidence=(raw.max(-1).values-raw_logz).exp(),
            accepted=accepted, top=self.previous_top,
            previous_winner_confidence=(self.confidence if previous_probability is None
                                       else previous_probability))
        # Never retain references to sampler-owned buffers between calls.
        self.trace.append({k: (v.detach().cpu().numpy().copy().reshape(-1)
                              if torch.is_tensor(v) else v) for k, v in current.items()})

    def finish_step(self, result, cur_step):
        self.trace[-1].update(native_stop=bool(result[3].item()), cap_stop=cur_step==1)


def setup(root):
    final, calibration = base._load_rows()
    cfg = model_config()
    sources = [Path(__file__), Path(base.__file__),
               Path(__file__).with_name('query_adaptive.py'),
               Path(__file__).with_name('query_sensitivity_uniform.py'),
               Path(__file__).with_name('integration.py')]
    policies = {'C_gate': json.loads((GATED/'thresholds/C_gate_s50.json').read_text())['policy'],
                'T_prior': json.loads((PRIOR/'thresholds/T_prior_s50.json').read_text())['policy']}
    protocol = dict(schema='causal_query_signal_audit_v1', conditions=CONDITIONS,
        seeds=SEEDS, calibration_ids=[r['id'] for r in calibration],
        final_ids=[r['id'] for r in final], model_config=cfg, policies=policies,
        labels='Future renoising/flip at +1 and union over +1,+2; complete horizons only, within canvas',
        fitting='Six calibration question IDs only; other 24 IDs for predictive evaluation across seeds',
        limitation='Future instability is a proxy, not a causal label for value of extra attention',
        source_hashes={str(p.resolve()):sha(p) for p in sources})
    protocol = json.loads(json.dumps(protocol))
    protocol['fingerprint'] = fingerprint(protocol)
    path = root/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('trace code/config changed; use a new root')
    else:
        atomic(path, protocol)
        atomic(root/'final_manifest.json', final)
    return protocol, final, cfg, policies


def collect(root):
    protocol, final, cfg, policies = setup(root)
    adapter = base._adapter(cfg)
    projections = base.Projections()
    for condition in CONDITIONS:
        for seed in SEEDS:
            for source in final:
                row = dict(source, seed=seed)
                stem = root/'traces'/condition/f'seed{seed}'/str(row['source_id'])
                identity = fingerprint([protocol['fingerprint'], condition, row['id'], seed])
                if stem.with_suffix('.json').exists():
                    previous = json.loads(stem.with_suffix('.json').read_text())
                    if previous['identity'] != identity or not stem.with_suffix('.npz').exists():
                        raise ValueError(f'invalid trace cache: {stem}')
                    continue
                atomic(root/'status.json', dict(condition=condition, seed=seed, id=row['id'], started=time.time()))
                policy = policies.get(condition)
                ctx = (nullcontext((None, None)) if condition=='dense' else
                    install(adapter, cfg['library'], policy['late'], mode='value',
                            projections=projections, torch_library=cfg['torch_library'], collect=False))
                with ctx as (binding, router):
                    if binding is not None:
                        base._set_context(binding, row)
                    state = TraceState('T_prior' if condition=='dense' else condition, router,
                        m_ref=cfg['m_ref'], beta=3., gamma=.5,
                        trajectory_gamma=.65 if condition=='C_gate' else .5,
                        gate_tau=2.5, thresholds=policy['late'] if policy else {
                            k:dict(log_threshold=0.) for k in ('local','global')}, seed=seed)
                    start = time.time()
                    with observe(adapter.model, state):
                        output = adapter.generate(base._request(row))
                arrays = {k:np.stack([r[k] for r in state.trace]) for k in state.trace[0]}
                assert len(state.trace)==output.metadata['actual_denoising_step_count']
                stem.parent.mkdir(parents=True, exist_ok=True)
                temp = stem.with_suffix('.tmp.npz')
                np.savez_compressed(temp, **arrays)
                os.replace(temp, stem.with_suffix('.npz'))
                archive = (CONTROLS if condition=='dense' else GATED if condition=='C_gate' else PRIOR)
                cond = f'dense_seed{seed}' if condition=='dense' else f'{condition}_s50_seed{seed}'
                old_path = base._shard(archive, 'final', cond, row)
                old = json.loads(old_path.read_text())
                result = dict(identity=identity, condition=condition, seed=seed, id=row['id'],
                    prompt_hash=row['prompt_hash'], steps=len(state.trace),
                    score=float(base.aime_score(row, output.text)), prediction=output.text,
                    seconds_instrumented=time.time()-start,
                    archive_path=str(old_path), same_prediction=old['prediction']==output.text,
                    same_steps=old['steps']==len(state.trace),
                    archive_score=old['score'], archive_steps=old['steps'])
                atomic(stem.with_suffix('.json'), result)
                print(json.dumps({k:v for k,v in result.items() if k not in ('prediction','prompt_hash','identity')}), flush=True)
    atomic(root/'collection_complete.json', dict(passed=True, count=len(CONDITIONS)*len(SEEDS)*len(final), finished=time.time()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args=parser.parse_args()
    collect(args.root)


if __name__=='__main__':
    main()
