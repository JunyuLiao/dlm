"""Read-only routing probes around the frozen AIME runner; no policy changes.

Collect query-block counts that v14 discarded. Selected same-QKV replays are
diagnostic only: their outputs are never returned to the model. Verify complete
generation and physical-count parity with the historical shard before accepting
any new diagnostic. All files live in a separate result bundle.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import csv
import gzip
import json
import time

import torch

from . import aime_temporal_sweep as base
from .integration import Attention, _attention_type
from .experiment import atomic, sha

PARENT = base.DLMDIR / 'results/query_adaptive_aime_temporal_v14'
ROOT = base.DLMDIR / 'results/aime_temporal_tile_diagnostic_v1'
IDS = (2, 14, 23)  # Preselected calibration IDs; not selected by accuracy.
TARGETS = (30, 40, 50, 60, 70)
METHODS = ('gaussian32', 'temporal')


class Probe:
    def __init__(self, router, gaussian_policy, condition):
        self.router = router
        self.kernel = router.kernel
        self.policy = gaussian_policy
        self.condition = condition
        self.tiles = []
        self.comparisons = []

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        actual = self.kernel(*args, **kwargs)
        state = self.router.diagnostic_state
        layer, kind = self.router.diagnostic_layer
        nq = args[0].shape[-2]
        assert nq == 256, nq
        metadata = dict(canvas=state.canvas, iteration=state.iteration,
                        layer=layer, kind=kind)
        s = kwargs.get('sensitivity')
        if s is None:
            s = torch.ones((1, nq), device=args[0].device)
        tiled = s.reshape(1, 2, 128)
        sens = torch.stack((tiled.mean(-1), tiled.std(-1), tiled.amin(-1),
                            tiled.amax(-1), (tiled > 1.01).float().mean(-1)), -1)
        counts = torch.stack((actual.eligible.sum(-1), actual.skipped.sum(-1)), -1)
        self.tiles.append((metadata, counts, sens))
        # Four fixed layers, first canvas, three fixed chronological steps.
        # No sampling RNG is used, and no QKV/logits are persisted.
        if (self.condition.startswith('temporal') and state.canvas == 0
                and state.iteration in (3, 5, 10) and layer in (0, 5, 12, 17)):
            dense_kw = dict(kwargs, log_threshold=-float('inf'), sensitivity=None)
            dense = self.kernel(*args, **dense_kw)
            phase = 'early' if state.iteration <= 2 else 'late'
            variants = [('temporal', actual),
                        ('unit_same_tau', self.kernel(*args, **dict(kwargs, sensitivity=None))),
                        ('uniform_same_tau', self.kernel(*args, **dict(kwargs,
                            sensitivity=s.mean(-1, keepdim=True).expand_as(s).contiguous()))),
                        ('gaussian_frozen_tau', self.kernel(*args, **dict(kwargs,
                            sensitivity=None, log_threshold=self.policy[phase][kind]['log_threshold'])))]
            d = dense.output.float()
            for name, other in variants:
                x = other.output.float()
                assert torch.equal(actual.eligible, other.eligible)
                error = (x-d).square().sum()
                denom = d.square().sum()
                e = actual.eligible
                self.comparisons.append(dict(**metadata, variant=name,
                    eligible=int(e.sum()), skipped=int(other.skipped.sum()),
                    mask_disagreements_with_temporal=int(((other.skipped != actual.skipped) & e).sum()),
                    retained_by_temporal_only=int((other.skipped & ~actual.skipped & e).sum()),
                    skipped_by_temporal_only=int((~other.skipped & actual.skipped & e).sum()),
                    error_sq=float(error), dense_sq=float(denom),
                    relative_l2=float((error/denom.clamp_min(1.e-20)).sqrt()),
                    cosine=float((x*d).sum()/(x.square().sum()*denom).clamp_min(1.e-20).sqrt())))
            # Independent kernel calls may not mutate the producer-owned QKV/Z.
            repeat = self.kernel(*args, **kwargs)
            assert torch.equal(actual.output, repeat.output)
            assert torch.equal(actual.skipped, repeat.skipped)
        return actual

    def rows(self):
        counts = torch.stack([x[1] for x in self.tiles]).cpu().tolist()
        sensitivities = torch.stack([x[2] for x in self.tiles]).cpu().tolist()
        for (meta, _, _), cc, ss in zip(self.tiles, counts, sensitivities):
            assert len(cc) == 1
            for head, hh in enumerate(cc[0]):
                for qb, (eligible, skipped) in enumerate(hh):
                    if not eligible:
                        continue
                    yield dict(**meta, head=head, query_tile=qb,
                        eligible=eligible, skipped=skipped, sparsity=skipped/eligible,
                        **dict(zip(('sensitivity_mean','sensitivity_std','sensitivity_min',
                                    'sensitivity_max','sensitivity_active_fraction'), ss[0][qb])))


@contextmanager
def instrument(gaussian_policy, condition, probes):
    original_install, original_state = base.install, base.PhaseState
    original_call = Attention.__call__

    class InstrumentedState(original_state):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.router.diagnostic_state = self

    @contextmanager
    def install(*args, **kwargs):
        with original_install(*args, **kwargs) as (binding, router):
            probe = Probe(router, gaussian_policy, condition)
            probes.append(probe)
            router.kernel = probe
            yield binding, router

    def call(self, module, *args, **kwargs):
        self.diagnostic_layer = (int(module.layer_idx),
            _attention_type(module, kwargs.get('sliding_window')))
        return original_call(self, module, *args, **kwargs)

    base.install, base.PhaseState, Attention.__call__ = install, InstrumentedState, call
    try:
        yield
    finally:
        base.install, base.PhaseState, Attention.__call__ = original_install, original_state, original_call


def run(root, smoke=False):
    root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((PARENT/'configuration.json').read_text())
    manifest = json.loads((PARENT/'final_manifest.json').read_text())
    source_files = [Path(__file__), Path(base.__file__),
        Path(__file__).with_name('query_adaptive.py'),
        Path(__file__).with_name('query_adaptive_guardrail.py'),
        Path(__file__).with_name('integration.py'), Path(cfg['library']), Path(cfg['torch_library'])]
    config = dict(parent=str(PARENT), seed=42, ids=list(IDS), targets=list(TARGETS),
        methods=list(METHODS), source_hashes={str(p):sha(p) for p in source_files},
        same_qkv_layers=[0,5,12,17], same_qkv_steps=[3,5,10], same_qkv_canvas=0,
        selection='Three predeclared previously exposed calibration examples; not representative accuracy evidence',
        timing='Diagnostic overhead; never use these runtimes for speed claims')
    config_path = root/'configuration.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Diagnostic source changed; use a new result version'
    else:
        atomic(config_path, config)
    adapter = base._adapter(cfg)
    projections = base.Projections()
    targets = (50,) if smoke else TARGETS
    ids = (2,) if smoke else IDS
    for target in targets:
        gp = json.loads((PARENT/'thresholds'/f'gaussian32_s{target}.json').read_text())['policy']
        for method in METHODS:
            condition = f'{method}_s{target}_seed42'
            policy = json.loads((PARENT/'thresholds'/f'{method}_s{target}.json').read_text())['policy']
            for source_id in ids:
                dest = root/'runs'/f'{condition}_id{source_id}'
                if (dest/'audit.json').exists():
                    assert json.loads((dest/'audit.json').read_text())['passed']
                    continue
                row = dict(next(r for r in manifest if int(r['source_id']) == source_id), seed=42)
                old = json.loads(base._shard(PARENT, 'final', condition, row).read_text())
                atomic(root/'status.json', dict(condition=condition, id=row['id'], started=time.time()))
                probes = []
                with instrument(gp, condition, probes):
                    result = base._run_one(adapter, row, method, policy, cfg, projections)
                dest.mkdir(parents=True, exist_ok=True)
                assert len(probes) == 1
                passed = all(result[k] == old[k] for k in ('completion_tokens','steps','counts','phase_counts','score'))
                atomic(dest/'parity.json', dict(passed=passed,
                    fields={k:result[k] == old[k] for k in ('completion_tokens','steps','counts','phase_counts','score')},
                    historical_sha=sha(base._shard(PARENT,'final',condition,row))))
                if not passed:
                    atomic(dest/'failed_output.json',result)
                    raise RuntimeError('Historical generation parity failed; do not merge diagnostic results')
                rows = list(probes[0].rows())
                for kind in ('whole','local','global'):
                    group = [r for r in rows if kind == 'whole' or r['kind']==kind]
                    assert all(sum(r[f] for r in group)==result['counts'][kind][f] for f in ('eligible','skipped'))
                with gzip.open(dest/'query_tiles.csv.gz', 'wt') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                atomic(dest/'same_qkv.json',probes[0].comparisons)
                atomic(dest/'audit.json',dict(passed=True,condition=condition,id=row['id'],query_tiles=len(rows),
                    matched_historical_output=True,counts_reconciled=True,extra_kernel_results_discarded=True,
                    steps=result['steps'],canvases=result['total_canvases']))
                print(json.dumps(dict(condition=condition,id=row['id'],tiles=len(rows),parity=True)),flush=True)
    atomic(root/('smoke_complete.json' if smoke else 'complete.json'),dict(passed=True,finished=time.time()))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.root,a.smoke)
