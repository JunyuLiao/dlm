"""Reuse established calibration code; only its execution backend is replaced."""
import json
from pathlib import Path
import traceback
from unittest.mock import patch

from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _append
from experiments.diffusion_gemma_value_aware_followup import calibrate as original, policies
from experiments.diffusion_gemma_value_aware_followup.development import policy_for
from .protocol import PARENT, METHODS, CONFIGS
from .runner import cached


def run_group(adapter, root, rows, condition, config, policy, execution):
    if len(rows) != 6 or any(r['split'] != 'calibration' for r in rows):
        raise ValueError('Only the six original calibration rows may fit thresholds')
    outputs, sources, failures = [], {}, []
    for row in sorted(rows, key=lambda r:(r['generation_budget'], r['id'])):
        try:
            out = cached(adapter, root, row, 'calibration', condition, config, policy, execution)
            path = shard_path(root, 'calibration', condition, row['id'])
            outputs.append(out)
            sources[row['id']] = dict(path=str(path), sha256=sha(path.read_bytes()),
                fingerprint=execution['fingerprint'], stage='calibration')
        except Exception:
            error = dict(stage='calibration', id=row['id'], condition=condition, traceback=traceback.format_exc())
            _append(root/'failures.jsonl', error)
            failures.append(error)
            print(error, flush=True)
            import torch
            torch.cuda.empty_cache()
    return outputs, sources, failures


def prepare_policies(adapter, root, setup, execution):
    parent_execution = json.loads((PARENT/'execution_contract.json').read_text())
    parent_setup = json.loads((PARENT/'setup.json').read_text())
    proposals, _ = policies.inputs(PARENT, parent_execution)
    failures = []
    for benchmark in ('aime26', 'longbench_v2'):
        for name in METHODS:
            try:
                old, source = policy_for(PARENT, parent_setup, parent_execution, benchmark, name, CONFIGS[name], .5)
                frozen_write(root/'imported_policies'/benchmark/f'{name}_s50.json', dict(
                    target=.5, config=CONFIGS[name], policy=old['policy'], measured=old['measured'],
                    source=source, calibration_ids=old['calibration_ids'], heldout_used=False,
                    reuse_rule='Unchanged50% thresholds; source audited by original policy_for, kernel parity required',
                    parent_policy=old))
            except Exception:
                error = dict(stage='policy50', benchmark=benchmark, method=name, traceback=traceback.format_exc())
                _append(root/'failures.jsonl', error); failures.append(error)
    # No new calibration rule: use existing dense-calibration proposals,
    # inverse-L BLASST refinement, scalar-risk CDF refinement, lambda1 boundary,
    # joint-point selection and raw verification exactly as implemented before.
    def old_distributions(unused_root, rows, name, config, unused_execution):
        return policies.distributions(PARENT, rows, name, config, parent_execution)
    with patch.object(original, 'run_group', run_group), patch.object(original, 'distributions', old_distributions):
        for benchmark in ('aime26', 'longbench_v2'):
            for name in METHODS:
                try:
                    original.calibrate_one(adapter, root, setup, execution, proposals, name,
                                           CONFIGS[name], .75, benchmark, max_rounds=3)
                except Exception:
                    error = dict(stage='policy75', benchmark=benchmark, method=name, traceback=traceback.format_exc())
                    _append(root/'failures.jsonl', error); failures.append(error)
                    print(error, flush=True)
    return failures


def condition(root, setup, execution, name, target):
    if name == 'dense':
        return dict(config={}, target=0., target_metric='physical_sparsity', thresholds={}, policy_sources={})
    if name not in METHODS or target not in (.5, .75):
        raise ValueError('Unrequested condition')
    threshold, sources = {}, {}
    for benchmark in ('aime26', 'longbench_v2'):
        path = root/('imported_policies' if target == .5 else 'verified_policies')/benchmark/f'{name}_s{int(target*100)}.json'
        policy = json.loads(path.read_text())
        if policy['config'] != CONFIGS[name] or policy['target'] != target or policy['heldout_used']:
            raise ValueError('Policy identity/selection violation')
        if target == .75:
            policies.audit_policy(policy, policies.calibration_rows(setup, benchmark), CONFIGS[name], execution)
        else:
            source = policy['source']
            if sha(Path(source['path']).read_bytes()) != source['sha256']:
                raise ValueError('Imported50 policy changed')
        threshold[benchmark] = policy['policy']
        sources[benchmark] = dict(path=str(path), sha256=sha(path.read_bytes()))
    return dict(config=CONFIGS[name], target=target, target_metric='physical_sparsity',
                thresholds=threshold, policy_sources=sources)
