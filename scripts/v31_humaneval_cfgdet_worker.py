"""K128/K256 evaluation dispatch derived from the frozen original worker.

Only budget checks/labels are generalized. Selectors and native decoding remain
those of the source-matched original execution plus the named compiler policy.
"""
import functools
import json
import os
from pathlib import Path
import runpy
import types

CONTROL = 'current_v31_control'
V2 = 'value_v2_online_preserve_mass'
V3A = 'value_v3a_singleton_delete'
REFINEMENTS = ('full_swap8', 'residual_direction_guard')
METHODS = (CONTROL, V2, V3A, *REFINEMENTS)


def bind(adapter, runtime, original_mage, original_receipt, schedule):
    if not schedule or schedule[0]['arm'] != CONTROL:
        raise ValueError('explicit implicit-control warmup required')
    if any(x['arm'] not in METHODS for x in schedule):
        raise ValueError('unknown live method')
    old_begin = adapter.begin_request
    refined_mage, refined_receipt = adapter._mage, adapter._value_receipt
    old_select = adapter._select_initial_refresh
    state = dict(index=-1, entry=None, budget_checks=[], units=0)

    @functools.wraps(old_begin)
    def begin(self, *args, **kwargs):
        index = state['index'] + 1
        if index >= len(schedule):
            raise RuntimeError('request exceeds frozen schedule')
        entry = schedule[index]
        arm = entry['arm']
        self.value_selector = arm if arm in (V2, V3A) else None
        self.value_threshold = entry.get('threshold') if arm == V2 else None
        if arm == V2 and (self.value_threshold is None or self.value_threshold < 0):
            raise ValueError('missing online threshold')
        self._mage = refined_mage if arm in REFINEMENTS else original_mage
        if arm in REFINEMENTS:
            runtime.mode = arm
        result = old_begin(*args, **kwargs)
        state.update(index=index, entry=entry, budget_checks=[], units=0)
        return result

    def select(self, layer, q, b, scale, prefix, n):
        out, kept = old_select(layer, q, b, scale, prefix, n)
        if self.value_audit and state['entry']['arm'] != V2:
            pt = prefix // 64
            state['budget_checks'].append((kept[..., :pt].sum(-1) != min(self.mage_k // 64, pt)).sum())
            state['units'] += kept.shape[1] * kept.shape[2]
        return out, kept

    def receipt(self):
        import torch
        entry = state['entry']
        source = refined_receipt if entry['arm'] in REFINEMENTS else original_receipt
        result = source()
        errors = (int(torch.stack(state['budget_checks']).sum())
                  if state['budget_checks'] else 0) if self.value_audit else None
        result.update(humaneval_live_arm=entry['arm'],
                      humaneval_threshold=entry.get('threshold'),
                      humaneval_schedule_label=entry.get('label', entry['arm']),
                      humaneval_request_index=state['index'],
                      humaneval_prefix_budget_tokens=self.mage_k,
                      humaneval_fixed_budget_errors=errors,
                      humaneval_fixed_budget_units=state['units'] if self.value_audit else None,
                      humaneval_budget_policy=('online_variable_threshold' if entry['arm'] == V2 else f'fixed_prefix_k{self.mage_k // 64}'),
                      humaneval_value_scope=('all_five_global_layers' if entry['arm'] in (V2, V3A)
                                             else 'global_layer29_only' if entry['arm'] in REFINEMENTS
                                             else 'inherited_v31_all_global_layers'))
        if entry['arm'] == V2:
            result['value_budget_policy'] = 'online_variable_threshold'
        return result

    adapter.begin_request = types.MethodType(begin, adapter)
    adapter._select_initial_refresh = types.MethodType(select, adapter)
    adapter._value_receipt = types.MethodType(receipt, adapter)
    return state


def install_oracle():
    import torch
    from types import SimpleNamespace
    from experiments.numerical_qk_reuse.v31_value_fast_runtime import FastRuntime
    from experiments.numerical_qk_reuse.v31_value_refinement_runtime import Runtime
    original = FastRuntime.initialize

    def initialize(self, adapter, q, b, scale, prefix, n, out, identity):
        limit = int(os.environ.get('HUMANEVAL_ORACLE_ANCHORS_PER_REQUEST', '2'))
        if limit < 1:
            raise ValueError('native qualification requires at least one oracle anchor')
        if self.counts.get('humaneval_native_oracle_checks', 0) >= limit:
            return original(self, adapter, q, b, scale, prefix, n, out, identity)
        st = dict(adapter.mage_state[29])
        st['kept'] = st['kept'].clone()
        reference = SimpleNamespace(mage_state={29: st}, mage_k=adapter.mage_k)
        oracle = Runtime(self.mode)
        oracle.model, oracle.grams = self.model, self.grams
        oracle.initialize(reference, q, b, scale, prefix, n, out, identity)
        original(self, adapter, q, b, scale, prefix, n, out, identity)
        if not torch.equal(st['kept'], adapter.mage_state[29]['kept']):
            raise RuntimeError(f'K{adapter.mage_k // 64} native same-state support mismatch')
        if self.corrects and not self.state.get('allkept'):
            a, c = oracle.state['direction_cache'], self.state['direction_cache']
            for name in ('anchor_selected', 'residual', 'dual', 'residual_squared'):
                if not torch.equal(getattr(a, name), getattr(c, name)):
                    raise RuntimeError(f'K{adapter.mage_k // 64} correction cache mismatch: ' + name)
        self.add('humaneval_native_oracle_checks', 1)
    FastRuntime.initialize = initialize


def main():
    from experiments.numerical_qk_reuse import vllm_adapter
    from experiments.numerical_qk_reuse.v31_value_fast_runtime import install
    from experiments.numerical_qk_reuse.v31_value_workspace import ValueWorkspace
    if os.environ.get('HUMANEVAL_ORACLE') == '1':
        if os.environ.get('VALUE_AUDIT') != '1':
            raise ValueError('same-state oracle is audit qualification only')
        install_oracle()
    schedule = json.loads(Path(os.environ['HUMANEVAL_REQUEST_ARMS']).read_text())
    runtime = install('full_swap8')
    old_install = vllm_adapter.install_vllm_patches

    def install_adapter(adapter):
        expected = int(os.environ['HUMANEVAL_PREFIX_BUDGET'])
        if expected not in (128, 256) or adapter.mage_k != expected:
            raise ValueError('frozen K128/K256 budget mismatch')
        original_mage, original_receipt = adapter._mage, adapter._value_receipt
        result = old_install(adapter)
        adapter._value_workspace = ValueWorkspace(int(os.environ['MAX_MODEL_LEN']))
        bind(adapter, runtime, original_mage, original_receipt, schedule)
        return result
    vllm_adapter.install_vllm_patches = install_adapter
    runpy.run_path(os.environ['HUMANEVAL_BASE_WORKER'], run_name='__main__')


if __name__ == '__main__':
    main()
