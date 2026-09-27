"""CPU contracts for the bounded v21 diagnostic; GPU replay is host-qualified."""
import json

import pytest
import torch

from scripts import v21_numerical_diagnostic as diagnostic


def test_frozen_eighteen_states_are_first_two_protocol_ids():
    protocol = json.loads(diagnostic.PROTOCOL.read_text())
    targets = diagnostic.frozen_targets(protocol)
    assert len(targets) == len({diagnostic.target_key(t) for t in targets}) == 18
    for dataset in ('aime26', 'longbench_v2', 'ruler4k'):
        task = [t for t in targets if t['dataset'] == dataset]
        assert [t['id'] for t in task[::3]] == protocol['ids'][dataset][:2]
        assert [(t['canvas'], t['call_index']) for t in task[:3]] == [(0, 0), (0, 1), (0, 3)]
        second = ([(0, 0), (0, 1), (0, 3)] if dataset == 'ruler4k'
                  else [(1, 0), (1, 3), (1, 6)])
        assert [(t['canvas'], t['call_index']) for t in task[3:]] == second


def test_config_rejects_target_drift_before_any_model_work():
    protocol = json.loads(diagnostic.PROTOCOL.read_text())
    targets = diagnostic.frozen_targets(protocol)
    with pytest.raises(ValueError, match='eighteen states'):
        diagnostic.validate_config(dict(seed=202, targets=targets, arms=[]))
    with pytest.raises(ValueError, match='question triple'):
        diagnostic.validate_config(dict(seed=101, targets=targets[:-1], arms=[]))
    with pytest.raises(ValueError, match='question triple'):
        diagnostic.validate_config(dict(seed=101, targets=targets[3:6] + targets[:3], arms=[]))
    # A host's three complete question triples reach arm validation next.
    with pytest.raises(ValueError, match='seven-arm'):
        diagnostic.validate_config(dict(seed=101, targets=targets[:3] + targets[6:9], arms=[]))


def test_logits_error_metrics_have_explicit_native_denominator_and_tail():
    native = torch.tensor([[[1., 3., 2.], [5., 0., -1.]]])
    other = torch.tensor([[[1., 2., 4.], [5., 0., -1.]]])
    row = diagnostic.error_metrics(native, other)
    assert row['finite_comparison']
    assert row['native_norm_denominator'] == pytest.approx(float(torch.linalg.vector_norm(native)))
    assert row['max_abs'] == 2.
    assert row['p99_abs'] > 1.
    assert row['p99_method'] == 'deterministic_stride_sample'
    assert row['top1_mismatch_positions'] == 1
    bad = diagnostic.error_metrics(native, other.fill_(float('nan')))
    assert not bad['finite_comparison']
    assert bad['nonfinite_other'] == native.numel()


def test_attention_metrics_use_full_fp32_reference_denominator():
    reference = torch.tensor([[[10., 0.]]])
    observed = torch.tensor([[[1., 0.]]])
    row = diagnostic.tensor_metrics(observed, reference)
    assert row['reference_norm_denominator'] == 10.
    assert row['relative_l2'] == pytest.approx(.9)
    assert 'top1_mismatch_positions' not in row


def test_signal_probe_passes_native_input_ids_and_step_to_processor():
    class Processor:
        def __call__(self, input_ids, logits, *, cur_step):
            assert input_ids.shape == (1, 3)
            assert cur_step == 45
            return logits
    class Stopper:
        confidence_threshold = .005
        argmax_canvas_history = None
        def __call__(self, top, logits):
            return torch.tensor([False])
    class Snapshot:
        def prepare(self, controller=None):
            return dict(input_ids=torch.ones((1, 3), dtype=torch.long), cur_step=45,
                        logits_processor=Processor(), diffusion_stopping_criteria=Stopper(),
                        finished_denoising=torch.tensor([False]))
    logits = torch.tensor([[[20., 0.], [20., 0.], [20., 0.]]])
    row = diagnostic.signal_probe(Snapshot(), logits)
    assert row['prior_finished'] is False
    assert row['native_criterion_stop'] is False
    assert row['effective_stop'] is False
    class FinishedSnapshot(Snapshot):
        def prepare(self, controller=None):
            kw = super().prepare(controller)
            kw['finished_denoising'] = torch.tensor([True])
            return kw
    with pytest.raises(ValueError, match='unfinished native pre-step'):
        diagnostic.signal_probe(FinishedSnapshot(), logits)


def test_stage_failure_is_retained_without_error_message_or_retry():
    report = {'stage_errors': []}
    calls = []
    def fail():
        calls.append(1)
        raise RuntimeError('sensitive raw payload')
    assert diagnostic._safe_stage(report, 'group', 'probe', fail) is None
    assert calls == [1]
    assert report['stage_errors'][0]['group'] == 'group'
    assert report['stage_errors'][0]['stage'] == 'probe'
    assert report['stage_errors'][0]['error_type'] == 'RuntimeError'
    assert report['stage_errors'][0]['code_location']['function'] == 'fail'
    assert 'sensitive raw payload' not in json.dumps(report)


def test_layout_comparison_requires_equal_digests_and_support():
    same = dict(output_digests={'0': 'abc'}, support_digests={'0': 'support'})
    group = {'arms': {'M3_R3_legacy': same, 'M3_R3_layout': dict(same),
                      'M3_R3_new': same, 'M3_R3_combined': dict(same)}}
    result = diagnostic._layout_report(group)
    assert all(v['0']['exact_output_elements'] and v['0']['identical_support']
               for v in result.values())
    group['arms']['M3_R3_layout'] = dict(output_digests={'0': 'changed'}, support_digests={'0': 'support'})
    with pytest.raises(AssertionError, match='layout-only'):
        diagnostic._layout_report(group)
