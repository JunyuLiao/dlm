import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / 'qualification001' / 'code' / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p = load('pulsehold.py')
s = load('state_restore.py')
import sys
sys.path.insert(0, str(ROOT / 'qualification001' / 'code'))
import support_bank as b


def test_anchor_boundaries_are_strictly_before_terminal_call():
    assert p.first_canvas_anchors(11) == (5, 9)
    assert p.first_canvas_anchors(3) == (2,)
    assert p.first_canvas_anchors(2) == ()


def test_single_deletion_bank_is_unit_stratified_and_deterministic():
    a = p.build_short_single_deletion_bank((2, 2, 2, 2), 8, seed=17)
    b = p.build_short_single_deletion_bank((2, 2, 2, 2), 8, seed=17)
    assert a == b
    assert len(a) == 8
    assert {x['deleted_unit'] for x in a} == {0, 1, 2, 3}
    assert all(x['physical_deletions'] == 1 for x in a)


def test_factorial_contrasts_and_schedule_are_explicit():
    assert p.factorial_schedule('support-a', 'SA') == ('support-a', 'A')
    assert p.factorial_schedule('support-a', 'AS') == ('A', 'support-a')
    assert p.factorial_schedule('support-a', 'SS') == ('support-a', 'support-a')
    values = {name: {'delta': value} for name, value in
              zip(('AA', 'SA', 'AS', 'SS'), (1.0, 1.5, 1.25, 2.0))}
    c = p.pulse_hold_contrasts(values, 'delta')
    assert c['propagated_pulse'] == .5
    assert c['direct_second_pulse'] == .25
    assert c['additional_held_application'] == .5
    assert c['interaction'] == .25
    assert p.next_branch_action('SA', 0, False) == 'S'
    assert p.next_branch_action('SA', 1, False) == 'A'
    assert p.next_branch_action('SA', 2, False) == 'A'
    assert p.next_branch_action('SA', 0, True) == 'NATIVE_DENSE'


def test_finalist_selection_uses_one_feasible_set_and_is_pre_outcome():
    candidates = [dict(support_hash=f'h{i}', feasible=True, current_error=float(i+1)/100,
        response_score=float(4-i), whole_deleted_mass=float(i)/100) for i in range(4)]
    scores = {x['support_hash']: dict(score_tuple=[i, .01, .1]) for i,x in enumerate(candidates)}
    out = p.select_finalists(candidates, scores)
    assert out['identifying']
    assert out['feasible_count'] == 4
    assert out['selections']['current_error'] == 'h0'
    assert out['selections']['response'] == 'h3'
    assert out['selections']['decision'] == 'h0'
    assert 'score_firewall' in out


def test_finalist_selection_abstains_below_identification_floor():
    candidates = [dict(support_hash='h0', feasible=True, current_error=.01,
                       response_score=.1, whole_deleted_mass=.01)]
    out = p.select_finalists(candidates, {})
    assert out['abstain']
    assert not out['identifying']


def test_reference_winner_margin_is_signed_and_positionwise():
    reference = torch.tensor([[3., 1., 0.], [0., 2., 1.]])
    candidate = torch.tensor([[2., 2.5, 0.], [0., 3., 2.]])
    out = p.reference_winner_margin(reference, candidate)
    assert out['reference_winner_boundary_crossings'] == 1
    assert out['margin_erosion_worst'] > 0
    assert out['margin_abs_change_mean'] > 0


def test_centered_metrics_separate_common_logit_shift():
    reference = torch.tensor([[1., 2., 4.]], dtype=torch.float64)
    candidate = reference + 7
    out = p.centered_distribution_metrics(reference, candidate)
    assert out['raw_logit_l2'] > 0
    assert out['centered_logit_l2'] == 0
    assert abs(out['js_mean']) < 1e-7


def test_native_argmax_tie_and_same_device_acceptance_are_labeled():
    logits = torch.tensor([[1., 1., 0.], [4., 0., 1.]])
    summary = p.native_argmax_tie_summary(logits, torch.tensor([0, 0]))
    assert summary['tie_positions'] == 1
    assert summary['native_winner_mismatches'] == 0
    accepted, receipt = p.acceptance_reconstructed_on_device(logits, .1)
    assert accepted.shape == logits.shape[:-1]
    assert receipt['exact_native_acceptance_capture'] is False
    assert receipt['measurement'].startswith('reconstructed_on_device')


def test_snapshot_verifies_live_tensor_metadata_and_aliases():
    a = torch.arange(8, dtype=torch.float32)
    roots = {'a': a, 'view': a.view(2, 4)}
    snap = s.Snapshot(roots)
    report = snap.verify()
    assert report['passed']
    assert report['metadata_mismatches'] == []
    assert report['alias_groups']
    a.add_(1)
    assert not snap.verify()['passed']
    snap.restore()
    assert snap.verify()['passed']


def test_support_bank_long_budget_has_equal_four_unit_work_and_matching_metadata():
    g = torch.Generator().manual_seed(9)
    q = torch.randn((256, 16, 8), generator=g, dtype=torch.float32)
    k = torch.randn((704, 2, 8), generator=g, dtype=torch.float32)
    v = torch.randn((704, 2, 6), generator=g, dtype=torch.float32)
    rows, receipt = b.candidate_bank(q, k, v, 640,
        dict(kind='per_unit_fraction', fraction=.9, scale=.3), seed=13, max_candidates=8)
    assert rows
    assert all(x['physical_eligible'] == 40 for x in rows)
    assert all(x['physical_kept'] == rows[0]['physical_kept'] for x in rows)
    assert receipt['scope']['gqa_mapping'] == {'0': 0, '8': 1}
    assert all(len(x['units']) == 4 for x in rows)


def test_matched_pairs_do_not_treat_singletons_as_identification():
    rows = [dict(support_hash='a', feasible=True, current_error=.01,
                 whole_deleted_mass=.1, response_score=.1),
            dict(support_hash='b', feasible=True, current_error=.0102,
                 whole_deleted_mass=.105, response_score=.2)]
    pairs = b.matched_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]['matched']
    assert pairs[0]['response_identifying_pair']


def test_converged_denoise_uses_phase_captured_before_native_mutation():
    import native_measurements as nm
    logits=torch.tensor([[[4.,0.]]])
    slots=torch.tensor([0]);canvas=torch.tensor([[0]])
    args=[logits,slots,slots,slots,torch.tensor([1]),canvas,canvas.clone(),torch.tensor([1]),
          torch.tensor([True]),torch.tensor([True]),torch.zeros((1,1,2)),torch.zeros((2,2)),
          torch.tensor(1.),torch.zeros((1,2,1),dtype=torch.long),torch.tensor([2]),canvas.clone()]
    _,row=nm.capture_native_sample(lambda *a,**kw:logits,*args,pre_commit=False,
        confidence_threshold=.005,entropy_bound=.1)
    assert row['is_commit'] is False
    assert row['native_commit_pending'] is True
    assert row['winner_measurement']['native_winner_equal']
