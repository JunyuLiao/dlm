"""Read only completed anchor receipts after the frozen qualification stop.

This is a partial-stage readout, never a substitute for the four-anchor marker.
No generated text, private tensors, or task labels are read.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, data):
    with p.open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output-name', default='analysis_partial002')
    args = ap.parse_args()
    root = args.root.resolve()
    out = root / args.output_name
    out.mkdir(exist_ok=False)
    protocol = read(root / 'protocol.json')
    exit_receipt = read(root / 'exit_receipt.json')
    assert exit_receipt['returncode'] == 1
    assert not (root / 'execution001/final_complete.json').exists()
    assert sha(root / 'protocol.json') == read(root / 'source_fingerprints.json')['protocol_sha256']
    assert all(sha(f) == h for f, h in protocol['sources'].items())
    assert all(sha(f) == h for f, h in protocol['scorer_sources'].items())
    assert sha(root / 'private/tokenized_manifest.json') == protocol['private_manifest_sha256']
    assert sha(root / 'manifest_identity.json') == protocol['public_manifest_sha256']
    assert sha(root / 'runtime_fingerprints.json') == protocol['runtime_fingerprints_sha256']
    cpu = ET.parse(root / 'tests/cpu004.xml').getroot().find('testsuite').attrib
    assert cpu['tests'] == '24' and cpu['errors'] == cpu['failures'] == '0'
    keys = ['fingerprint_sha256', 'remaining_denoise', 'rng_cpu', 'rng_cuda',
            'forwards', 'steps', 'committed']
    anchors, incomplete, hashes = [], [], {}
    qualification = []
    measured = Counter()
    for item in sorted((root / 'execution001').glob('item*')):
        landmarks = read(item / 'reference_landmarks.json')
        L = landmarks['L']
        assert landmarks['anchors'] == sorted(set(a for a in (max(2, L // 2), max(2, L - 2)) if a < L))
        for boundary in landmarks['anchors']:
            folder = item / f'anchor{boundary:03d}'
            if not (folder / 'receipt.json').exists():
                incomplete.append(dict(id=landmarks['id'], anchor=boundary,
                    snapshot_exists=(folder / 'snapshot.json').exists(),
                    status='qualification_failed_before_bank' if folder.exists() else 'not_reached'))
                continue
            receipt = read(folder / 'receipt.json')
            bank = read(folder / 'candidate_bank.json')
            controls = read(folder / 'controls.json')
            choices = read(folder / 'selections.json')
            snap = read(folder / 'snapshot.json')
            assert snap['passed'] and not snap['metadata_mismatches'] and not snap['storage_mismatches']
            for name in ['native_repeat', 'new_measurement_disabled', 'post_native']:
                assert all(controls['native'][k] == controls[name][k] for k in keys)
            for name in ['allkept_repeat', 'post_allkept']:
                assert all(controls['allkept'][k] == controls[name][k] for k in keys)
            assert controls['measurement_parity'] and controls['native_first_canvas_parity']
            assert controls['allkept_first_canvas_parity'] and controls['gqa_qualification']['passed']
            assert max(controls['gqa_qualification']['worst_relative_output_error'].values()) <= .02
            # Completed branches must retain explicit forward and engine-call equality.
            for name in ['native', 'native_repeat', 'new_measurement_disabled',
                         'post_native', 'allkept', 'allkept_repeat', 'post_allkept']:
                ex = controls[name]
                assert ex['forwards'] == ex['steps']
                if name == 'new_measurement_disabled':
                    assert not ex['trace']
                else:
                    assert len(ex['trace']) == ex['remaining_denoise']
                    assert ex['trace'][-1]['converged']
                assert ex['committed']
                for call in ex['trace']:
                    wm, am = call['winner_measurement'], call['acceptance_measurement']
                    measured['call_records'] += 1
                    measured['native_winner_mismatches'] += wm['native_winner_mismatches']
                    measured['native_tie_positions'] += wm['tie_positions']
                    measured['reconstructed_acceptance_threshold_ties'] += am['threshold_tie_positions']
                    measured['reconstructed_acceptance_threshold_near'] += am['threshold_near_positions']
                    measured['cpu_threshold_discrepancies_given_device_entropy'] += am['cpu_threshold_discrepancies_given_device_entropy']
                    assert not am['exact_native_acceptance_capture']
            candidates = bank['candidates']
            feasible = [c for c in candidates if c['feasible']]
            assert len(feasible) == receipt['feasible'] == 2
            assert len(candidates) == 8
            assert all(c['physical_eligible'] == 8 and c['physical_kept'] == 7 for c in candidates)
            assert not choices['finalist_hashes'] and all(h is None for h in choices['selections'].values())
            assert not list(folder.glob('branch*.json'))
            excluded, candidate_rows = Counter(), []
            for c in candidates:
                reasons = []
                for field, limit, label in [('whole_deleted_mass', .2, 'deleted_mass'),
                                             ('current_error', .1, 'current_bias'),
                                             ('whole_renormalization', 1.25, 'renormalization')]:
                    if c[field] > limit:
                        reasons.append(label)
                        excluded[label] += 1
                assert c['feasible'] == (not reasons)
                deleted = [(u, i) for u, row in enumerate(c['support']) for i, kept in enumerate(row) if not kept]
                assert len(deleted) == 1
                candidate_rows.append(dict(support_hash=c['support_hash'], deleted_unit=deleted[0][0],
                    deleted_prefix_block=deleted[0][1], feasible=c['feasible'], exclusion_reasons=reasons,
                    current_error=c['current_error'], response_score=c['response_score'],
                    whole_deleted_mass=c['whole_deleted_mass'], whole_renormalization=c['whole_renormalization']))
            pulses = []
            for h, data in sorted(bank['first_pulse_metrics'].items()):
                ev = data['events'][0]
                margin = ev['reference_winner_margin']
                assert data['aligned_calls'] == 1
                pulses.append(dict(support_hash=h, native_winner_mismatch=ev['actual_native_winner_mismatch'],
                    canvas_after_mismatch=ev['canvas_mismatch'], native_history_mismatch=ev['native_history_mismatch'],
                    self_condition_l2=ev['self_condition_l2'], native_confidence_changed=ev['native_confidence_changed'],
                    native_convergence_changed=ev['native_commit_pending_changed'],
                    reconstructed_acceptance_mismatch=ev['acceptance_reconstructed_mismatches'],
                    mean_absolute_margin_change=margin['margin_abs_change_mean'],
                    worst_margin_erosion=margin['margin_erosion_worst'],
                    reference_winner_crossings=margin['reference_winner_boundary_crossings'],
                    centered_logit_l2=ev['logits']['centered_logit_l2'], js_mean=ev['logits']['js_mean'],
                    decision=bank['first_pulse_decisions'][h]))
            aa = controls['allkept']
            native_restored = controls['native_repeat']
            assert aa['global_skipped_tiles'] == aa['local_skipped_tiles'] == 0
            assert aa['global_kept_tiles'] == aa['global_eligible_tiles']
            assert aa['local_kept_tiles'] == aa['local_eligible_tiles']
            assert all(x['action'] == 'A' and x['skipped'] == 0 for x in aa['physical'])
            row = dict(**receipt, candidate_count=len(candidates), candidate_rows=candidate_rows,
                excluded_by_guard=dict(excluded), first_pulse_only=pulses,
                current_error_matched_pairs=sum(p['matched'] for p in bank['matched_pairs']),
                response_identifying_pairs=sum(p['response_identifying_pair'] for p in bank['matched_pairs']),
                AA_remaining_calls=aa['remaining_denoise'], native_restored_remaining_calls=native_restored['remaining_denoise'],
                AA_terminal_cap=aa['trace'][-1]['cap_reached'], native_terminal_cap=native_restored['trace'][-1]['cap_reached'],
                selected_scores=choices['selections'],
                restored_control_work=dict(global_eligible=aa['global_eligible_tiles'], global_kept=aa['global_kept_tiles'],
                    global_skipped=aa['global_skipped_tiles'], local_eligible=aa['local_eligible_tiles'],
                    local_kept=aa['local_kept_tiles'], local_skipped=aa['local_skipped_tiles'],
                    denominator=aa['work_denominator']),
                allkept_native_current_next_contrast=controls['allkept_vs_native'],
                factorial_contrasts=None)
            anchors.append(row)
            qualification.append(dict(id=receipt['id'], anchor=boundary, passed=True,
                scope='native first-canvas stop and commit only; zero emitted tokens at anchor',
                storages=snap['storages'], tensor_views=snap['tensors'], alias_groups=len(snap['alias_groups']),
                live_metadata_checked=True, repeated_native_allkept_exact=True,
                post_perturbation_controls_exact=True, new_measurement_disabled_parity=True,
                full_request_qualified=False, GQA=controls['gqa_qualification']))
        for p in item.rglob('*.json'):
            hashes[str(p.relative_to(root))] = sha(p)
    assert len(anchors) == 2 and len(incomplete) == 2
    assert measured['native_winner_mismatches'] == 0
    assert exit_receipt['allocated_worker_seconds'] <= 7200 and exit_receipt['forwards'] <= 10000
    summary = dict(status='partial_stage_qualification_stop', protocol_sha256=sha(root / 'protocol.json'),
        source_checkpoint=read(root / 'launch_receipt.json')['source_checkpoint'],
        planned_development_anchors=4, completed_anchor_receipts=2, incomplete_anchors=incomplete,
        anchors=anchors, candidate_records=16, feasible_records=4, unique_support_geometries=8,
        unique_feasible_support_geometries=2, matched_pairs=0, response_identifying_pairs=0,
        distinct_selected_supports=0, selected_factorial_branches=0,
        candidate_full_requests=0, official_task_outcomes=None, method_promoted=False,
        development_identification_gate_passed=False, hypothesis_tested=False,
        interpretation='Non-identifying bank plus full-request qualification failure; neither response mechanism rejection nor trajectory benefit.',
        resource=exit_receipt,
        limitations=[
            'One previously exposed AIME identity, seed42; two anchors are not two independent questions.',
            'HumanEval124 anchor4 full-request equality bundle failed; differing fields were not persisted. No full-request certificate.',
            'HumanEval anchor6 not reached; all confirmation, LongBench and scored finalists deferred before generation.',
            'No selected SA/AS/SS outcomes: causal factorial contrasts unavailable, not zero.',
            'Four first-pulse current-call records exist only; no future effect or sparse stopping claim.',
            'Acceptance is device reconstructed; zero threshold discrepancies do not certify compiled acceptance equality.',
            'sampled_canvas is the native emitted buffer, not captured multinomial proposals; use canvas_after for actual mutated canvas.',
            'Unrestored-native physical counters include pre-anchor calls; use restored-control counters for matched work.',
            'First-pulse and perturbation execution metadata were not saved; intended single-rectangle deletion is source-enforced but per-pulse physical receipts are incomplete.',
            'No clean timing, selector overhead, task score, accuracy interval, FULL continuation or long-context generalization.',
            'Snapshot metadata/bytes and empirical replay do not certify opaque runtime state generally.'
        ])
    write(out / 'summary.json', summary)
    write(out / 'input_hashes.json', hashes)
    write(out / 'qualification_summary.json', dict(completed_anchors=qualification,
        measurement_totals_over_repeated_controls=dict(measured),
        acceptance_level='same-device/dtype reconstruction from native returned scaled logits; not compiled acceptance buffer',
        full_request_qualification_passed=False))
    decisions = dict(ResponseGuard=dict(decision='no_go', reason='No identifying matched pairs or score-selected future branches; no proxy or production selector.'),
        MarginGuard=dict(decision='no_go', reason='Current-call decision measurements only; no distinct replicated useful future effect; no exhaustive oracle.'),
        ValidityReuse=dict(decision='parked', reason='Norm-only certificate remains unsupported and outside this leaf.'),
        confirmation=dict(decision='no_go', reason='Development bank failed identification gate; qualification stop before remaining anchors.'),
        next_experiment=dict(decision='review_first', recommendation='Qualification-only HE124 anchor4 replay with each equality field and first divergent call persisted before comparison; no support bank or confirmation. Separately design a frozen bank with >=4 feasible within-unit alternatives, without relaxing guards based on future outcomes.'),
        native_trajectory_acceleration=False, sparse_accuracy_preservation=False)
    write(out / 'method_decisions.json', decisions)
    # Scientific plot: feasibility and current-only operator statistics, no causal labels.
    os.environ.setdefault('MPLCONFIGDIR', str(root / 'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar([0, 1], [8, 8], color='lightgray', label='Enumerated')
    axes[0].bar([0, 1], [2, 2], color='tab:blue', label='Feasible')
    axes[0].axhline(4, linestyle='--', color='tab:red', label='Ranking minimum')
    axes[0].set_xticks([0, 1], ['AIME18 after5', 'AIME18 after9'])
    axes[0].set_ylabel('Support count'); axes[0].legend(fontsize=8)
    for row in anchors:
        for c in row['candidate_rows']:
            axes[1].scatter(c['current_error'], c['response_score'],
                marker='o' if row['anchor'] == 5 else '^',
                color='tab:blue' if c['feasible'] else 'gray')
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_xlabel('Current worst-row relative output bias')
    axes[1].set_ylabel('Residual-response proxy')
    fig.suptitle('Partial development: bank does not identify a ranking effect')
    fig.tight_layout(); fig.savefig(out / 'bank_identification.png', dpi=180)
    fig.savefig(out / 'bank_identification.pdf'); plt.close(fig)
    write(out / 'readout_complete.json', dict(passed=True,
        scope='descriptive analysis of two completed anchors; experimental stage remains partial',
        stage_complete=False, summary_sha256=sha(out / 'summary.json'),
        analysis_source_sha256=sha(Path(__file__)), protocol_sha256=sha(root / 'protocol.json')))
    print(json.dumps({k: summary[k] for k in ['status', 'completed_anchor_receipts', 'candidate_records',
        'feasible_records', 'matched_pairs', 'selected_factorial_branches', 'hypothesis_tested']}))


if __name__ == '__main__':
    main()
