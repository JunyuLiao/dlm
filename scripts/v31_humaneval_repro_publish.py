"""Publish verified numeric evidence from the completed baseline diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def publish(root, destination):
    marker = root/'qualification/attempt001/baseline_runtime_qualified.json'
    qualification = read(marker)
    assert qualification['status'] == 'baseline_runtime_qualified'
    assert qualification['full_runs_qualified'] is False
    assert qualification['all_requested_methods_qualified'] is False
    manifest = root/'qualification/attempt001/manifest.json'
    assert sha(manifest) == qualification['manifest_sha256']
    assert read(manifest)['protocol_sha256'] == sha(root/'protocol.json')
    assert all(row['passed'] for row in qualification['comparisons'])
    for job in qualification['jobs']:
        leaf = root/'diagnosis'/job['job']
        assert sha(leaf/'attempt_complete.json') == job['marker_sha256']
        completed = read(leaf/'attempt_complete.json')
        assert completed['status'] == 'complete'
        for name, digest in completed['files_sha256'].items():
            assert sha(leaf/name) == digest
        assert not job['bookend_changes'] and job['measured_captures'] == 0
        assert job['requested_v31_budget_scope_ok']
    repeat = qualification['unobserved_native_repeat']
    assert repeat['pairs'] == 12 and not any(repeat['differences'].values())
    for name, digest in repeat['markers_sha256'].items():
        leaf = root/'diagnosis'/name
        assert sha(leaf/'attempt_complete.json') == digest
        for rel, expected in read(leaf/'attempt_complete.json')['files_sha256'].items():
            assert sha(leaf/rel) == expected
    operator = read(root/'operator_diagnostics/attempt002/summary.json')
    assert operator['native_forwards'] == 0
    link = read(root/'analysis/attempt001/reduction_replay_native_link.json')
    assert sha(root/'operator_diagnostics/attempt002/operator_complete.json') == link['operator_complete_sha256']
    assert link['prior_segment_operands_valid_equal']
    budget = read(root/'gpu_budget.json')
    diagnosis = [r for r in budget['reservations'] if r['phase'] == 'diagnosis']
    assert all('elapsed_s' in r for r in diagnosis)
    result = dict(status='complete_baseline_diagnostic_only',
        protocol_sha256=sha(root/'protocol.json'), manifest=read(manifest),
        baseline_marker_sha256=sha(marker), qualification=qualification,
        operator=operator, operator_native_link=link,
        gpu_reservation_s=sum(r['elapsed_s'] for r in diagnosis),
        diagnostic_cap_s=budget['caps_s']['diagnosis'],
        policy=dict(native_selector_source='20e6c0be', numerical_policy_source='10341ab2',
            explicit_model_and_unchanged_native_helpers=True,
            horizontal_combo_fusion=False, combo_benchmarking=False,
            native_full_piecewise_graphs=True,
            native_sampler_formulas_and_stopping_preserved=True),
        limits=['six previously examined questions; baseline methods only',
            'runtime reduction configuration is a demonstrated numerical mechanism, not proof of an exclusive cause',
            'failed segment002 supports bounded pre-defect operator evidence only',
            'full164 requested-method reproducibility and clean performance remain separate gates',
            'no corrected accuracy, selector advantage, or speedup is established by this diagnostic'])
    serialized = json.dumps(result, indent=2, sort_keys=True)+'\n'
    # This exporter has an allowlisted numeric/hash schema and never copies
    # native records, completions, activation arrays, or RNG state bytes.
    for forbidden in ('/home/', '/media/', 'token_ids', '"completion":', '"prompt":'):
        assert forbidden not in serialized, forbidden
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'summary.json').write_text(serialized)
    (destination/'summary.md').write_text(
        '# HumanEval baseline reproducibility diagnostic\n\n'
        'A runtime choice of reduction configuration changes rounding in fused '
        'post-MLP normalization on identical inputs. R0_BLOCK512 versus1024 '
        'changes11 BF16 layer-output elements,86596 internal FP32 elements and10 '
        'BF16 elements entering the next Q/K/V projection (481536 valid elements). '
        'Each configuration repeats exactly and matches its corresponding saved output.\n\n'
        'The qualified policy explicitly applies deterministic Inductor options '
        'to the model and three unchanged native helper Python functions. '
        'Horizontal combo fusion and benchmarking are disabled; native FULL/PIECEWISE '
        'graphs and sampler formulas/stopping remain. All53 recorded kernels '
        'receipt deterministic settings. Environment-only and model-only failures '
        'are retained.\n\n'
        'Independent unobserved FULL engines match12/12 output/N/C/T pairs. '
        'Eleven further dense/V31 engines, both thinking modes, pass all declared '
        'full pre/post sampler-state, valid-history, emission and RNG comparisons; '
        'bookends match and measured captures are zero. This is a complete '
        'six-question baseline diagnostic, not a164-question accuracy study.\n\n'
        f'Diagnosis reservation: {result["gpu_reservation_s"]:.3f}/'
        f'{result["diagnostic_cap_s"]} seconds. Original results and older pauses '
        'are preserved. Corrected K256/K128 matrices require their own method, '
        'replication and official-scoring completion seals.\n')
    files = {p.name: sha(p) for p in destination.iterdir() if p.is_file()}
    (destination/'verified_complete.json').write_text(json.dumps(dict(
        status=result['status'], source_protocol_sha256=result['protocol_sha256'],
        source_baseline_marker_sha256=sha(marker), exporter_sha256=sha(Path(__file__)),
        files_sha256=files), indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status=result['status'], jobs=len(qualification['jobs']),
                         unobserved_pairs=repeat['pairs'], files_sha256=files)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    publish(args.root,args.destination)
