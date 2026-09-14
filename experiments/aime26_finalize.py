"""Independent provenance/attainment audit and final interpretation."""
import json
import math
from pathlib import Path

from experiments.diffusion_gemma_aime26.run import ROOT, _write, aggregate, sha
from experiments.diffusion_gemma_aime26.report import report
from experiments.diffusion_gemma_aime30.protocol import numeric_score


def finalize():
    root = ROOT / 'refined'
    report(root)
    summary = json.loads((root / 'summary.json').read_text())
    policy = summary['policy']
    manifest = json.loads((root / 'manifest.json').read_text())
    provenance = json.loads((root / 'refinement_provenance.json').read_text())
    issues = list(summary['audit']['violations'])
    if not summary['audit']['complete']:
        issues.append('Incomplete final evaluation')
    if provenance['code_sha256'] != sha(Path('experiments/aime26_refinement.py').read_bytes()):
        issues.append('Refinement source mismatch')
    config = json.loads((ROOT / 'experiment.json').read_text())
    smoke = json.loads((ROOT / 'smoke.json').read_text())
    if not smoke['passed'] or smoke['fingerprint'] != policy['fingerprint']:
        issues.append('Missing matching CUDA parity smoke')
    if len(manifest) != 30 or len({r['source_id'] for r in manifest}) != 30:
        issues.append('Not the full unique 30-question benchmark')
    for path, digest in config['code'].items():
        if sha(Path(path).read_bytes()) != digest:
            issues.append(f'Original inference source changed: {path}')
    runtime = json.loads((root / 'model_runtime.json').read_text())
    types = runtime['model']['text_config']['layer_types']
    calibration_ids = {r['source_id'] for r in manifest if r['calibration']}
    if calibration_ids != set(provenance['calibration_ids']) or len(calibration_ids) != 6:
        issues.append('Wrong calibration selection')
    attainment = {}
    for target, selected in policy['targets'].items():
        name = f'blasst_s{int(float(target)*100)}'
        rows = [json.loads((root / name / f"{r['source_id']}.json").read_text())
                for r in manifest if r['calibration']]
        stats = aggregate(c for r in rows for c in r['calls'])
        measured = {k: stats[k]['full_tile_sparsity'] for k in ('global', 'local')}
        if any(not math.isclose(measured[k], selected['measured'][k], abs_tol=1e-12) for k in measured):
            issues.append(f'{target}: calibration measurements do not reproduce')
        attainment[target] = dict(measured=measured,
            within_two_points=all(abs(s-float(target)) <= .02 for s in measured.values()))
    lengths = {}
    reference_metadata = None
    ceiling = {k: dict(eligible_tiles=0, mandatory_tiles=0, skipped_tiles=0, calls=0,
                       verified_calls=0) for k in ('global', 'local')}
    for name in summary['audit']['counts']:
        records = [json.loads((root / name / f"{r['source_id']}.json").read_text()) for r in manifest]
        lengths[name] = dict(mean_output_tokens=sum(len(r['completion_tokens']) for r in records)/30,
                            budget_terminated=sum(r['termination_reason']=='length' for r in records))
        for row, r in zip(manifest, records):
            if numeric_score(r['prediction'], row['expected']) != r['score']:
                issues.append(f'{name}/{r["id"]}: score does not reproduce')
            metadata = r['generation_metadata']
            invariant = {k: metadata[k] for k in ('denoising_configuration', 'native_canvas_length', 'sampling', 'thinking')}
            if reference_metadata is None:
                reference_metadata = invariant
            elif invariant != reference_metadata:
                issues.append(f'{name}/{r["id"]}: decoding settings differ')
            if name == 'blasst_s90':
                for c in r['calls']:
                    v = ceiling[c['attention_type']]
                    v['calls'] += 1
                    # In these short, fully valid calls every physical tile has
                    # 128 valid query votes. Each query tile must keep its first
                    # valid KV tile under the previous-maximum extension.
                    query_rows = c['retained_attention_mass_rows']
                    if (c['query_length'] % 128 == 0 and query_rows % 128 == 0
                            and c['valid_row_votes'] == 128*c['eligible_tiles']):
                        mandatory = query_rows // 128
                        v['verified_calls'] += 1
                        v['eligible_tiles'] += c['eligible_tiles']
                        v['mandatory_tiles'] += mandatory
                        v['skipped_tiles'] += c['skipped_tiles']
            for c in r['per_layer']:
                expected = 'local' if types[c['layer']] == 'sliding_attention' else 'global'
                if c['attention_type'] != expected:
                    issues.append(f'{name}/{r["id"]}: layer classification')
    for v in ceiling.values():
        v['all_calls_covered'] = v['calls'] == v['verified_calls']
        v['trajectory_ceiling'] = 1-v['mandatory_tiles']/v['eligible_tiles'] if v['eligible_tiles'] else None
        v['ceiling_reached'] = v['skipped_tiles'] == v['eligible_tiles']-v['mandatory_tiles']
    audit = dict(complete=not issues, violations=issues, calibration_attainment=attainment,
                 saturated_policy_geometry=ceiling,
                 output_lengths=lengths, full_evaluation_count=150,
                 scope='Calibration uses six fixed prompts; final accuracy is full30 and heldout24; unattained targets explicitly disclosed')
    _write(root / 'completion_audit.json', audit)
    lines = ['','## Calibration interpretation and limitations','',
        'The original exponential fits included saturated observations. Their predictions were poor; the recorded empirical dense-margin correction and sparse-only verification therefore materially determine the deployed scales. This is paper-inspired calibration with explicit corrections, not a claim that the exponential model alone fits these data well.', '',
        'The follow-up preserves all original results. It tests fixed additional scale pairs using only the six calibration prompts, selects solely by the maximum local/global sparsity error, freezes the policy, and then evaluates the remaining 24 prompts. Calibration generations are reused in the full30 table. The final answers and correctness labels are not used to select thresholds.', '',
        'Missing a target is not evidence of implementation failure, and a plateau on tested sparse trajectories is not proof that every possible policy is bounded there. Very aggressive masking can produce malformed, short completions; shorter sequences have fewer KV tiles over which to amortize mandatory first-tile retention. No generation budget or stopping rule was changed to manufacture higher sparsity.', '',
        '| Condition | Mean generated tokens | Budget-limited outputs |',
        '|---|---:|---:|']
    for name, v in lengths.items():
        lines.append(f"| {name} | {v['mean_output_tokens']:.1f} | {v['budget_terminated']}/30 |")
    lines += ['', 'The 90%-target trajectory first-tile bound (not a universal bound on other trajectories):', '']
    for kind, v in ceiling.items():
        lines.append(f"- {kind}: {v['verified_calls']}/{v['calls']} calls satisfy full 128-query tile validity; bound={v['trajectory_ceiling']}; reached={v['ceiling_reached']}.")
    lines += ['', 'Scoring reuses the existing numeric final-answer extractor (including its last-number fallback). Malformed outputs and extraction provenance remain available in every raw shard. These are reference-mask results, not speed measurements.', '',
              f'Independent completion audit: {audit}', '']
    path = root / 'report.md'
    path.write_text(path.read_text() + '\n'.join(lines))
    if issues:
        raise RuntimeError(issues)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    finalize()
