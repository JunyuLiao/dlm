"""Additive paired rank comparisons from the completed, audited AIME sweep."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci

ROOT = Path('results/diffusion_gemma_jl_aime_gaussian_dimensions_v8')
PAIRS = tuple((f'jl_gaussian_r{rank}_s50', ref) for rank in (8, 16, 24)
              for ref in ('jl_gaussian_r32_s50', 'full_centered_s50')) + (
    ('jl_gaussian_r32_s50', 'full_centered_s50'),)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(raw):
    compared = []
    for split, count in (('full', 30), ('noncalibration24', 24), ('calibration6', 6)):
        for candidate, reference in PAIRS:
            groups = []
            for condition in (candidate, reference):
                rows = [r for r in raw if r['benchmark'] == 'aime26' and r['condition'] == condition
                        and (split == 'full' or bool(r['calibration']) == (split == 'calibration6'))]
                group = {r['id']: r for r in rows}
                if len(rows) != count or len(group) != count:
                    raise ValueError('Require every question exactly once in the requested split')
                groups.append(group)
            a, b = groups
            if set(a) != set(b):
                raise ValueError('Rank comparisons are not paired on identical IDs')
            delta = [a[i]['accuracy']-b[i]['accuracy'] for i in sorted(a)]
            lo, hi = paired_bootstrap_ci(delta)
            row = dict(split=split, candidate=candidate, reference=reference, count=count,
                candidate_correct=sum(r['accuracy'] for r in a.values()),
                reference_correct=sum(r['accuracy'] for r in b.values()),
                delta_pp=100*sum(delta)/count, paired_ci95_pp=[100*lo, 100*hi])
            gaps = []
            for kind in ('overall', 'global', 'local'):
                sparsities = []
                for group in groups:
                    eligible = sum(r['aggregates'][kind]['eligible'] for r in group.values())
                    skipped = sum(r['aggregates'][kind]['skipped'] for r in group.values())
                    if not 0 <= skipped <= eligible or eligible <= 0:
                        raise ValueError('Invalid physical tile counts')
                    sparsities.append(skipped/eligible)
                row[f'candidate_{kind}_sparsity_pct'] = 100*sparsities[0]
                row[f'reference_{kind}_sparsity_pct'] = 100*sparsities[1]
                gaps.append(sparsities[0]-sparsities[1])
                row[f'{kind}_sparsity_gap_pp'] = 100*gaps[-1]
            row['within2pp_all_types'] = all(abs(g) <= .02+1e-12 for g in gaps)
            compared.append(row)
    return compared


def run(root=ROOT):
    audit = json.loads((root/'audit.json').read_text())
    verification = json.loads((root/'regeneration_verification.json').read_text())
    if not (audit['complete'] and audit['completed'] == audit['expected'] == 270
            and not audit['missing'] and not audit['violations'] and verification['passed']
            and verification['completed'] == 270 and verification['inference_performed'] is False
            and verification['audit_sha256'] == digest(root/'audit.json')):
        raise ValueError('Require the independently verified complete270-output experiment')
    for name, expected in audit['artifacts'].items():
        if digest(root/name) != expected:
            raise ValueError(f'Canonical report artifact changed: {name}')
    rows = analyze(json.loads((root/'per_sample.json').read_text()))
    output = root/'dimension_pairwise.json'
    output.write_text(json.dumps(rows, indent=2, sort_keys=True)+'\n')
    csv_path = root/'dimension_pairwise.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    text = '# Direct AIME26 dimension comparisons\n\n'
    text += 'Positive deltas favor the candidate. Same frozen prompts, seeds and decoding; 50% target, actual physical sparsity from summed tile counts. Paired prompt-bootstrap95% intervals condition on the fixed policies and seed, are exploratory and not multiplicity-corrected. The full30 includes six calibration questions; the additional24 were previously examined and are not fresh held-out confirmation. No inference, threshold fitting or projection selection occurs here.\n\n'
    for split in ('full', 'noncalibration24', 'calibration6'):
        text += f'## {split}\n\n'
        text += '| Candidate − reference | Correct | Accuracy delta (pp),95% CI | Sparsity gap overall/global/local (pp) | Within2pp,all types |\n| --- | --- | --- | --- | --- |\n'
        for row in (r for r in rows if r['split'] == split):
            lo, hi = row['paired_ci95_pp']
            gaps = '/'.join(f"{row[k+'_sparsity_gap_pp']:+.2f}" for k in ('overall','global','local'))
            text += f"| {row['candidate']} − {row['reference']} | {row['candidate_correct']:g}/{row['count']} vs {row['reference_correct']:g}/{row['count']} | {row['delta_pp']:+.2f},[{lo:+.2f},{hi:+.2f}] | {gaps} | {row['within2pp_all_types']} |\n"
        text += '\n'
    text += 'One question changes headline accuracy by3.33percentage points. Changing rank also changes the deterministic Gaussian matrix; these are fixed-seed results, not an average over projection directions. Consult report.md for retained mass, agreement, shared-state diagnostics and all cached baselines.\n'
    report = root/'dimension_pairwise.md'; report.write_text(text)
    sources = {str(p):digest(p) for p in (Path(__file__), Path('tests/test_jl_aime_dimensions_analysis.py'),
        root/'audit.json', root/'regeneration_verification.json', root/'per_sample.json')}
    result = dict(passed=True, comparisons=len(rows), inference_performed=False,
        threshold_selection_performed=False, sources=sources,
        artifacts={str(p.relative_to(root)):digest(p) for p in (output,csv_path,report)})
    (root/'dimension_pairwise_audit.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=ROOT)
    print(json.dumps(run(parser.parse_args().output)))
