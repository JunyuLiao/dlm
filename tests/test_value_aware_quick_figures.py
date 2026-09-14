import json

import pytest

from experiments.diffusion_gemma_value_aware_followup import quick_review_figures as figures


def fixture(root, mutation=None):
    rows, raw = [], []
    for benchmark, count in (('aime26', 10), ('longbench_v2', 15)):
        for condition in figures.ORDER:
            row = dict(benchmark=benchmark, condition=condition)
            for kind in ('overall', 'global', 'local'):
                row.update({f'{kind}_eligible': 110, f'{kind}_skipped': 95,
                            f'{kind}_physical_sparsity': 95 / 110})
            rows.append(row)
            raw.extend(dict(benchmark=benchmark, condition=condition) for _ in range(count))
    if mutation == 'weighting':
        rows[0]['overall_physical_sparsity'] = (.9 + .5) / 2
    if mutation == 'coverage':
        raw.pop()
    for name, value in (('summary.json', rows), ('per_sample.json', raw)):
        (root / name).write_text(json.dumps(value))
    audit = dict(complete=mutation != 'incomplete', completed=325,
                 artifacts={n: figures.digest(root / n) for n in ('summary.json', 'per_sample.json')})
    (root / 'audit.json').write_text(json.dumps(audit))


def test_companion_reads_only_complete_audited_count_weighted_results(tmp_path):
    fixture(tmp_path)
    rows, raw, _ = figures.load_checked(tmp_path)
    assert len(rows) == 26 and len(raw) == 325
    assert rows[0]['overall_physical_sparsity'] == 95 / 110


@pytest.mark.parametrize('mutation', ['weighting', 'coverage', 'incomplete', 'hash'])
def test_companion_refuses_incomplete_or_changed_inputs(tmp_path, mutation):
    fixture(tmp_path, mutation)
    if mutation == 'hash':
        with (tmp_path / 'summary.json').open('a') as f:
            f.write(' ')
    with pytest.raises(ValueError):
        figures.load_checked(tmp_path)
