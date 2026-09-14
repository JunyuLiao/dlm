"""Readable, separately audited companion figures; never changes frozen results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ORDER = (
    'dense', 'blasst_original_s50', 'blasst_aggressive_s50', 'value_s50',
    'mass_s50', 'mass_value_s50', 'risk_s50', 'aligned_s50', 'centered_s50',
    'mass_exact_s50', 'no_value_control_s50', 'compensate_s50', 'zero_pv_s50',
)
LABELS = (
    'Dense', 'BLASST original', 'BLASST aggressive', 'Value (vector mean)',
    'Mass only', 'Mass x value (RMS)', 'Output risk (mean)', 'Token aligned',
    'Output centered', 'Exact online mass', 'No-value control',
    'Mean compensation [PV]', 'Zero-PV [PV]',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checked(root):
    audit = json.loads((root / 'audit.json').read_text())
    if not audit['complete'] or audit['completed'] != 325:
        raise ValueError('Companion figures require the complete frozen review')
    inputs = {}
    for name in ('summary.json', 'per_sample.json'):
        if digest(root / name) != audit['artifacts'][name]:
            raise ValueError(f'Unaudited input: {name}')
        inputs[name] = json.loads((root / name).read_text())
    rows, raw = inputs['summary.json'], inputs['per_sample.json']
    if Counter((r['benchmark'], r['condition']) for r in raw) != Counter({
        (b, c): n for b, n in (('aime26', 10), ('longbench_v2', 15)) for c in ORDER
    }):
        raise ValueError('Incorrect sample-condition coverage')
    if len(rows) != 26 or len({(r['benchmark'], r['condition']) for r in rows}) != 26:
        raise ValueError('Duplicate or missing summary rows')
    for row in rows:
        for kind in ('overall', 'global', 'local'):
            ratio = row[f'{kind}_skipped'] / row[f'{kind}_eligible']
            if abs(ratio - row[f'{kind}_physical_sparsity']) > 1e-12:
                raise ValueError('Physical sparsity is not count weighted')
    return rows, raw, audit


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows, raw, audit = load_checked(root)
    dest = root / 'readable_figures'
    dest.mkdir(exist_ok=True)
    index = {(r['benchmark'], r['condition']): r for r in rows}
    names = []
    for benchmark in ('aime26', 'longbench_v2'):
        group = [index[benchmark, c] for c in ORDER]
        fig, axes = plt.subplots(1, 6, figsize=(20, 8), sharey=True)
        for ax in axes:
            ax.set_ylim(12.7, -.7)
            ax.set_yticks(range(13), LABELS)
            ax.grid(axis='x', alpha=.2)
            ax.tick_params(axis='y', length=0, labelsize=10)
            for y in (2.5, 10.5):
                ax.axhline(y, color='#cccccc', linewidth=.7)
        for k, color, offset in zip(('overall', 'global', 'local'), ('#222222', '#1677b8', '#df7a16'), (-.18, 0, .18)):
            axes[0].scatter([100*r[f'{k}_physical_sparsity'] for r in group],
                            [i+offset for i in range(13)], label=k, color=color, s=24)
        axes[0].set(title='Physical deletion (%)', xlim=(-3, 63))
        axes[0].axvline(50, color='grey', linestyle=':', linewidth=1)
        axes[0].legend(loc='lower right', fontsize=8)
        metrics = (
            ('accuracy', 100, 'Accuracy (%)', (-4, 115)),
            ('overall_mass', 100, 'Exact-PV mass (%)', (-4, 115)),
            ('overall_relative_error', 1, 'Relative output error', (-.02, .72)),
            ('token_agreement', 100, 'Token agreement (%)', (-4, 115)),
            ('overall_pv_omission', 100, 'PV omission (%)', (-3, 65)),
        )
        for ax, (field, scale, title, limits) in zip(axes[1:], metrics):
            for y, r in enumerate(group):
                val = scale*r[field]
                color = '#9c4f96' if r['condition'] in ORDER[-2:] else '#1677b8'
                ax.scatter(val, y, color=color, s=28)
                label = f"{r['correct']:g}/{r['count']}" if field == 'accuracy' else f'{val:.3f}' if scale == 1 else f'{val:.1f}'
                ax.annotate(label, (val, y), xytext=(5, 0), textcoords='offset points', va='center', fontsize=9)
            ax.set(title=title, xlim=limits)
        fig.suptitle(f'{benchmark}: complete quick review, 50% target (not a sparsity sweep)', fontsize=15)
        caveat = 'AIME: exploratory, 10 previously exposed non-calibration problems.' if benchmark == 'aime26' else 'LongBench v2: 5/domain, 7/15 prompts truncated at 32k; dense 9/15 answers unparseable at the 128-token limit.'
        fig.text(.02, .055, caveat, fontsize=10)
        fig.text(.02, .025, '[PV] methods preserve 100% denominator mass and delete no physical tiles. Mass/error use exact attention on each run\'s own states. No speedup claim.', fontsize=10)
        fig.tight_layout(rect=(0, .085, 1, .94))
        name = f'{benchmark}_comparison.png'
        fig.savefig(dest / name, dpi=160)
        plt.close(fig)
        names.append(name)
    anomalies = []
    for r in raw:
        tokens = r['completion_tokens']
        tail = 0
        for token in reversed(tokens):
            if token != tokens[-1]:
                break
            tail += 1
        anomalies.append({k: r[k] for k in ('id', 'benchmark', 'condition', 'accuracy', 'termination_reason', 'output_length', 'unparsed_answer')} | dict(final_same_token_run=tail))
    (dest / 'output_diagnostics.json').write_text(json.dumps(anomalies, sort_keys=True, indent=2) + '\n')
    names.append('output_diagnostics.json')
    result = dict(complete=True, completed=325, inference=False, scientific_settings_changed=False,
                  sources={n: audit['artifacts'][n] for n in ('summary.json', 'per_sample.json')},
                  frozen_audit_sha256=digest(root / 'audit.json'), script_sha256=digest(Path(__file__)),
                  artifacts={n: digest(dest / n) for n in names})
    (dest / 'audit.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, default=Path('results/diffusion_gemma_value_aware_followup/quick50_review'))
    print(json.dumps(render(parser.parse_args().review), sort_keys=True))
