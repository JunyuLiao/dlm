"""Independent completeness, archive parity, and paired dense-ablation analysis."""
import argparse
import json
from pathlib import Path
import numpy as np
from .experiment import atomic, sha


def run(root, probe):
    cfg = json.loads((root/'configuration.json').read_text())
    records = [json.loads(p.read_text()) for p in (root/'timing').glob('*.json')]
    expected = {(repeat, i, m) for repeat in range(cfg['repeats']) for i in range(130) for m in cfg['methods']}
    assert {(r['repeat'], r['index'], r['method']) for r in records} == expected
    assert len(records) == len(expected)
    assert all(sha(Path(p)) == digest for p, digest in cfg['source_hashes'].items())
    base = Path(__file__).resolve().parents[2]/'results'/'value_direction_sparsity_steps_v2'
    manifest = json.loads((base/'manifest.json').read_text())
    first = {m: sorted([r for r in records if r['repeat'] == 0 and r['method'] == m], key=lambda r: r['index']) for m in cfg['methods']}
    parity = {}
    for m, rows in first.items():
        assert [r['id'] for r in rows] == [r['id'] for r in manifest]
        old = {r['id']: r for r in (json.loads(p.read_text()) for p in (base/'adaptive/shards'/m).glob('*.json'))}
        if old:
            parity[m] = dict(exact_tokens=sum(r['tokens'] == old[r['id']]['completion_tokens'] for r in rows),
                             exact_steps=sum(r['steps'] == old[r['id']]['steps'] for r in rows))
            assert parity[m] == dict(exact_tokens=130, exact_steps=130)
    rng = np.random.default_rng(1729)
    samples = rng.integers(0, 130, size=(10000, 130))
    comparisons = {}
    for m, rows in first.items():
        native = first['native_dense']
        delta = np.array([r['steps']-n['steps'] for r, n in zip(rows, native)])
        quality = np.array([r['score']-n['score'] for r, n in zip(rows, native)])
        comparisons[m] = dict(mean_extra_steps=float(delta.mean()),
            extra_steps_ci95=np.quantile(delta[samples].mean(1), [.025,.975]).tolist(),
            fewer_steps=int((delta<0).sum()), equal_steps=int((delta==0).sum()), more_steps=int((delta>0).sum()),
            exact_native_outputs=sum(r['tokens']==n['tokens'] for r,n in zip(rows,native)),
            accuracy_delta_pp=100*float(quality.mean()),
            accuracy_delta_pp_ci95=(100*np.quantile(quality[samples].mean(1), [.025,.975])).tolist())
    sparsity = {}
    for m in ('gaussian32_s50', 'blasst_s50'):
        audits = [json.loads(p.read_text()) for p in (root/'audit'/m).glob('*.json')]
        assert len(audits) == 130
        byid = {r['id']: r for r in first[m]}
        archived = {r['id']: r for r in (json.loads(p.read_text()) for p in (base/'adaptive/shards'/m).glob('*.json'))}
        for r in audits:
            assert (r['tokens'],r['steps']) == (byid[r['id']]['tokens'],byid[r['id']]['steps'])
            assert r['counts'] == archived[r['id']]['counts']
        sparsity[m] = {k:sum(r['counts'][k]['skipped'] for r in audits)/sum(r['counts'][k]['eligible'] for r in audits) for k in ('whole','local','global')}
    result = dict(complete=True, timed_generations=len(records), archive_parity=parity,
                  dense_comparisons=comparisons, achieved_sparsity=sparsity, source_hashes_verified=True)
    if probe:
        result['dense_local_probe'] = json.loads((probe/'summary.json').read_text())
        probe_rows = [json.loads(p.read_text()) for p in (probe/'timing').glob('*.json')]
        assert len(probe_rows) == 520
        probe_first = {m: sorted([r for r in probe_rows if r['repeat'] == 0 and r['method'] == m], key=lambda r:r['index'])
                       for m in ('native_dense', 'native_local_custom_global')}
        for old,new in zip(first['native_dense'], probe_first['native_dense']):
            assert (old['id'],old['tokens'],old['steps']) == (new['id'],new['tokens'],new['steps'])
        for m,rr in probe_first.items():
            assert len(rr) == 130
            late = sorted([r for r in probe_rows if r['repeat'] == 1 and r['method'] == m], key=lambda r:r['index'])
            assert len(late) == 130
            for a,b in zip(rr,late):
                assert (a['id'],a['tokens'],a['steps']) == (b['id'],b['tokens'],b['steps'])
        delta = np.array([h['steps']-n['steps'] for h,n in zip(probe_first['native_local_custom_global'],probe_first['native_dense'])])
        result['dense_local_probe_comparison'] = dict(mean_extra_steps=float(delta.mean()),
            extra_steps_ci95=np.quantile(delta[samples].mean(1), [.025,.975]).tolist(),
            exact_native_outputs=sum(h['tokens']==n['tokens'] for h,n in zip(probe_first['native_local_custom_global'],probe_first['native_dense'])))
    atomic(root/'independent_audit.json', result)
    atomic(root/'analysis_provenance.json', dict(source=str(Path(__file__).resolve()),
           sha256=sha(Path(__file__)), bootstrap_seed=1729, bootstrap_samples=10000,
           probe=None if probe is None else str(probe.resolve())))
    lines = ['# Dense trajectory interpretation', '',
             'The native/custom dense comparison originally changed both masks and numerical operations. The new native-mask ablation removes the custom extra sliding-window restriction and keeps the custom arithmetic and native stopping criterion.', '',
             '| Custom condition vs native | Mean extra steps | Paired 95% CI | Fewer / same / more steps | Exact output sequences |',
             '|---|---:|---|---|---:|']
    for m,r in comparisons.items():
        if m == 'native_dense':
            continue
        lo,hi = r['extra_steps_ci95']
        lines.append(f"| {m} | {r['mean_extra_steps']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {r['fewer_steps']} / {r['equal_steps']} / {r['more_steps']} | {r['exact_native_outputs']}/130 |")
    lines += ['',
        'Arithmetic still differs: custom QK explicitly rounds the dot product to BF16 and rounds again after scaling; the online softmax/PV path rounds per-tile weighted probabilities and rescales accumulated output. Matching structural masks therefore does not establish bitwise native parity.', '',
        'For exact native dense behavior, select the original native attention implementation when the configured policy retains all tiles. This can be selected once when binding the policy, adding no sparse-kernel work. Its speed is the measured native baseline; it is not a claim that custom arithmetic became bitwise identical.', '',
        'The dense-only hybrid probe uses native SDPA on local layers and the unchanged custom kernel on global layers. It preserves the optimized global schedule while eliminating custom local attention overhead and local numerical/mask differences. This is an experimental dense dispatch choice, not a replacement for the frozen sparse policies.', '',
        'A further arithmetic experiment could retain FP32 QK scores through scaling, avoiding the two explicit BF16 score conversions, and align online normalization with the native fused path. That may be inexpensive but is untested here: closer arithmetic does not guarantee identical argmax/entropy decisions or fewer iterations. Any sparse use must requalify routing and achieved sparsity because thresholds were calibrated for the existing score convention.', '',
        'Do not force four iterations or relax the entropy/stability criterion to claim numerical equivalence. The unchanged stopper requires stability across all 256 canvas positions and mean entropy below its threshold; small logit changes can alter the trajectory, including beyond returned output tokens.']
    if probe:
        lines += ['', 'Dense-only local/native probe (separate contemporaneous native baseline):', '',
                  '| Method | Steps | Accuracy | E2E / API TTFT (ms) | Amortized TPT (ms) | Speedup |',
                  '|---|---:|---:|---:|---:|---:|']
        for m,r in result['dense_local_probe'].items():
            lines.append(f"| {m} | {r['mean_steps']:.3f} | {r['accuracy']:.2%} | {r['mean_seconds']*1000:.2f} | {r['tpt_ms']:.3f} | {r['speedup_vs_native']:.3f}× |")
    (root/'dense_analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--probe', type=Path)
    a = p.parse_args()
    run(a.root, a.probe)
