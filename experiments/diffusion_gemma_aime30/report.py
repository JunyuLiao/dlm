"""Regenerate AIME numerical scores, weighted sparsity, and paired results."""
import csv
import json
from pathlib import Path
import numpy as np
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _fingerprint
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import aggregate, token_counts
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci
from .protocol import MODES, conditions, numeric_score, sha


def write_csv(path,rows):
    if not rows: return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def report(root):
    manifest={r['id']:r for r in json.loads((root/'final.json').read_text())}
    policy=json.loads((root/'policy.json').read_text()) if (root/'policy.json').exists() else None
    loaded={c['name']:[json.loads(p.read_text()) for p in (root/'conditions'/c['name']/'shards').glob('*.json')] for c in conditions()}
    dense={r['id']:r for r in loaded['dense']};summary=[];samples=[];layers=[];violations=[]
    for c in conditions():
        name=c['name'];all_rows=loaded[name]
        if len({r['id'] for r in all_rows})!=len(all_rows): violations.append(name+': duplicate IDs')
        config_path=root/'conditions'/name/'config.json'
        config=json.loads(config_path.read_text()) if config_path.exists() else {}
        for r in all_rows:
            m=manifest.get(r['id'])
            if m is None or any(m[k]!=r[k] for k in ('prompt_hash','prompt_token_hash','seed','generation_budget','mode')): violations.append(name+': manifest mismatch')
            if r['fingerprint']!=config.get('fingerprint'): violations.append(name+': fingerprint mismatch')
            if r['method']!=c['method'] or r['target_sparsity']!=c['target'] or r['beta']!=c['beta']: violations.append(name+': threshold metadata mismatch')
            if policy and c['method']=='blasst':
                for kind in ('local','global'):
                    chosen=policy['modes'][r['mode']][kind]['targets'][str(c['target'])]['lambda_value']
                    if r['lambdas'][kind]!=chosen: violations.append(name+': policy mismatch')
                    if any(x['effective_blasst_lambda']!=chosen for x in r['calls'] if x['attention_type']==kind): violations.append(name+': effective lambda mismatch')
            stats=aggregate(r['calls'])
            for key in ('eligible_tiles','skipped_tiles'):
                if stats['overall'][key]!=stats['local'][key]+stats['global'][key]: violations.append(name+': aggregation mismatch')
            fresh=numeric_score(r['prediction'],m['expected']) if m else {}
            if fresh!=r['score']: violations.append(name+': score regeneration mismatch')
            samples.append(dict(condition=name,id=r['id'],mode=r['mode'],expected=m['expected'] if m else None,
                **fresh,termination_reason=r['termination_reason'],generated_tokens=len(r['completion_tokens']),stats=stats))
        for mode in MODES:
            rows=sorted([r for r in all_rows if r['mode']==mode],key=lambda r:r['id'])
            if not rows: continue
            stats=aggregate(x for r in rows for x in r['calls'])
            correct=sum(r['score']['correct'] for r in rows);pairs=[r for r in rows if r['id'] in dense]
            deltas=[int(r['score']['correct'])-int(dense[r['id']]['score']['correct']) for r in pairs]
            counts=[token_counts(dense[r['id']]['completion_tokens'],r['completion_tokens']) for r in pairs]
            matched=sum(x for x,y in counts);compared=sum(y for x,y in counts)
            cal={k:policy['modes'][mode][k]['targets'][str(c['target'])] for k in ('local','global')} if policy and c['method']=='blasst' else {}
            summary.append(dict(mode=mode,condition=name,method=c['method'],target_sparsity=c['target'],beta=c['beta'],
                lambda_local=rows[0]['lambdas']['local'] if rows[0]['lambdas'] else None,
                lambda_global=rows[0]['lambdas']['global'] if rows[0]['lambdas'] else None,
                n=len(rows),correct=correct,accuracy=correct/len(rows),
                actual_sparsity=stats['overall']['full_tile_sparsity'],global_sparsity=stats['global']['full_tile_sparsity'],local_sparsity=stats['local']['full_tile_sparsity'],
                paired_dense_accuracy=sum(dense[r['id']]['score']['correct'] for r in pairs)/len(pairs) if pairs else None,
                paired_accuracy_delta=float(np.mean(deltas)) if deltas else None,paired_delta_ci=list(paired_bootstrap_ci(deltas)) if deltas else None,
                matched_tokens=matched,compared_tokens=compared,token_agreement=matched/compared if compared else None,
                sequence_match=sum(r['completion_tokens']==dense[r['id']]['completion_tokens'] for r in pairs)/len(pairs) if pairs else None,
                retained_mass=stats['overall']['retained_dense_attention_mass'],length_terminated=sum(r['termination_reason']=='length' for r in rows),
                extraction_failures=sum(r['score']['extracted'] is None for r in rows),stats=stats,
                calibration_unattainable={k:p['unattainable'] for k,p in cal.items()},calibration_achieved={k:p['sparsity'] for k,p in cal.items()}))
            for layer in sorted({x['layer'] for r in rows for x in r['per_layer']}):
                calls=[x for r in rows for x in r['per_layer'] if x['layer']==layer]
                layers.append(dict(mode=mode,condition=name,layer=layer,attention_type=calls[0]['attention_type'],**aggregate(calls)['overall']))
    complete=all(len(v)==60 for v in loaded.values()) and not violations
    audit=dict(complete=complete,expected=540,completed={k:len(v) for k,v in loaded.items()},violations=violations)
    _write(root/'audit.json',audit);_write(root/'summary.json',dict(audit=audit,results=summary));_write(root/'per_sample_metrics.json',samples)
    write_csv(root/'summary.csv',summary);write_csv(root/'per_sample.csv',samples);write_csv(root/'per_layer.csv',layers)
    texts=['# Full AIME2024: zero-shot and five-shot sparse attention','',f'Status: {"complete" if complete else "partial"}. {sum(len(v) for v in loaded.values())}/540 final generations.','',
        'Thirty AIME2024 problems are tested in each prompt mode, using the same model revision, BF16 precision, seed 42, tokenized prompts and 2,048-token budget across all nine conditions. Five fixed, seed-selected AIME2025 worked solutions are included verbatim in every five-shot prompt. Six other AIME2025 questions calibrate BLASST. Test, demonstration and calibration problems are disjoint; their hashes and source IDs are frozen in the manifests. This is full AIME2024 (30 questions), not a combination of different AIME years.', '',
        'The inherited decoder uses its native seeded temperature schedule (0.4–0.8), maximum 48 denoising steps per canvas, confidence threshold 0.005, stability threshold 1, entropy bound 0.1 and canvas length 256. Thinking is disabled at prompt formatting; prompts explicitly request worked reasoning. The adapter temperature=0 argument preserves this native schedule; it is not greedy sampling.', '',
        'Both algorithms use the existing 64×64 physical-tile implementations. Prefix and canvas KV tiles are eligible; Sol standardizes mean-pooled Q/K proxies across their combined population. Sol beta stays fixed at -0.674490, 0, +0.674490, +1.281552. BLASST row max/running max votes drop a physical tile only when all valid query rows vote to skip.', '',
        'BLASST calibration is fresh and separate for each prompt mode and local/global attention. One dense calibration trajectory per held-out problem records the maximum valid-row online-softmax margin per physical tile. This is sufficient to evaluate scalar lambda exactly offline. The previous candidate grid plus lambda=1 is scanned; 32 log-space bisection steps select the closest measured target. Ties are retained. If lambda=1 cannot reach the target, lambda=1 is deployed and the target is labelled unattainable on calibration. The prior exponential fit is exported as a diagnostic, but no length-transfer assumption is needed: each prompt mode gets its own two scalar thresholds. Selected policies are verified on sparse calibration generations; residual trajectory-dependent errors are reported, without using final-test labels or retuning on final results.', '',
        'Accuracy uses gold-independent final-number extraction: the last boxed answer takes precedence, followed by the last explicit answer marker, then the last numeric token. Numeric values are compared with absolute tolerance 1e-6, accepting leading zeros, integral decimals and scientific notation. It does not accept an answer merely because the gold number occurs somewhere in the reasoning. Extraction failures and token-budget terminations are reported.', '',
        'Sparsity is sum(skipped eligible physical tiles)/sum(eligible physical tiles), separately for all decoder layers and local/global layers. Dense encoder and vision attention are outside this denominator. Retained mass is measured against dense softmax on the current generation trajectory and is weighted by valid query rows. Token agreement compares the union of generated positions, treating missing/extra tokens as disagreements and continuing after divergence. No speedup claims are made.', '']
    def pc(x): return '—' if x is None else f'{100*x:.2f}%'
    for mode in MODES:
        texts += [f'## {mode}','', '| Method | Target | β | λ local / global | Actual | Global | Local | Correct | Accuracy | Δ dense |','|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
        for r in [r for r in summary if r['mode']==mode]:
            lam='—' if r['lambda_local'] is None else f'{r["lambda_local"]:.7g} / {r["lambda_global"]:.7g}'
            texts.append(f'| {r["method"]} | {pc(r["target_sparsity"])} | {r["beta"] if r["beta"] is not None else "—"} | {lam} | {pc(r["actual_sparsity"])} | {pc(r["global_sparsity"])} | {pc(r["local_sparsity"])} | {r["correct"]}/{r["n"]} | {pc(r["accuracy"])} | {pc(r["paired_accuracy_delta"])} |')
        texts.append('')
    texts+=['## Calibration and limitations','']
    if policy:
        for mode in MODES:
            for kind in ('local','global'):
                p=policy['modes'][mode][kind]
                unreachable=[pc(float(t)) for t,v in p['targets'].items() if v['unattainable']]
                texts.append(f'- {mode}/{kind}: dense calibration lambda=1 ceiling {pc(p["lambda1_ceiling"])}; unattainable targets: {", ".join(unreachable) or "none"}.')
    for r in summary:
        if r['length_terminated'] or r['extraction_failures']:
            texts.append(f'- {r["mode"]}/{r["condition"]}: {r["length_terminated"]}/{r["n"]} budget terminations, {r["extraction_failures"]}/{r["n"]} answer-extraction failures.')
    texts+=['','Thirty questions give 3.33 percentage points per correct answer. The five-shot comparison changes prompt length and includes worked demonstrations together, so any difference cannot be attributed solely to either factor. Calibration is estimated from six held-out problems per mode and sparse trajectories can change achieved sparsity. Paired bootstrap intervals, counts and verification residuals are in the JSON artifacts.','',f'Audit: {json.dumps(audit,sort_keys=True)}','']
    (root/'report.md').write_text('\n'.join(texts))
    if summary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,mode in zip(axes,MODES):
            for method in ('sol','blasst'):
                points=sorted([r for r in summary if r['mode']==mode and r['method']==method],key=lambda r:r['target_sparsity'])
                ax.plot([100*r['actual_sparsity'] for r in points],[100*r['accuracy'] for r in points],'o-',label=method)
            d=next((r for r in summary if r['mode']==mode and r['method']=='dense'),None)
            if d: ax.axhline(100*d['accuracy'],color='grey',ls='--',label='dense')
            ax.set(title=mode,xlabel='Measured physical tile sparsity (%)',xlim=(0,100),ylim=(0,105));ax.grid(alpha=.25);ax.legend()
        axes[0].set_ylabel('Numeric accuracy (%)');fig.tight_layout()
        fig.savefig(root/'accuracy_vs_sparsity.png',dpi=180);fig.savefig(root/'accuracy_vs_sparsity.pdf');plt.close(fig)
    return audit
