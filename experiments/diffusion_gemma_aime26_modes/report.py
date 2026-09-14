"""CPU-only report regeneration, with explicit partial-set/provenance audits."""
import json
from pathlib import Path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _fingerprint
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import aggregate, token_counts, csv_write
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci
from .protocol import MODES, CONDITIONS, numeric_score, final_response
from .run import selected


def report(root):
    manifest={r['id']:r for r in json.loads((root/'final.json').read_text())}
    fp=_fingerprint(json.loads((root/'experiment.json').read_text()))
    policy=json.loads((root/'policy.json').read_text()) if (root/'policy.json').exists() else None
    dense={}; summary=[]; violations=[]; counts={}; samples=[]; layers=[]
    for name,target in CONDITIONS:
        rows=[json.loads(p.read_text()) for p in (root/'conditions'/name/'shards').glob('*.json')]
        counts[name]=len(rows)
        if len({r['id'] for r in rows})!=len(rows): violations.append(name+': duplicate IDs')
        for r in rows:
            m=manifest.get(r['id'])
            if not m or any(r[k]!=m[k] for k in ('prompt_hash','prompt_token_hash','seed','generation_budget','thinking','mode')):
                violations.append(name+': manifest mismatch');continue
            if r['fingerprint']!=fp: violations.append(name+': fingerprint mismatch')
            expected_lam=selected(policy,r['mode'],target) if target is not None and policy else None
            if r['lambdas']!=expected_lam: violations.append(name+': lambda mismatch')
            if expected_lam and any(c['effective_blasst_lambda']!=expected_lam[c['attention_type']] for c in r['calls']):
                violations.append(name+': effective threshold mismatch')
            answer=final_response(r['raw_completion'],r['thinking'])
            score=numeric_score(answer,m['expected'])
            if score!=r['score'] or answer!=r['final_response']: violations.append(name+': scoring mismatch')
            stat=aggregate(r['calls'])
            for key in ('skipped_tiles','eligible_tiles'):
                if stat['overall'][key]!=stat['local'][key]+stat['global'][key]: violations.append(name+': count mismatch')
            if name=='dense': dense[r['id']]=r
            samples.append(dict(condition=name,id=r['id'],mode=r['mode'],**score,termination=r['termination_reason'],
                tokens=len(r['completion_tokens']),stats=stat))
        for mode in MODES:
            rr=[r for r in rows if r['mode']==mode]
            if not rr: continue
            stat=aggregate(c for r in rr for c in r['calls'])
            pairs=[r for r in rr if r['id'] in dense]
            tc=[token_counts(dense[r['id']]['completion_tokens'],r['completion_tokens']) for r in pairs]
            matched=sum(a for a,b in tc);compared=sum(b for a,b in tc)
            deltas=[int(r['score']['correct'])-int(dense[r['id']]['score']['correct']) for r in pairs]
            correct=sum(r['score']['correct'] for r in rr)
            row=dict(mode=mode,condition=name,target_sparsity=target,n=len(rr),correct=correct,accuracy=correct/len(rr),
                actual_sparsity=stat['overall']['full_tile_sparsity'],global_sparsity=stat['global']['full_tile_sparsity'],
                local_sparsity=stat['local']['full_tile_sparsity'],lambda_local=rr[0]['lambdas']['local'] if target else None,
                lambda_global=rr[0]['lambdas']['global'] if target else None,
                unattainable={k:policy['modes'][mode][k]['targets'][str(target)]['unattainable'] for k in ('local','global')} if target else {},
                retained_mass=stat['overall']['retained_dense_attention_mass'],matched_tokens=matched,compared_tokens=compared,
                token_agreement=matched/compared if compared else None,paired_dense_n=len(pairs),
                paired_accuracy_delta=sum(deltas)/len(deltas) if deltas else None,
                paired_delta_ci=list(paired_bootstrap_ci(deltas)) if deltas else None,
                length_terminated=sum(r['termination_reason']=='length' for r in rr),
                empty_final=sum(not r['final_response'] for r in rr),extraction_failures=sum(r['score']['extracted'] is None for r in rr),stats=stat)
            summary.append(row)
            for layer in sorted({c['layer'] for r in rr for c in r['per_layer']}):
                calls=[c for r in rr for c in r['per_layer'] if c['layer']==layer]
                layers.append(dict(mode=mode,condition=name,layer=layer,attention_type=calls[0]['attention_type'],**aggregate(calls)['overall']))
    audit=dict(complete=all(v==120 for v in counts.values()) and not violations,expected=600,completed=counts,violations=violations)
    _write(root/'audit.json',audit);_write(root/'summary.json',dict(audit=audit,results=summary))
    _write(root/'per_sample_metrics.json',samples);csv_write(root/'summary.csv',summary);csv_write(root/'per_layer.csv',layers)
    csv_write(root/'per_sample.csv',samples)
    text=['# AIME2026: BLASST, thinking and few-shot','',f'Status: {"complete" if audit["complete"] else "partial"}; {sum(counts.values())}/600 saved generations.','',
        'Paper-aligned, not exact reproduction. All 30 AIME2026 problems; BF16 pinned DiffusionGemma; native 256-token canvas, max48 denoising steps, temperature schedule 0.8→0.4, confidence0.005, stability1, entropy0.1. Each mode/method receives one seed42 generation, capped at the user-approved 32,768 output tokens. The cap is a safety limit, not a published paper setting. Exact paper prompts and repeat count were not located. Five-shot is our extension using five fixed AIME2025 worked demonstrations. Native thinking is controlled through the official chat template, not inferred from CoT wording.', '',
        'Six disjoint AIME2025 questions calibrate each prompt/thinking mode. Existing 64×64 BLASST physical-tile masks allow prefix and canvas skipping. Local/global scalar lambdas use dense physical block-margin calibration and offline monotonic search over (0,1]; if lambda1 cannot reach a target, lambda1 is deployed and the target flagged unattainable. Six sparse calibration trajectories verify each target without tuning on final answers. No lambda>1 extension or Sol routing is used.', '',
        'Accuracy uses the final response channel only: unfinished thoughts cannot earn credit. Gold-independent boxed/answer-marker/final-number extraction compares numeric values with tolerance1e-6. All raw channels are preserved for audit. Physical sparsity is summed skipped/eligible tiles, including local/global splits. Retained mass is measured against dense softmax on the current sparse trajectory, not replay of the cached dense trajectory. Positional token agreement includes thought tokens, continues after divergence, and counts missing/extra tokens as mismatches. No speedup is claimed.', '']
    for mode in MODES:
        text += [f'## {mode}','','| Condition | Target | λ local/global | Actual | Global | Local | Accuracy | Budget stops |','|---|---:|---|---:|---:|---:|---:|---:|']
        for r in [r for r in summary if r['mode']==mode]:
            lam='—' if r['lambda_local'] is None else f'{r["lambda_local"]:.6g}/{r["lambda_global"]:.6g}'
            target='—' if r['target_sparsity'] is None else f'{100*r["target_sparsity"]:.0f}%'
            text.append(f'| {r["condition"]} | {target} | {lam} | {100*r["actual_sparsity"]:.2f}% | {100*r["global_sparsity"]:.2f}% | {100*r["local_sparsity"]:.2f}% | {r["correct"]}/{r["n"]} ({100*r["accuracy"]:.1f}%) | {r["length_terminated"]} |')
        text.append('')
    text += ['## Limitations','','Thirty questions means 3.33 percentage points per answer. Partial groups are not directly comparable to full groups. Token-limit terminations and empty final channels remain errors, not dropped samples. See CSV/JSON for paired bootstrap intervals, retained mass and token agreement. The paper reports AIME2026 TD no-thinking50.8% and thinking69.1%; our one-seed custom prompts do not reproduce its full undisclosed evaluation protocol.','',
        '[Technical report](https://arxiv.org/html/2608.00146v1)','',f'Audit: {json.dumps(audit)}']
    if policy:
        text += ['','## Calibration lambda=1 ceilings','']
        for mode in MODES:
            for k in ('local','global'):
                p=policy['modes'][mode][k]
                unavailable=[t for t,v in p['targets'].items() if v['unattainable']]
                text.append(f'- {mode}/{k}: {100*p["lambda1_ceiling"]:.2f}%; unattainable targets {unavailable}.')
    (root/'report.md').write_text('\n'.join(text)+'\n')
    if summary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(2,2,figsize=(10,7))
        for ax,mode in zip(axs.flat,MODES):
            rr=[r for r in summary if r['mode']==mode]
            sparse=[r for r in rr if r['target_sparsity'] is not None]
            ax.plot([100*r['actual_sparsity'] for r in sparse],[100*r['accuracy'] for r in sparse],'o-')
            for r in rr:
                if r['target_sparsity'] is None: ax.axhline(100*r['accuracy'],ls='--',color='grey',label='dense')
            ax.set(title=mode,xlabel='Measured tile sparsity (%)',ylabel='Accuracy (%)',xlim=(0,100),ylim=(0,100));ax.grid(alpha=.3)
        fig.tight_layout();fig.savefig(root/'accuracy_vs_sparsity.png',dpi=150);plt.close(fig)
    return audit
