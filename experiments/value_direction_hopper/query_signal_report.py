"""Report existing signal metrics with question-clustered paired intervals."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/exouser/aime_query_signal_audit_v1')

def main(root=ROOT):
    out=root/'analysis'
    metrics=list(csv.DictReader((out/'metrics.csv').open()))
    questions=list(csv.DictReader((out/'question_metrics.csv').open()))
    diagnostics=list(csv.DictReader((out/'feature_diagnostics.csv').open()))
    def select(rows,**kw):
        return [r for r in rows if all(str(r[k])==str(v) for k,v in kw.items())]
    common=dict(split='heldout',horizon=1,label='either')
    def value(condition,feature,state='all',metric='auroc'):
        row=select(metrics,**common,condition=condition,feature=feature,state=state,scope='invocation_macro')
        assert len(row)==1
        return float(row[0][metric])
    intervals=[]
    for condition in ('dense','C_gate','T_prior'):
        for left,right in [('u_conf','C_gate'),('entropy_u','C_gate'),('u_margin','T_prior'),('T_hybrid','T_prior')]:
            for state in ('all','accepted','rejected'):
                pair=[]
                for feature in (left,right):
                    part=select(questions,**common,condition=condition,feature=feature,state=state)
                    pair.append({int(r['question']):float(r['auroc']) for r in part})
                ids=sorted(pair[0].keys() & pair[1].keys()); delta=np.array([pair[0][i]-pair[1][i] for i in ids])
                assert len(delta)==24 and np.isfinite(delta).all()
                rng=np.random.default_rng(20261001)
                boot=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
                lo,hi=np.quantile(boot,[.025,.975])
                intervals.append(dict(condition=condition,state=state,left=left,right=right,
                    mean_delta=float(delta.mean()),lo=float(lo),hi=float(hi),questions=24))
    with (out/'paired_question_bootstrap.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(intervals[0])); w.writeheader();w.writerows(intervals)
    lines=['# AIME26 causal query-signal audit','',
      'Completed 270 traces: 30 questions × 3 seeds × dense/C_gate/T_prior. All predictions and native call counts exactly match the archives. Focused tests: 43 passed before the new candidate, then 44 passed with H_anchor.',
      '', '## Protocol', '',
      '- Features after completed call k predict call k+1 or the union of events in k+1,k+2, never across canvases. Complete horizons only; final stopped/capped windows are censored.',
      '- The six existing calibration IDs are 2, 8, 14, 20, 23, 30. Tables below use the remaining 24 IDs. These are a predictive holdout for established formulas; all questions were already exposed in task-quality experiments.',
      '- Each first routing call has h=1, s=4. It has no history to rank, so the earliest scored routing call is call2. Phase labels refer to that future routing call.',
      '- AUROC is averaged within calls with both classes present. Conditional metrics restrict to queries accepted/rejected after call k. Question macro averages within calls then over questions; bootstrap resamples questions with all seeds grouped.',
      '- P@25/50 uses fractional inclusion of tied scores. It is a query-ranking budget, not physical tile sparsity. Brier is descriptive squared error of heuristic hazards, not a claim that these are calibrated probabilities.',
      '- Scores are replayed on fixed dense and sparse trajectories. They do not intervene in attention. All canvas positions are included, including possible unused final-canvas tail positions. Timing from these instrumented runs is not a clean latency benchmark.',
      '', '## Next-call renoise-or-flip: within-call AUROC','',
      '| Signal/formula | Dense | C_gate trajectory | T_prior trajectory |','|---|---:|---:|---:|']
    for feature in ('u_margin','u_conf','u_raw_conf','entropy_u','M_prior','C_prior','T_prior','T_smooth','T_hybrid','T_run','M_gate','C_gate','T_gate','C_run'):
        lines.append('| '+feature+' | '+' | '.join(f'{value(c,feature):.4f}' for c in ('dense','C_gate','T_prior'))+' |')
    lines+=['','u_conf uses processed top-1 probability; u_raw_conf undoes the known scalar temperature. entropy_u = H/(1+H), with processed logits. Raw/processed confidence are close but not identical. The data do not prove complete temperature invariance.',
      '', '## Conditional ranking reveals the saturation problem','',
      '| Trace | Currently accepted: C_gate / confidence / entropy | Currently rejected: C_gate / confidence / entropy |','|---|---:|---:|']
    for c in ('dense','C_gate','T_prior'):
        cols=[' / '.join(f'{value(c,f,state):.4f}' for f in ('C_gate','u_conf','entropy_u')) for state in ('accepted','rejected')]
        lines.append(f'| {c} | '+ ' | '.join(cols)+' |')
    lines+=['', 'These are mathematically expected failures of the combination, even though the individual signals carry information:',
      '', '- Prior: h=q+(1-q)u, so dh/du=1-q. A never-accepted query keeps q=1; the coefficient cannot distinguish its confidence or margin.',
      '- Gate: h=1-a(r)(1-q)(1-u). Any last-call rejection or top-1 flip sets r=0 and a=0, forcing h=1 regardless of u. Hence AUROC is exactly 0.5 within rejected queries.',
      '- q, stable runs, and uncertainty share information about sampler acceptance; interpreting their noisy-OR as independent probabilities is unjustified. The entropy sampler uses a joint cumulative budget, not an independent per-token acceptance threshold.',
      '- Top-1 probability drift changes when certainty improves as well as deteriorates, and it can miss identity swaps with similar probabilities. Its monotone positive hazard interpretation is questionable; it is not an attention-error measure.',
      '', '## Saturation on the C_gate trajectory (horizon 1, predictive holdout)','',
      '| Routing phase | C_gate fraction h >= 0.995 | C_gate mean h | Confidence mean u |','|---|---:|---:|---:|']
    for phase in ('call2','calls3_8','calls9plus'):
        kw=dict(condition='C_gate',split='heldout',horizon=1,phase=phase)
        g=select(diagnostics,**kw,feature='C_gate')[0]; u=select(diagnostics,**kw,feature='u_conf')[0]
        lines.append(f"| {phase} | {100*float(g['saturated']):.2f}% | {float(g['mean']):.4f} | {float(u['mean']):.4f} |")
    lines+=['','Late phases condition on canvases that survive to those calls. The averages cannot show that slow sensitivity decay causes excess denoising calls; cheaper attention can also slow convergence.',
      '', '## Robustness and meaning','',
      'The confidence/entropy ranking advantage persists across dense, C_gate and T_prior trajectories, one- and two-call horizons, all three seeds, and 128-query row groups. Margin ranks future flips better than entropy. Per-seed rows and question bootstrap intervals are included in the CSV artifacts.',
      '', '| Trace | Confidence minus C_gate question-macro AUROC | 95% paired question-bootstrap interval |','|---|---:|---:|']
    for c in ('dense','C_gate','T_prior'):
        r=select(intervals,condition=c,state='all',left='u_conf',right='C_gate')[0]
        lines.append(f"| {c} | {r['mean_delta']:.4f} | [{r['lo']:.4f}, {r['hi']:.4f}] |")
    lines+=['','This establishes a reproducible loss of predictive ranking, not that the highest uncertainty queries receive the largest benefit from extra attention. Sparse trajectories have different survivor populations and may change their own input signals. Seed stability of AUROC does not imply reduced seed variance of final accuracy.',
      '', '## Exploratory pilot and next decision','',
      'A separate H_anchor candidate was implemented: h=0.1q+0.9H/(1+H), q EMA gamma=0.65; s=1+3h and first call s=4. One local/global threshold pair is required throughout. It restores ranking among queries at q=1 because its entropy derivative remains positive.',
      '', 'The candidate and 0.90 mixture were explored after inspecting the 24-question predictive tables. anchor_metrics.csv therefore contains development diagnostics, not untouched holdout confirmation or a calibration-only fitted model. No learned hazard weights were fitted.',
      '', 'H_anchor is a diagnostic pilot, not a demonstrated improvement: it changes ranking, scaling and temporal protection together, and its accepted-query reactivation ranking remains below entropy alone. Production quality/call-count results must decide whether to extend it.',
      '', 'A more conservative follow-up, if this pilot fails, is an initialized EMA of the confidence signal: e0=1; e_next=gamma*e_previous+(1-gamma)*sqrt(1-p). It keeps early protection from causal initialization and retains a nonzero dependence on confidence even for rejected queries. This is proposed, not yet evaluated. Retain C_gate, M_prior and T_prior as controls.',
      '', '## Files','',
      '- metrics.csv: full established-formula tables, two horizons, conditional groups, phase, seed, pooled, invocation and question macro metrics.',
      '- question_metrics.csv and paired_question_bootstrap.csv: grouped comparison evidence.',
      '- feature_diagnostics.csv: saturation and coefficient diagnostics.',
      '- anchor_metrics.csv: exploratory H_anchor diagnostics.',
      '- integrity.csv / summary.json: trace integrity and protocol limitations.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(out/'report.md')

if __name__=='__main__': main()
