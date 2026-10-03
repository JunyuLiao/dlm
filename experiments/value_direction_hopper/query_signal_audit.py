"""Causal, canvas-local signal audit. Future instability is an attention-value proxy.

Features after completed call k predict k+1 or the union at k+1,k+2.
No current logit may influence its own call's routing. First routing call's
maximum coefficient has no history and is outside these ranking metrics.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

CALIBRATION_IDS = {2, 8, 14, 20, 23, 30}
FEATURES = ('reject_now', 'flip_now', 'q_g05', 'q_g065', 'q_g08', 'q_g09',
            'u_margin', 'u_conf', 'u_raw_conf', 'entropy_u', 'z_g05',
            'drift_g05', 'winner_loss', 'run_u', 'M_prior', 'C_prior',
            'T_prior', 'T_smooth', 'T_hybrid', 'T_run', 'M_gate', 'C_gate',
            'T_gate', 'C_run', 'H_prior', 'H_anchor', 'C_soft', 'C_tail',
            'C_gate_soft',
            'T_blend')
LABELS = ('renoise', 'flip', 'either')
METRICS = ('auroc', 'ap', 'precision25', 'precision50', 'prevalence', 'brier')


def _ema(values, gamma, initial=0.):
    out = np.empty_like(values, dtype=np.float32)
    state = np.full(values.shape[1:], initial, dtype=np.float32)
    for i, v in enumerate(values):
        state = gamma * state + (1-gamma)*v
        out[i] = state
    return out


def _derive(arr, m_ref=14.258454322814941):
    """One canvas only; reset all histories before its first observation."""
    if 'canvas' in arr and len(np.unique(arr['canvas'])) != 1:
        raise ValueError('derive accepts exactly one canvas')
    accepted = arr['accepted'].astype(bool)
    conf = arr['confidence'].astype(np.float32)
    flip = np.zeros_like(accepted)
    flip[1:] = arr['top'][1:] != arr['top'][:-1]
    q = {g: _ema(~accepted, g, 1.) for g in (.5, .65, .8, .9)}
    z = _ema(flip, .5)
    change = np.zeros_like(conf)
    change[1:] = np.abs(conf[1:] - conf[:-1])
    drift = _ema(change, .5)
    run = np.zeros_like(conf)
    state = np.zeros(conf.shape[1:], dtype=np.float32)
    for i in range(len(conf)):
        state = (state+1)*accepted[i]
        state[flip[i]] = 0
        run[i] = state
    uc = np.sqrt(np.clip(1-conf, 0, 1))
    um = m_ref/(np.maximum(arr['margin'], 0)+m_ref)
    uh = arr['entropy']/(1+arr['entropy'])
    winner_loss = np.zeros_like(conf)
    winner_loss[1:] = np.clip(conf[:-1]-arr['previous_winner_confidence'][1:], 0, 1)
    out = dict(reject_now=(~accepted).astype(np.float32), flip_now=flip.astype(np.float32),
               q_g05=q[.5], q_g065=q[.65], q_g08=q[.8], q_g09=q[.9], u_margin=um,
               u_conf=uc, u_raw_conf=np.sqrt(np.clip(1-arr['raw_confidence'], 0, 1)),
               entropy_u=uh, z_g05=z, drift_g05=drift, run_u=np.exp(-run/2),
               winner_loss=winner_loss)
    for name, u in [('M_prior', um), ('C_prior', uc), ('T_prior', z),
                    ('H_prior', uh), ('T_blend', (z+drift)/2)]:
        out[name] = q[.5]+(1-q[.5])*u
    out['T_smooth'] = q[.8]+(1-q[.8])*z
    out['T_hybrid'] = 1-(1-q[.5])*(1-z)*(1-drift)
    out['T_run'] = out['run_u']+(1-out['run_u'])*z
    out['H_anchor'] = .1*q[.65]+.9*uh
    out['C_soft'] = .25*q[.65]+(1-.25*q[.65])*uc
    rank = (rankdata(uc, axis=-1, method='average') - 1) / max(uc.shape[-1]-1, 1)
    tail = np.clip((rank-.75)/.25, 0, 1)
    out['C_tail'] = np.clip(uc + .5*q[.65]*tail*(1-uc), 0, 1)
    gate = -np.expm1(-run/2.5)
    a = gate*(1-q[.65])
    a = .2 + .8*a
    out['C_gate_soft'] = 1-a*(1-uc)
    for name, u, tau in [('M_gate', um, 2.5), ('C_gate', uc, 2.5), ('T_gate', z, 3.5)]:
        out[name] = 1-(-np.expm1(-run/tau))*(1-q[.65])*(1-u)
    out['C_run'] = 1-(-np.expm1(-run/2.5))*(1-uc)
    return out


def load_condition(root, condition, horizon=1, m_ref=14.258454322814941):
    """Dense columnar arrays avoid millions of Python token dictionaries."""
    xs, ys, accepted, metadata = [], [], [], []
    for path in sorted((root/'traces'/condition).glob('seed*/*.json')):
        meta = json.loads(path.read_text())
        seed = meta['seed']; qid = int(meta['id'].split('/')[-1])
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as saved:
            arr = dict(saved)
        for canvas in np.unique(arr['canvas']):
            take = arr['canvas'] == canvas
            a = {k: v[take] for k,v in arr.items()}
            n = len(a['call'])-horizon
            if n <= 0:
                continue
            assert np.array_equal(a['call'], np.arange(1, len(a['call'])+1))
            f = _derive(a, m_ref)
            y = np.zeros((n, a['accepted'].shape[1], 3), dtype=bool)
            for h in range(1, horizon+1):
                y[:,:,0] |= ~a['accepted'][h:h+n]
                y[:,:,1] |= a['top'][h:h+n] != a['top'][h-1:h+n-1]
            y[:,:,2] = y[:,:,0] | y[:,:,1]
            xs.append(np.stack([f[k][:n] for k in FEATURES], axis=-1))
            ys.append(y); accepted.append(a['accepted'][:n])
            metadata.extend((qid, seed, int(canvas), int(call)+1) for call in a['call'][:n])
    if not xs:
        raise ValueError(f'no completed traces for {condition}')
    return dict(x=np.concatenate(xs).astype(np.float32), y=np.concatenate(ys),
                accepted=np.concatenate(accepted), meta=np.asarray(metadata))


def row_metrics(y, score, mask=None):
    """Vectorized metrics per invocation; fractional inclusion of boundary ties.

    y: N,Q,L boolean; score: N,Q. Mask conditions on previous outcomes.
    Missing AUC (one label class) and AP (no positives) remain NaN.
    """
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    valid = np.ones(score.shape, bool) if mask is None else np.asarray(mask, bool)
    n = valid.sum(1)
    pos = (y & valid[:,:,None]).sum(1)
    neg = n[:,None]-pos
    ranks = rankdata(np.where(valid, score, np.nan), axis=1, nan_policy='omit')
    auc_num = np.nansum(ranks[:,:,None]*y, axis=1)-pos*(pos+1)/2
    with np.errstate(invalid='ignore', divide='ignore'):
        auc = auc_num/(pos*neg)
    order = np.argsort(-np.where(valid, score, -np.inf), axis=1, kind='stable')
    scores = np.take_along_axis(score, order, axis=1)
    keep = np.take_along_axis(valid, order, axis=1)
    yy = np.take_along_axis(y, order[:,:,None], axis=1) & keep[:,:,None]
    cumulative = yy.cumsum(1)
    width = score.shape[1]
    idx = np.broadcast_to(np.arange(width), score.shape)
    ends = np.ones(score.shape, bool)
    ends[:,:-1] = (scores[:,:-1] != scores[:,1:]) | (keep[:,:-1] != keep[:,1:])
    end_idx = np.minimum.accumulate(np.where(ends, idx, width)[:,::-1], axis=1)[:,::-1]
    cum_at_end = np.take_along_axis(cumulative, end_idx[:,:,None], axis=1)
    precision = cum_at_end/(end_idx[:,:,None]+1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ap = (yy*precision).sum(1)/pos
    # Descending tie ranks, with omitted tokens ignored.
    lo = rankdata(np.where(valid, -score, np.nan), axis=1, method='min', nan_policy='omit')
    hi = rankdata(np.where(valid, -score, np.nan), axis=1, method='max', nan_policy='omit')
    precisions = []
    for fraction in (.25, .5):
        budget = fraction*n
        fraction_keep = np.nan_to_num(np.clip((budget[:,None]-lo+1)/(hi-lo+1), 0, 1))
        with np.errstate(invalid='ignore', divide='ignore'):
            precisions.append((fraction_keep[:,:,None]*y).sum(1)/budget[:,None])
    with np.errstate(invalid='ignore', divide='ignore'):
        prevalence = pos/n[:,None]
        brier = (((score[:,:,None]-y)**2)*valid[:,:,None]).sum(1)/n[:,None]
    return np.stack([auc, ap, *precisions, prevalence, brier], axis=-1)


def mean_valid(a, axis=0):
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.nansum(a, axis=axis)/np.isfinite(a).sum(axis=axis)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def evaluate(data, condition, horizon, calibration_ids):
    metrics, per_question, diagnostics = [], [], []
    meta, x, y, acc = (data[k] for k in ('meta','x','y','accepted'))
    for split in ('calibration', 'heldout'):
        selected = np.isin(meta[:,0], list(calibration_ids))
        if split=='heldout': selected = ~selected
        if not selected.any(): continue
        xx, yy, mm, aa = x[selected], y[selected], meta[selected], acc[selected]
        for i, feature in enumerate(FEATURES):
            score = xx[:,:,i]
            for state in ('all', 'accepted', 'rejected'):
                mask = np.ones(aa.shape,bool) if state=='all' else aa if state=='accepted' else ~aa
                per_inv = row_metrics(yy, score, mask)
                pooled = row_metrics(yy.reshape(1,-1,3), score.reshape(1,-1), mask.reshape(1,-1))[0]
                qm = []
                for qid in np.unique(mm[:,0]):
                    item = mean_valid(per_inv[mm[:,0]==qid])
                    qm.append(item)
                    for li,label in enumerate(LABELS):
                        per_question.append(dict(condition=condition, horizon=horizon, split=split,
                            state=state, feature=feature, question=int(qid), label=label,
                            **dict(zip(METRICS,item[li]))))
                scopes = [('pooled', pooled), ('invocation_macro', mean_valid(per_inv)),
                          ('question_macro', mean_valid(np.asarray(qm)))]
                for seed in np.unique(mm[:,1]):
                    scopes.append((f'seed{seed}_invocation_macro', mean_valid(per_inv[mm[:,1]==seed])))
                # Phase refers to the FUTURE routing call receiving this score.
                for name, take in [('call2',mm[:,3]==2), ('calls3_8',(mm[:,3]>=3)&(mm[:,3]<=8)),
                                   ('calls9plus', mm[:,3]>=9)]:
                    if take.any(): scopes.append((name+'_invocation_macro',mean_valid(per_inv[take])))
                # 128-query row-group ranking is still only a proxy for physical
                # tile routing: K/V risk and per-head reductions are absent.
                if state=='all':
                    tile = row_metrics(yy.reshape(-1,128,3),score.reshape(-1,128))
                    scopes.append(('querygroup128_macro',mean_valid(tile)))
                for scope, values in scopes:
                    for li,label in enumerate(LABELS):
                        metric=dict(zip(METRICS,values[li]))
                        metric['lift25']=metric['precision25']/metric['prevalence'] if metric['prevalence']>0 else np.nan
                        metrics.append(dict(condition=condition,horizon=horizon,split=split,state=state,
                            feature=feature,label=label,scope=scope,invocations=len(mm),
                            valid_auc_invocations=int(np.isfinite(per_inv[:,li,0]).sum()),**metric))
            for name, take in [('call2',mm[:,3]==2), ('calls3_8',(mm[:,3]>=3)&(mm[:,3]<=8)),
                               ('calls9plus', mm[:,3]>=9)]:
                if not take.any(): continue
                vals=score[take]
                diagnostics.append(dict(condition=condition,horizon=horizon,split=split,phase=name,
                    feature=feature,mean=float(vals.mean()),median=float(np.median(vals)),
                    p90=float(np.quantile(vals,.9)),saturated=float(np.mean(vals>=.995))))
        print(json.dumps(dict(stage='metrics',condition=condition,horizon=horizon,split=split)),flush=True)
    return metrics, per_question, diagnostics


def analyze(root, out, conditions):
    protocol=json.loads((root/'protocol.json').read_text())
    calibration_ids={int(x.split('/')[-1]) for x in protocol['calibration_ids']}
    m_ref=protocol['model_config']['m_ref']
    out.mkdir(parents=True,exist_ok=True)
    all_metrics=[]; all_questions=[]; all_diagnostics=[]; integrity=[]
    for condition in conditions:
        files=list((root/'traces'/condition).glob('seed*/*.json'))
        assert len(files)==90, (condition,len(files))
        for p in files:
            m=json.loads(p.read_text())
            integrity.append({k:m[k] for k in ('condition','seed','id','same_prediction','same_steps')})
        for horizon in (1,2):
            data=load_condition(root,condition,horizon,m_ref)
            metrics,questions,diag=evaluate(data,condition,horizon,calibration_ids)
            all_metrics.extend(metrics); all_questions.extend(questions); all_diagnostics.extend(diag)
            write_csv(out/f'{condition}_h{horizon}_metrics.csv',metrics)
            write_csv(out/f'{condition}_h{horizon}_questions.csv',questions)
            write_csv(out/f'{condition}_h{horizon}_diagnostics.csv',diag)
    write_csv(out/'metrics.csv',all_metrics)
    write_csv(out/'question_metrics.csv',all_questions)
    write_csv(out/'feature_diagnostics.csv',all_diagnostics)
    write_csv(out/'integrity.csv',integrity)
    summary=dict(schema='causal_query_signal_audit_v2',conditions=conditions,
        calibration_ids=sorted(calibration_ids),predictive_holdout_ids=sorted(set(range(1,31))-calibration_ids),
        traces=len(integrity),prediction_mismatches=sum(not m['same_prediction'] for m in integrity),
        step_mismatches=sum(not m['same_steps'] for m in integrity),
        labels='Renoised/argmax flips in complete future one- or two-call windows, within canvas.',
        first_call='All production causal methods use h=1 on call1; no-history ranking is undefined.',
        macro='Question macro is mean within-invocation metric per question, then across questions.',
        brier='Descriptive squared error of heuristic h; h is not a fitted probability.',
        limitation='Instability prediction does not prove marginal benefit of attention. All 30 AIME prompts were previously inspected; predictive holdout is not new task-level validation.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/home/exouser/aime_query_signal_audit_v1'))
    p.add_argument('--out',type=Path)
    p.add_argument('--conditions',nargs='+',default=['dense','C_gate','T_prior'])
    a=p.parse_args(); analyze(a.root,a.out or a.root/'analysis',a.conditions)


if __name__=='__main__': main()
