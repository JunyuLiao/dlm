"""Predeclared calibration-only audit of continuous causal confidence history.

The fixed candidates below are motivated by binary-prior saturation. No final
accuracy is read, and no coefficient is selected using the other 24 questions.
"""
import csv
import json
from pathlib import Path
import numpy as np
from .query_signal_audit import FEATURES,load_condition,_ema,row_metrics,mean_valid,write_csv
ROOT=Path('/home/exouser/aime_query_signal_audit_v1')

def run():
    d=load_condition(ROOT,'dense',1); idx={f:i for i,f in enumerate(FEATURES)}
    ids={2,8,14,20,23,30}; take=np.isin(d['meta'][:,0],list(ids))
    x=d['x'][take]; y=d['y'][take]; m=d['meta'][take]; acc=d['accepted'][take]
    uc=x[:,:,idx['u_conf']]; q=x[:,:,idx['q_g065']]; z=x[:,:,idx['z_g05']]
    scores={name:x[:,:,idx[name]] for name in ('C_gate','C_prior','T_prior','u_conf')}
    for rho in (.25,.5,.75): scores[f'C_soft_rho{rho}']=rho*q+(1-rho*q)*uc
    for gamma in (.5,.65,.8):
        e=np.zeros_like(uc)
        for qid,seed,canvas in np.unique(m[:,:3],axis=0):
            ix=np.flatnonzero(np.all(m[:,:3]==[qid,seed,canvas],axis=1))
            assert np.array_equal(m[ix,3],np.arange(2,len(ix)+2))
            e[ix]=_ema(uc[ix],gamma,1.)
        scores[f'C_ema_g{gamma}']=e
        if gamma==.65:scores['T_confprior_g0.65']=e+(1-e)*z
    rows=[]; questions=[]
    for name,s in scores.items():
        for state,mask in [('all',None),('accepted',acc),('rejected',~acc)]:
            per=row_metrics(y,s,mask)
            for qid in sorted(ids):
                qmean=mean_valid(per[m[:,0]==qid])
                for j,label in enumerate(('renoise','flip','either')):
                    questions.append(dict(method=name,state=state,question=qid,label=label,
                        auc=float(qmean[j,0]),p25=float(qmean[j,2]),brier=float(qmean[j,5])))
            ave=mean_valid(per)
            for j,label in enumerate(('renoise','flip','either')):
                rows.append(dict(method=name,state=state,label=label,auc=float(ave[j,0]),
                    p25=float(ave[j,2]),brier=float(ave[j,5]),
                    call2_mean=float(s[m[:,3]==2].mean()),
                    saturated=float((s>=.995).mean())))
    out=ROOT/'analysis'
    write_csv(out/'soft_calibration_only.csv',rows)
    write_csv(out/'soft_calibration_questions.csv',questions)
    (out/'soft_protocol.json').write_text(json.dumps(dict(calibration_ids=sorted(ids),conditions=['dense'],
        seeds=[42,43,44],methods=list(scores),selection='No automatic winner: ranking cannot select temporal protection or predict convergence.',
        hypothesis='Replace binary rejection saturation with continuous confidence information; all start at h=1 before any observations.'),indent=2)+'\n')
    print(json.dumps([r for r in rows if r['label']=='either' and r['state']=='all'],indent=2))
if __name__=='__main__':run()
