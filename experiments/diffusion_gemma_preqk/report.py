"""Regenerate screen aggregates from completed, fingerprinted sample shards."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .config import ROOT
from .routing import FIELDS


def ratios(values):
    d=dict(zip(FIELDS,map(float,values)))
    div=lambda a,b:d[a]/d[b] if d[b] else None
    return dict(sparsity=div('skipped','eligible'),retained_mass=div('mass_sum','rows'),
        relative_error=(d['error_sq']/d['dense_sq'])**.5 if d['dense_sq'] else 0,
        mean_query_error=div('query_error_sum','rows'),important_recall=div('important_kept','important'),
        newly_important_miss_rate=div('new_important_missed','new_important'),
        prefix_sparsity=div('prefix_skipped','prefix_eligible'),canvas_sparsity=div('canvas_skipped','canvas_eligible'),
        mixed_sparsity=div('mixed_skipped','mixed_eligible'))


def csv_write(path,rows):
    if not rows:return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def predictor_scores(raw,predictor):
    # Collector v1 used value_rms/value_max for BOTH weighted-score keys and
    # producer-norm side information; the latter overwrote only the rank log,
    # not masks or output diagnostics. Reconstruct the exact causal multiplier
    # from the separately preserved last-mass score and producer summary.
    if predictor in ('value_rms','value_max'):return raw['last_mass']*raw[predictor]
    return raw[predictor]


def rank_correlations(a,b,eligible):
    """Vectorized tie-aware Spearman over eligible tiles, one result per head."""
    def centered(x):
        assert np.isfinite(x[eligible]).all()
        ranks=rankdata(np.where(eligible,x,np.inf),method='average',axis=-1)
        mean=(ranks*eligible).sum(-1)/eligible.sum(-1).clip(min=1)
        return (ranks-mean[:,None])*eligible
    ar,br=centered(a),centered(b)
    norm=np.sqrt((ar*ar).sum(-1)*(br*br).sum(-1))
    valid=(eligible.sum(-1)>1)&(norm>0)
    return ((ar*br).sum(-1)/np.maximum(norm,1e-30))[valid].tolist()


def paired_ablations(prompt_buckets):
    """Prompts, not heads/calls, are independent. Negative error deltas help."""
    index={}
    for (ident,benchmark,name,kind),values in prompt_buckets.items():
        index[ident,benchmark,name,kind]=values
        key=(ident,benchmark,name,'overall')
        index[key]=index.get(key,np.zeros(len(FIELDS)))+values
    groups=defaultdict(list);unmatched=defaultdict(int)
    for (ident,benchmark,name,kind),values in index.items():
        method,suffix=name.rsplit('_s',1)
        if method.startswith(('oracle_','random_')):continue
        baseline='random_'+method.removeprefix('protect_') if method.startswith('protect_') else ('proxy' if method=='last_mass' else 'last_mass')
        other=index.get((ident,benchmark,baseline+'_s'+suffix,kind))
        if other is None:continue
        key=(benchmark,name,baseline+'_s'+suffix,kind)
        # Any rescue/excess can violate exact local budgets. Conservatively omit
        # that prompt comparison and disclose it; all raw results remain intact.
        if values[0]!=other[0] or values[1]!=other[1] or values[9:11].sum()+other[9:11].sum()>0:
            unmatched[key]+=1;continue
        a,b=ratios(values),ratios(other)
        groups[key].append((a['relative_error']-b['relative_error'],a['retained_mass']-b['retained_mass']))
    rows=[]
    for key,data in sorted(groups.items()):
        benchmark,name,baseline,kind=key;x=np.asarray(data);rng=np.random.default_rng(42)
        samples=x[rng.integers(0,len(x),(4000,len(x))),0].mean(-1);lo,hi=np.quantile(samples,[.025,.975])
        rows.append(dict(benchmark=benchmark,candidate=name,control=baseline,attention_type=kind,
            paired_prompts=len(x),unmatched_prompts=unmatched[key],mean_error_delta=float(x[:,0].mean()),
            error_delta_ci_low=float(lo),error_delta_ci_high=float(hi),mean_mass_delta=float(x[:,1].mean()),
            fraction_prompts_lower_error=float((x[:,0]<0).mean())))
    return rows


def report(root=ROOT):
    protocol=json.loads((root/'protocol.json').read_text());expected={r['id'] for r in protocol['development']}
    buckets=defaultdict(lambda:np.zeros(len(FIELDS),np.float64));counts=defaultdict(int)
    tails=defaultdict(list);prompt_buckets=defaultdict(lambda:np.zeros(len(FIELDS),np.float64))
    fingerprints=set();seen=set();ranks=defaultdict(list);norms=defaultdict(list);rediscovery=[]
    for path in sorted((root/'screen'/'shards').glob('*.json')):
        shard=json.loads(path.read_text());assert shard['id'] in expected and shard['id'] not in seen
        from .audit import validate_consecutive_coverage
        validate_consecutive_coverage(shard['records'],layers=shard['trace_audit']['layers'],heads=shard['trace_audit']['heads'])
        seen.add(shard['id']);fingerprints.add(shard['fingerprint']);benchmark=shard['benchmark'];temporal={}
        for record in shard['records']:
            with np.load(record['path']) as raw:
                if 'selected_q0' in raw:
                    masks=raw['selected_q0'][:,0]
                    for i,name in enumerate(record['names']):
                        if name.startswith('oracle_'):continue
                        suffix=name.rsplit('_s',1)[1];oracle=masks[record['names'].index('oracle_mass_s'+suffix)]
                        tkey=(record['layer'],name);previous=temporal.get(tkey)
                        if previous is None or not record['history_available']:
                            pending={};old=np.zeros_like(oracle)
                        else:old,pending=previous
                        if record['history_available']:
                            for h,t in zip(*np.where(oracle&~old&~masks[i])):pending.setdefault((int(h),int(t)),record['step'])
                            for (h,t),began in list(pending.items()):
                                status='rediscovered' if oracle[h,t] and masks[i,h,t] else ('lost_importance' if not oracle[h,t] else None)
                                if status:
                                    rediscovery.append(dict(id=shard['id'],benchmark=benchmark,method=name,layer=record['layer'],
                                        attention_type=record['attention_type'],head=h,query_tile=0,tile=t,began_step=began,
                                        end_step=record['step'],delay=record['step']-began,status=status));del pending[h,t]
                        temporal[tkey]=(oracle.copy(),pending)
                if not record['history_available']:continue # cold/reset masks are not temporal predictions
                metrics=raw['metrics'];assert tuple(record['fields'])==FIELDS
                assert np.isfinite(metrics).all() and (metrics[...,1]<=metrics[...,0]).all()
                for i,name in enumerate(record['names']):
                    data=metrics[i].sum(axis=(0,1,2));target=int(name.rsplit('_s',1)[1])/100
                    prompt_buckets[shard['id'],benchmark,name,record['attention_type']]+=data
                    for dimension,label in (('overall','all'),('attention_type',record['attention_type']),
                        ('layer',str(record['layer'])),('step',str(record['step'])),('task',shard['task'])):
                        key=benchmark,name,dimension,label
                        buckets[key]+=data;counts[key]+=metrics.shape[2]*metrics.shape[3]
                        tails[key].extend(metrics[i,...,7].ravel().tolist())
                    for h in range(metrics.shape[2]):
                        key=benchmark,name,'head',str(h);buckets[key]+=metrics[i,:,h].sum(axis=(0,1));counts[key]+=metrics.shape[3]
                    for q in range(metrics.shape[3]):
                        key=benchmark,name,'query_tile',str(q);buckets[key]+=metrics[i,:,:,q].sum(axis=(0,1));counts[key]+=metrics.shape[2]
                # Raw sampled-Q0 rank diagnostics use tie-aware scipy Spearman.
                eligible=raw['eligible'][0]
                for predictor in protocol['screening']['configs']:
                    if predictor=='last_mask':continue # target-dependent binary ranks are stored separately
                    ranks[benchmark,predictor,record['attention_type']].extend(
                        rank_correlations(predictor_scores(raw,predictor)[0],raw['oracle_mass'][0],eligible))
                for h,e in enumerate(eligible):
                    v=raw['value_rms'][0,h,e]
                    if len(v):norms[benchmark,record['attention_type']].append(float(v.std()/max(v.mean(),1e-12)))
        for (layer,name),(oracle,pending) in temporal.items():
            layer_records=[r for r in shard['records'] if r['layer']==layer]
            last=max(r['step'] for r in layer_records);kind=layer_records[0]['attention_type']
            for (h,t),began in pending.items():
                rediscovery.append(dict(id=shard['id'],benchmark=benchmark,method=name,layer=layer,attention_type=kind,
                    head=h,query_tile=0,tile=t,began_step=began,end_step=last,delay=last-began,status='end_of_trace_censored'))
    assert len(fingerprints)<=1,'mixed code fingerprints'
    if fingerprints:
        smoke=json.loads((root/'smoke.json').read_text())
        assert smoke['passed'] and fingerprints=={smoke['fingerprint']},'raw shards lack matching passing CUDA smoke'
    rows=[]
    for (benchmark,name,dimension,label),values in sorted(buckets.items()):
        key=benchmark,name,dimension,label
        rows.append(dict(benchmark=benchmark,method=name,dimension=dimension,group=label,
            target_sparsity=int(name.rsplit('_s',1)[1])/100,**dict(zip(FIELDS,values.tolist())),**ratios(values),
            query_tiles=counts[key],mean_tile_query_p95=float(np.mean(tails[key])) if tails[key] else None))
    csv_write(root/'screen_summary.csv',rows)
    prompt_rows=[dict(id=i,benchmark=b,method=n,attention_type=t,**dict(zip(FIELDS,v.tolist())),**ratios(v)) for (i,b,n,t),v in sorted(prompt_buckets.items())]
    csv_write(root/'screen_per_prompt.csv',prompt_rows)
    csv_write(root/'screen_paired_ablations.csv',paired_ablations(prompt_buckets))
    rank_rows=[dict(benchmark=b,predictor=p,attention_type=t,mean_spearman=float(np.mean(x)),head_call_count=len(x)) for (b,p,t),x in sorted(ranks.items())]
    csv_write(root/'screen_rank_correlations.csv',rank_rows)
    csv_write(root/'screen_rediscovery_q0.csv',rediscovery)
    _write(root/'screen_audit.json',dict(expected=len(expected),completed=len(seen),missing=sorted(expected-seen),
        complete=seen==expected,fingerprints=sorted(fingerprints),source='completed sample shards only',
        temporal_steps='1 through 7, first canvas only',executed_qk_saved=0,executed_pv_saved=0,
        value_rms_cv=[dict(benchmark=b,attention_type=t,mean_cv=float(np.mean(x)),head_call_count=len(x)) for (b,t),x in sorted(norms.items())]))
    text=['# Pre-QK router — development screening', '',f'Completed: {len(seen)}/{len(expected)} development prompts. Held-out prompts remain unevaluated.',
        '', 'This is dense-trajectory diagnostic screening, not online sparse quality or measured speedup. '
        'Masks are fixed before current QK. History is acquired from previous dense steps. '
        'Only consecutive steps 1–7 of the first canvas enter temporal comparisons; step 0 is initialization. '
        'All heads, layers and query tiles are included. Prefix-only, canvas-only and mixed boundary tiles form disjoint physical populations.',
        '', '| Benchmark | Candidate | Target | Actual | Mass | Rel. output error | Important recall |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['dimension']=='overall':text.append(f"| {r['benchmark']} | {r['method']} | {r['target_sparsity']:.0%} | {r['sparsity']:.2%} | {r['retained_mass']:.4f} | {r['relative_error']:.4f} | {r['important_recall']:.3f} |")
    text+=['','Physical sparsity is sum(skipped)/sum(eligible). The overall error is sqrt(sum squared error / sum squared dense output). '
        'Per-tile query p95 values are not a pooled query percentile. Rank-correlation samples cover Q tile 0; '
        'these attention observations are not independent benchmark samples. Full rollout quality and prompt-level uncertainty are still required.',
        '', 'Stages 2–4 and candidate advancement decisions are pending. No net computation saving is established.']
    (root/'screen_report.md').write_text('\n'.join(text)+'\n')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,benchmark in zip(axes,('longbench','aime24')):
            for method in ('last_mass','ema_mass','frequency','peak','proxy','oracle_mass','oracle_blasst'):
                subset=[r for r in rows if r['dimension']=='overall' and r['benchmark']==benchmark and r['method'].rsplit('_s',1)[0]==method]
                if subset:ax.plot([r['sparsity'] for r in subset],[r['relative_error'] for r in subset],'o-',label=method)
            ax.set(title=benchmark,xlabel='Measured physical tile sparsity',ylabel='Same-state relative output error');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(root/'screen_error_curve.png',dpi=160);plt.close(fig)
