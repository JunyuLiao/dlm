"""Survival-controlled step contrasts and supplementary diagnostic analyses."""
import argparse
from collections import defaultdict,Counter
import csv
import json
from pathlib import Path
import numpy as np
from .config import ROOT
from .report import ratios,csv_write,rank_correlations
from .routing import FIELDS
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write


def analyze(root=ROOT):
    audit=json.loads((root/'screen_audit.json').read_text())
    if not audit['complete']:raise RuntimeError('complete audited screen required')
    keep_names={'last_mass_s50','oracle_mass_s50','oracle_blasst_s50','protect_diagonal_s50','protect_beginning_s50'}
    buckets=defaultdict(lambda:np.zeros(len(FIELDS),dtype=np.float64));coverage=defaultdict(set)
    deletion=defaultdict(list);deletion_counts=Counter()
    for path in sorted((root/'screen'/'shards').glob('*.json')):
        shard=json.loads(path.read_text());b=shard['benchmark'];ident=shard['id']
        for record in shard['records']:
            if not record['history_available']:continue
            step=record['step'];kind=record['attention_type'];coverage[b,step].add(ident)
            with np.load(record['path']) as raw:
                metrics=raw['metrics']
                for i,name in enumerate(record['names']):
                    if name not in keep_names:continue
                    values=metrics[i].sum(axis=(0,1,2),dtype=np.float64)
                    for group in ('overall',kind):buckets[b,ident,name,group,step]+=values
                valid=raw['eligible'][0] & (raw['deletion_risk_count'][0]>0)
                risk=raw['deletion_risk_mean'][0]
                for signal,key in (('mass','oracle_mass'),('contribution','oracle_contribution'),('value_norm_only','value_rms')):
                    deletion[b,kind,signal].extend(rank_correlations(raw[key][0],risk,valid))
                deletion_counts[b,kind,'finite_query_block_observations']+=int(raw['deletion_risk_count'].sum())
                deletion_counts[b,kind,'finite_rank_blocks']+=int(valid.sum())
                deletion_counts[b,kind,'no_finite_risk_blocks']+=int((raw['eligible'][0]&~valid).sum())
    rows=[]
    for (b,step),ids in sorted(coverage.items()):
        if step==1:continue
        paired=ids&coverage[b,1]
        for name in sorted(keep_names):
            for kind in ('overall','local','global'):
                differences=[];starts=[];ends=[]
                for ident in sorted(paired):
                    first=buckets.get((b,ident,name,kind,1));last=buckets.get((b,ident,name,kind,step))
                    if first is None or last is None:continue
                    a,c=ratios(first),ratios(last);starts.append(a);ends.append(c)
                    differences.append(c['relative_error']-a['relative_error'])
                if not differences:continue
                x=np.asarray(differences);rng=np.random.default_rng(42)
                draws=x[rng.integers(0,len(x),(4000,len(x)))].mean(-1);lo,hi=np.quantile(draws,[.025,.975])
                rows.append(dict(benchmark=b,method=name,attention_type=kind,start_step=1,end_step=step,
                    paired_prompts=len(x),start_step_available_prompts=len(coverage[b,1]),
                    mean_prompt_error_start=float(np.mean([r['relative_error'] for r in starts])),
                    mean_prompt_error_end=float(np.mean([r['relative_error'] for r in ends])),
                    paired_error_delta=float(x.mean()),delta_ci_low=float(lo),delta_ci_high=float(hi),
                    mean_prompt_mass_start=float(np.mean([r['retained_mass'] for r in starts])),
                    mean_prompt_mass_end=float(np.mean([r['retained_mass'] for r in ends]))))
    csv_write(root/'screen_matched_prompt_steps.csv',rows)
    risk_rows=[]
    for (b,kind,signal),values in sorted(deletion.items()):
        if not values:continue
        risk_rows.append(dict(benchmark=b,attention_type=kind,signal=signal,head_calls=len(values),
            mean_spearman_with_mean_deletion_risk=float(np.mean(values)),median_spearman=float(np.median(values)),
            p10_spearman=float(np.quantile(values,.1)),p90_spearman=float(np.quantile(values,.9)),
            finite_query_block_observations=deletion_counts[b,kind,'finite_query_block_observations'],
            finite_rank_blocks=deletion_counts[b,kind,'finite_rank_blocks'],
            no_finite_risk_blocks=deletion_counts[b,kind,'no_finite_risk_blocks']))
    csv_write(root/'screen_deletion_risk.csv',risk_rows)
    events=defaultdict(Counter);delays=defaultdict(list)
    with (root/'screen_rediscovery_q0.csv').open() as f:
        for event in csv.DictReader(f):
            key=(event['benchmark'],event['method'],event['attention_type']);events[key][event['status']]+=1
            if event['status']=='rediscovered':delays[key].append(int(event['delay']))
    event_rows=[]
    for key,counts in sorted(events.items()):
        b,name,kind=key;values=delays[key]
        event_rows.append(dict(benchmark=b,method=name,attention_type=kind,initially_missed_new_important_events=sum(counts.values()),
            rediscovered=counts['rediscovered'],lost_importance=counts['lost_importance'],end_of_trace_censored=counts['end_of_trace_censored'],
            mean_delay_conditional_on_rediscovery=float(np.mean(values)) if values else None,
            median_delay_conditional_on_rediscovery=float(np.median(values)) if values else None))
    csv_write(root/'screen_rediscovery_summary.csv',event_rows)
    _write(root/'temporal_diagnostic_audit.json',dict(source='completed audited development shards',
        coverage=[dict(benchmark=b,step=step,prompts=len(ids),ids=sorted(ids)) for (b,step),ids in sorted(coverage.items())],
        temporal_comparison='same prompt IDs at start and end; per-prompt metrics before averaging; early first canvas only',
        deletion_scope='query tile0, all heads/layers; per-block mean single-deletion risk, alpha<1-1e-6; not exact jointly removed-block ranking',
        rediscovery_scope='query tile0; initially missed newly important mass-oracle tiles; lost importance is a competing event and end-of-trace is censoring'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT)
    analyze(parser.parse_args().output)
