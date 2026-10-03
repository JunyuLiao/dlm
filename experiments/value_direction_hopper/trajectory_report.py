"""Regenerate trajectory diagnostics exclusively from completed JSON shards."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from .experiment import atomic, sha
from .report import agreement, paired_ci


def mean(values):
    return float(np.mean(values)) if len(values) else None


def native_confidence(entropy,dtype,threshold):
    values=torch.tensor(entropy,dtype=getattr(torch,dtype.split('.')[-1]))
    value=values.mean()
    return float(value),bool(value<threshold)


def correlation(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:
        return None
    return float(np.corrcoef(x,y)[0,1])


def csvwrite(path,rows):
    if not rows:
        return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def comparison(s,d):
    a,b=np.array(s['top1']),np.array(d['top1'])
    active=np.array(s['active_before'])|np.array(d['active_before'])
    sa,da=np.array(s['accepted']),np.array(d['accepted'])
    confidence=np.array(s['confidence'])-np.array(d['confidence'])
    both=sa|da
    return dict(disagreement=float((a!=b).mean()),
        active_disagreement=mean((a!=b)[active]),
        commit_disagreement=float((sa!=da).mean()),sparse_only=int((sa&~da).sum()),
        dense_only=int((da&~sa).sum()),jaccard=float((sa&da).sum()/max(1,both.sum())),
        confidence_delta=float(confidence.mean()),
        same_token_confidence_delta=mean(confidence[a==b]),
        changed_token_confidence_delta=mean(confidence[a!=b]),
        delayed_same_token=int((~sa&da&(a==b)).sum()),
        delayed_changed_token=int((~sa&da&(a!=b)).sum()))


def report(root):
    root=Path(root)
    config=json.loads((root/'configuration.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    raw={};summary=[];sample_rows=[];step_rows=[];paired=[];aligned=[];attention_rows=[];position_rows=[];violations=[]
    manifest_by_id={r['id']:r for r in manifest}
    for regime in ('adaptive','fixed512'):
        for path in sorted((root/regime/'shards').glob('*/*.json')):
            record=json.loads(path.read_text())
            if record['fingerprint']!=config['fingerprint'] or sha(record['trace_path'])!=record['trace_sha256']:
                raise ValueError(f'Changed shard: {path}')
            row=manifest_by_id[record['id']]
            if any(record[k]!=row[k] for k in ('prompt_hash','seed','generation_budget','task')):
                raise ValueError('Prompt/settings mismatch')
            with gzip.open(record['trace_path'],'rt') as f:
                trace=json.load(f)
            steps=trace['steps']
            # Compress in-memory analysis state; original detailed records remain on disk.
            routing_counts=defaultdict(lambda:dict(eligible=0,skipped=0))
            for r in trace['routing']:
                for key in ('eligible','skipped'):
                    routing_counts[r['attention_type']][key]+=r[key]
            trace['routing']=[dict(attention_type=k,**v) for k,v in routing_counts.items()]
            for s in steps:
                for k in ('confidence','entropy','top1','accepted','active_before','unstable_positions','margin'):
                    s[k]=np.asarray(s[k])
                s['raw']={'confidence':np.asarray(s['raw']['confidence'])}
                s['accepted_nonargmax_mask']=(np.asarray(s['sampled_token'])!=s['top1']) & s['accepted']
                s.pop('sampled_token',None);s.pop('logit_margin',None)
            if len(steps)!=record['metadata']['actual_denoising_step_count'] or (regime=='fixed512' and len(steps)!=512):
                raise ValueError('Step count mismatch')
            if abs(steps[-1]['draft_score']-record['score'])>1e-6:
                raise ValueError('Final draft scorer disagrees with normal benchmark score')
            raw[regime,record['method'],record['id']]=(record,trace)
    for (regime,method,identifier),(record,trace) in raw.items():
        steps=trace['steps'];errors=defaultdict(list)
        for e in trace['attention']:
            errors[e['step']].append(e)
            attention_rows.append(dict(regime=regime,method=method,id=identifier,**e))
        previous=np.zeros(256,dtype=bool);ever=previous.copy();unstable_count=np.zeros(256,dtype=int)
        above_count=np.zeros(256,dtype=int);near_count=np.zeros(256,dtype=int)
        rejected_count=np.zeros(256,dtype=int);revoked_count=np.zeros(256,dtype=int)
        entropy_sum=np.zeros(256);entropy_peak=np.zeros(256)
        own=[]
        for s in steps:
            accepted=np.asarray(s['accepted'],bool);active=np.asarray(s['active_before'],bool)
            confidence=np.asarray(s['confidence']);entropy=np.asarray(s['entropy'])
            # The native comparison rounds both mean and scalar threshold in
            # the entropy tensor's dtype. FP64 reanalysis can flip boundary cases.
            native_mean,confident=native_confidence(entropy,s['entropy_dtype'],s['confidence_threshold'])
            if (s['stable'] and confident)!=s['would_stop']:
                raise ValueError('Reconstructed stopping condition disagrees with native stopper')
            newly=accepted&~previous;revoked=previous&~accepted;ever|=accepted
            unstable=np.asarray(s['unstable_positions'],bool);unstable_count+=unstable
            above_count+=(entropy>=s['confidence_threshold'])
            near_count+=(abs(entropy-s['confidence_threshold'])<=.0025)
            rejected_count+=~accepted;revoked_count+=revoked
            entropy_sum+=entropy;entropy_peak=np.maximum(entropy_peak,entropy)
            es=errors[s['step']];den=sum(x['dense_sq'] for x in es)
            error=(sum(x['error_sq'] for x in es)/den)**.5 if den else (0. if method=='native_dense' else None)
            eos=set(record['metadata']['special_token_ids']['eos_token_ids'])
            eos_index=next((i for i,t in enumerate(s['top1']) if t in eos),256)
            answer_end=min(record['generation_budget'],eos_index+1)
            answer_entropy,answer_confident=native_confidence(entropy[:answer_end],s['entropy_dtype'],s['confidence_threshold'])
            answer_stable=not bool(unstable[:answer_end].any())
            quant=np.quantile(confidence,[.1,.25,.5,.75,.9])
            entry=dict(regime=regime,method=method,id=identifier,task=record['task'],step=s['step'],
                normalized_step=s['step']/len(steps),accepted=int(accepted.sum()),newly_accepted=int(newly.sum()),
                revoked=int(revoked.sum()),ever_accepted=int(ever.sum()),fraction_accepted=float(accepted.mean()),
                remaining_renoised=int((~accepted).sum()),active_before=int(active.sum()),
                confidence_mean=float(confidence.mean()),confidence_median=float(quant[2]),
                confidence_p10=float(quant[0]),confidence_p25=float(quant[1]),confidence_p75=float(quant[3]),confidence_p90=float(quant[4]),
                raw_confidence_mean=float(np.mean(s['raw']['confidence'])),
                active_confidence=mean(confidence[active]),accepted_confidence=mean(confidence[accepted]),
                active_margin=mean(np.array(s['margin'])[active]),entropy_mean=float(native_mean),
                confident=confident,stable=s['stable'],would_stop=s['would_stop'],
                terminated=s['terminated'],unstable=int(unstable.sum()),
                unstable_within_output=int(unstable[:answer_end].sum()),unstable_after_output=int(unstable[answer_end:].sum()),
                output_end=answer_end,attention_error=error,decoder_ms=s['decoder_ms'],draft_score=s['draft_score'])
            entry.update(output_region_entropy=answer_entropy,output_region_confident=answer_confident,
                         output_region_stable=answer_stable,
                         output_region_would_stop=answer_stable and answer_confident,
                         only_outside_blocks_stop=answer_stable and answer_confident and not s['would_stop'])
            entry['accepted_nonargmax']=int(s['accepted_nonargmax_mask'].sum())
            entry['accepted_nonargmax_within_output']=int(s['accepted_nonargmax_mask'][:answer_end].sum())
            entry['expected_accepted_nonargmax']=float((1-confidence[accepted]).sum())
            positive_entropy=np.maximum(entropy,0)
            entry['entropy_top8_share']=float(np.sort(positive_entropy)[-8:].sum()/max(1e-12,positive_entropy.sum()))
            entry['entropy_max']=float(entropy.max())
            entry['positions_above_entropy_threshold']=int((entropy>=s['confidence_threshold']).sum())
            ordering=np.argsort(entropy,kind='stable');naccepted=int(accepted.sum())
            boundary=ordering[max(0,naccepted-4):min(len(ordering),naccepted+4)]
            entry['acceptance_boundary_confidence']=mean(confidence[boundary])
            entry['active_confidence_median']=float(np.median(confidence[active])) if active.any() else None
            entry['accepted_confidence_median']=float(np.median(confidence[accepted])) if accepted.any() else None
            step_rows.append(entry);own.append(entry);previous=accepted
        counts=defaultdict(lambda:dict(eligible=0,skipped=0))
        for r in trace['routing']:
            for kind in ('whole',r['attention_type']):
                for key in ('eligible','skipped'):
                    counts[kind][key]+=r[key]
        native=raw.get((regime,'native_dense',identifier))
        matched=raw.get((regime,'kernel_dense',identifier))
        baseline=native or matched
        ag=agreement(record['completion_tokens'],baseline[0]['completion_tokens']) if baseline else {}
        first_stop=next((s['step'] for s in steps if s['would_stop']),None)
        output_stop=next((r for r in own if r['output_region_would_stop']),None)
        error_sq=sum(e['error_sq'] for e in trace['attention']);dense_sq=sum(e['dense_sq'] for e in trace['attention'])
        sample=dict(regime=regime,method=method,id=identifier,task=record['task'],score=record['score'],
            steps=len(steps),tokens=len(record['completion_tokens']),wall_seconds=record['wall_seconds'],
            decoder_seconds=sum(s['decoder_ms'] for s in steps)/1000,
            accepted_per_step=float(np.mean([sum(s['accepted']) for s in steps])),
            confidence=float(np.mean([r['confidence_mean'] for r in own])),
            unstable_steps=sum(not s['stable'] for s in steps),not_confident_steps=sum(not r['confident'] for r in own),
            stable_not_confident=sum(r['stable'] and not r['confident'] for r in own),
            confident_not_stable=sum(r['confident'] and not r['stable'] for r in own),
            both_stopping_conditions_fail=sum(not r['confident'] and not r['stable'] for r in own),
            stopping_criterion_passed=sum(r['would_stop'] for r in own),
            accepted_nonargmax=sum(r['accepted_nonargmax'] for r in own),
            accepted_nonargmax_within_output=sum(r['accepted_nonargmax_within_output'] for r in own),
            unique_unstable_positions=int((unstable_count>1).sum()),
            top8_instability_share=float(np.sort(np.maximum(0,unstable_count-1))[-8:].sum()/max(1,np.maximum(0,unstable_count-1).sum())),
            first_would_stop=first_stop,early_stop_score=(steps[first_stop-1]['draft_score'] if first_stop else None),
            first_output_region_stop=None if output_stop is None else output_stop['step'],
            output_region_stop_score=None if output_stop is None else output_stop['draft_score'],
            only_outside_blocks_stop=sum(r['only_outside_blocks_stop'] for r in own),
            last_draft_score=steps[-1]['draft_score'],
            error_sq=error_sq,dense_sq=dense_sq,
            attention_error=(error_sq/dense_sq)**.5 if dense_sq else 0.,
            matches=ag.get('matches',0),compared=ag.get('compared',0),exact=ag.get('exact',False))
        for kind in ('whole','local','global'):
            for key in ('eligible','skipped'):
                sample[kind+'_'+key]=counts[kind][key]
        sample_rows.append(sample)
        for position in range(256):
            position_rows.append(dict(regime=regime,method=method,id=identifier,position=position,
                final_token_id=int(steps[-1]['top1'][position]),after_final_output=position>=own[-1]['output_end'],
                unstable_after_first=int(max(0,unstable_count[position]-1)),
                rejected_steps=int(rejected_count[position]),revocations=int(revoked_count[position]),
                steps_above_entropy_threshold=int(above_count[position]),mean_entropy=float(entropy_sum[position]/len(steps)),
                peak_entropy=float(entropy_peak[position])))
        if method not in ('native_dense','kernel_dense'):
            for baseline_name in ('native_dense','kernel_dense'):
                dense=raw.get((regime,baseline_name,identifier))
                if not dense:continue
                ds=dense[1]['steps']
                for s,d in zip(steps,ds):
                    aligned.append(dict(regime=regime,method=method,baseline=baseline_name,id=identifier,
                        alignment='step',step=s['step'],dense_step=d['step'],**comparison(s,d)))
                dense_accepted=np.asarray([np.count_nonzero(d['accepted']) for d in ds])
                for s in steps:
                    # Acceptance can decrease: this is nearest accepted fraction, not elapsed time.
                    index=int(np.argmin(np.abs(dense_accepted-np.count_nonzero(s['accepted']))))
                    d=ds[index]
                    aligned.append(dict(regime=regime,method=method,baseline=baseline_name,id=identifier,
                        alignment='accepted_fraction',step=s['step'],dense_step=d['step'],**comparison(s,d)))
                    aligned.append(dict(regime=regime,method=method,baseline=baseline_name,id=identifier,
                        alignment='terminal_dense_reference',step=s['step'],dense_step=ds[-1]['step'],**comparison(s,ds[-1])))
    for regime in ('adaptive','fixed512'):
        methods=config['methods'] if regime=='adaptive' else config['fixed_methods']
        for method in methods:
            samples=[r for r in sample_rows if r['regime']==regime and r['method']==method]
            if not samples:continue
            tasks=defaultdict(list)
            for s in samples:tasks[s['task']].append(s['score'])
            nsteps=sum(s['steps'] for s in samples);ncompare=sum(s['compared'] for s in samples)
            dense_sq=sum(s['dense_sq'] for s in samples)
            item=dict(regime=regime,method=method,n=len(samples),accuracy=float(np.mean([np.mean(v) for v in tasks.values()])),
                realized_steps=nsteps,steps_per_sample=nsteps/len(samples),
                output_tokens_per_step=sum(s['tokens'] for s in samples)/nsteps,
                accepted_positions_per_step=sum(s['accepted_per_step']*s['steps'] for s in samples)/nsteps,
                mean_confidence=float(np.mean([s['confidence'] for s in samples])),
                dense_token_agreement=sum(s['matches'] for s in samples)/ncompare if ncompare else None,
                attention_error=(sum(s['error_sq'] for s in samples)/dense_sq)**.5 if dense_sq else 0.,
                wall_seconds=sum(s['wall_seconds'] for s in samples),decoder_seconds=sum(s['decoder_seconds'] for s in samples),
                stable_not_confident_steps=sum(s['stable_not_confident'] for s in samples),
                confident_not_stable_steps=sum(s['confident_not_stable'] for s in samples),
                both_stopping_conditions_fail=sum(s['both_stopping_conditions_fail'] for s in samples),
                stopping_criterion_passed=sum(s['stopping_criterion_passed'] for s in samples),
                unique_unstable_positions_mean=float(np.mean([s['unique_unstable_positions'] for s in samples])),
                top8_instability_share_mean=float(np.mean([s['top8_instability_share'] for s in samples])))
            for kind in ('whole','global','local'):
                eligible=sum(s[kind+'_eligible'] for s in samples)
                item[kind+'_sparsity']=sum(s[kind+'_skipped'] for s in samples)/eligible if eligible else 0.
            summary.append(item)
            baselines=('native_dense','kernel_dense','kernel_blasst') if method=='kernel_gaussian32' else ('native_dense','kernel_dense')
            for baseline in baselines:
                pairs=[(manifest_by_id[s['id']],raw[regime,method,s['id']][0],raw[regime,baseline,s['id']][0])
                       for s in samples if (regime,baseline,s['id']) in raw]
                if pairs and method!=baseline:
                    paired.append(dict(regime=regime,**paired_ci(pairs,method,baseline)))
    expected=len(manifest)*(len(config['methods'])+len(config['fixed_methods']))
    settings={regime:set() for regime in ('adaptive','fixed512')}
    first_state_checks=[]
    for (regime,method,identifier),(record,trace) in raw.items():
        m=record['metadata']
        settings[regime].add(json.dumps([m['denoising_configuration'],m['sampling'],m['thinking'],m['native_canvas_length']],sort_keys=True))
        if m['denoising_configuration']['max_denoising_steps']!=(512 if regime=='fixed512' else 48):
            violations.append('Unexpected generation maximum: '+regime)
        if regime=='fixed512' and ('adaptive',method,identifier) in raw:
            a=raw['adaptive',method,identifier][1]['steps'][0]
            b=trace['steps'][0]
            first_state_checks.append(bool(np.array_equal(a['top1'],b['top1']) and np.array_equal(a['confidence'],b['confidence'])))
    if any(len(v)>1 for v in settings.values()):
        violations.append('Generation settings differ across methods within a regime')
    if any(not same for same in first_state_checks):
        violations.append('Adaptive/fixed first-state mismatch requires investigation')
    audit=dict(completed=len(raw),expected=expected,complete=len(raw)==expected,violations=violations,
               raw_trace_hashes_verified=True,fixed_steps_verified=True,
               identical_generation_settings=all(len(v)<=1 for v in settings.values()),
               fixed_vs_adaptive_first_step_pairs=len(first_state_checks),
               fixed_vs_adaptive_first_step_mismatches=sum(not b for b in first_state_checks))
    csvwrite(root/'per_sample.csv',sample_rows);csvwrite(root/'per_step.csv',step_rows)
    task_groups=defaultdict(list)
    for r in sample_rows:task_groups[r['regime'],r['method'],r['task']].append(r)
    per_task=[]
    for (regime,method,task),items in sorted(task_groups.items()):
        per_task.append(dict(regime=regime,method=method,task=task,n=len(items),
            accuracy=float(np.mean([r['score'] for r in items])),
            mean_steps=float(np.mean([r['steps'] for r in items])),
            mean_confidence=float(np.mean([r['confidence'] for r in items])),
            sparsity=sum(r['whole_skipped'] for r in items)/max(1,sum(r['whole_eligible'] for r in items))))
    csvwrite(root/'per_task.csv',per_task)
    csvwrite(root/'aligned_disagreement.csv',aligned);csvwrite(root/'attention_errors.csv',attention_rows)
    csvwrite(root/'per_position.csv',position_rows)
    error_groups=defaultdict(lambda:dict(error_sq=0.,dense_sq=0.,cosine_sum=0.,rows=0))
    for r in attention_rows:
        for scope in ('whole',r['attention_type'],'layer_'+str(r['layer'])):
            group=error_groups[r['regime'],r['method'],scope,r['step']]
            for k in ('error_sq','dense_sq','cosine_sum','rows'):
                group[k]+=r[k]
    error_aggregates=[]
    for (regime,method,scope,step),values in sorted(error_groups.items()):
        error_aggregates.append(dict(regime=regime,method=method,scope=scope,step=step,
            relative_l2=(values['error_sq']/max(1e-12,values['dense_sq']))**.5,
            cosine=values['cosine_sum']/max(1,values['rows']),sampled_rows=values['rows']))
    csvwrite(root/'attention_error_aggregates.csv',error_aggregates)
    csvwrite(root/'summary.csv',summary)
    historical={}
    from .trajectory_experiment import BASE
    for method in ('native_dense','kernel_gaussian32','kernel_blasst'):
        previous=[json.loads(p.read_text()) for p in (BASE/'shards'/method).glob('*.json')]
        matched=[(p,raw['adaptive',method,p['id']][0]) for p in previous if ('adaptive',method,p['id']) in raw]
        historical[method]=dict(paired=len(matched),output_mismatches=sum(a['completion_tokens']!=b['completion_tokens'] for a,b in matched),
            step_mismatches=sum(a['metadata']['actual_denoising_step_count']!=b['metadata']['actual_denoising_step_count'] for a,b in matched),
            original_steps=sum(p['metadata']['actual_denoising_step_count'] for p in previous),
            original_wall_seconds=sum(p['wall_seconds'] for p in previous))
        x=np.asarray([p['metadata']['actual_denoising_step_count'] for p in previous],dtype=float)
        y=np.asarray([p['wall_seconds'] for p in previous])
        design=np.column_stack([np.ones(len(x)),x])
        coefficients=np.linalg.lstsq(design,y,rcond=None)[0]
        residual=((y-design@coefficients)**2).sum()
        historical[method]['descriptive_wall_regression']=dict(intercept_seconds=float(coefficients[0]),
            seconds_per_added_step=float(coefficients[1]),r_squared=float(1-residual/((y-y.mean())**2).sum()),
            caveat='Observational across prompts; prompt length/task and number of steps are not randomized')
        if historical[method]['output_mismatches'] or historical[method]['step_mismatches']:
            violations.append('Archived reproduction mismatch: '+method)
    atomic(root/'summary.json',dict(summary=summary,paired_accuracy=paired,audit=audit,historical=historical))
    atomic(root/'audit.json',audit)
    atomic(root/'analysis_provenance.json',dict(source_hashes={str(p.resolve()):sha(p)
        for p in (Path(__file__),Path(__file__).with_name('trajectory_interpret.py'),
                  Path(__file__).with_name('report.py'))},generation_fingerprint=config['fingerprint']))
    plots(root,step_rows,aligned,attention_rows)
    lines=['# DiffusionGemma denoising trajectory diagnosis','',
        f"Completed {len(raw)}/{expected} sample/configuration runs. {'Complete' if audit['complete'] else 'INTERIM: experiments remain incomplete'}.",'',
        'Same cached 130 RULER4K examples (13 tasks x 10), frozen model revision, BF16, seed42, Gaussian32 projection seed1729 and s70 thresholds. '
        'Adaptive generation uses the native 48-step maximum and 0.8 to 0.4 temperature schedule. '
        'Fixed512 suppresses adaptive termination and uses the native 512-step temperature schedule. This stretches annealing and is not a pure stopping-only intervention relative to the 48-step run.','',
        'The native sampler starts with random vocabulary tokens, reselects acceptance using an entropy budget every iteration, and renoises nonaccepted positions. '
        'Acceptance is reversible. There is no permanent masked-token count or monotone commitment progress. Reported remaining positions are those selected for renoising; unique-ever acceptance is stored separately. '
        'The stopper requires all256 argmax tokens to be stable AND mean processed entropy below0.005. It includes positions beyond the returned answer/EOS.','',
        'Native dense reproduces the original backend; kernel_dense retains every tile using the same kernel, masks and numeric path as Gaussian32. '
        'Their difference controls historical local-mask/numeric differences. BLASST capped uses the same faithful online maximum vote kernel with lambda<=1; '
        'kernel_blasst reproduces the frozen matched-sparsity aggressive calibration, which permits lambda>1.','',
        'Whole-model sparsity here pools all routed decoder layers; prefix encoding remains native dense and is not included in the routed-tile denominator. '
        'The original native SDPA backend uses its existing implementations, including the previously documented D512 math fallback.','',
        '|Regime|Method|N|Sparsity|Global|Local|Steps|Output tokens/step|Mean confidence|Token agreement|Sampled attention error|Accuracy|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        ag='n/a' if r['dense_token_agreement'] is None else f"{100*r['dense_token_agreement']:.2f}%"
        lines.append(f"|{r['regime']}|{r['method']}|{r['n']}|{100*r['whole_sparsity']:.2f}%|{100*r['global_sparsity']:.2f}%|{100*r['local_sparsity']:.2f}%|{r['realized_steps']}|{r['output_tokens_per_step']:.3f}|{r['mean_confidence']:.5f}|{ag}|{r['attention_error']:.4f}|{100*r['accuracy']:.2f}%|")
    lines+=['','Token agreement uses positional matches / max output length, continuing after first divergence. Adaptive reference is native dense; fixed reference is kernel_dense. '
            'Confidence is the equal-prompt mean of per-step/per-position processed probabilities. Physical sparsity is summed skipped / summed eligible tiles. '
            'Output tokens/step counts final returned tokens; accepted positions/step is separately reported because positions can be reaccepted.','',
            'Attention error is measured on each sparse trajectory against dense attention on identical QKV and the same structural mask, at layers0/5/29, heads0/8, six fixed query positions. '
            'It includes numerical output differences; native dense is the zero-error reference by definition. This sample does not estimate an all-layer error. '
            'Observed wall and decoder times include diagnostic overhead and must not be presented as production speed benchmarks.','',
            '## Paired accuracy differences','', '|Regime|Method minus baseline|Delta pp|95% paired prompt-bootstrap CI pp|', '|---|---|---:|---|']
    for p in paired:
        lines.append(f"|{p['regime']}|{p['first']} minus {p['second']}|{100*p['mean']:+.3f}|[{100*p['lower']:+.3f}, {100*p['upper']:+.3f}]|")
    lines+=['','## Plots','']
    for name in ('confidence_vs_step','confidence_quantiles_vs_step','tokens_committed_vs_step','remaining_masked_tokens_vs_step','dense_token_disagreement_vs_step','attention_output_error_vs_step','confidence_vs_normalized_step'):
        lines.append(f'![{name}](plots/{name}.png)')
    lines+=['','Curve means condition on examples still running at each step; late points can represent harder samples. Per-position raw traces allow paired reanalysis. '
        'Step alignment compares different states after divergence. Accepted-fraction alignment is also saved, but acceptance is nonmonotone and may saturate; it is not causal state matching.','',
        'Full interpretation and the eleven research questions are finalized only after all experiments and audits finish. '
        'Confidence alone does not establish correctness or safe early termination. Intermediate draft scores are diagnostic repeated evaluations on previously examined prompts.']
    policy=config['original']['policies']
    lines+=['','Frozen threshold provenance: Gaussian32 local/global log(tau) = '+
        ', '.join(str(policy['kernel_gaussian32'][k]['log_threshold']) for k in ('local','global'))+
        '; BLASST lambda=exp(log_scale)/valid_KV_length, local/global log_scale = '+
        ', '.join(str(policy['kernel_blasst'][k]['log_scale']) for k in ('local','global'))+
        '. The capped control applies min(lambda,1). No trajectory or final score selects these thresholds.']
    legacy=[]
    source=Path(config['original']['source'])
    for name,folder in (('earlier_dense','dense/dense/shards'),
                        ('earlier_gaussian32','final/jl_gaussian_r32_s70/shards'),
                        ('earlier_calibrated_blasst','final/blasst_s70/shards')):
        paths=sorted((source/folder).glob('*.json'))
        prior=[json.loads(p.read_text()) for p in paths]
        if not prior:continue
        if {r['id'] for r in prior}!=set(manifest_by_id):
            raise ValueError('Historical implementation cohort differs')
        decoding={json.dumps(r['generation_metadata']['denoising_configuration'],sort_keys=True) for r in prior}
        if len(decoding)!=1:
            raise ValueError('Historical decoding settings are mixed')
        if settings['adaptive'] and json.loads(next(iter(decoding)))!=json.loads(next(iter(settings['adaptive'])))[0]:
            raise ValueError('Historical decoding settings differ from current adaptive run')
        expected_policy=(None if name=='earlier_dense' else
            config['original']['policies']['kernel_gaussian32' if name=='earlier_gaussian32' else 'kernel_blasst'])
        for r in prior:
            row=manifest_by_id[r['id']]
            if any(r[k]!=row[k] for k in ('prompt_hash','seed','generation_budget')):
                raise ValueError('Historical implementation comparison has unmatched inputs')
            if expected_policy is not None and r['thresholds']!=expected_policy:
                raise ValueError('Historical routing thresholds differ')
        legacy.append(dict(method=name,n=len(prior),steps=sum(r['generation_metadata']['actual_denoising_step_count'] for r in prior),
            accuracy=float(np.mean([r['score'] for r in prior])),
            backends=sorted({r['backend'] for r in prior}),
            decoding=json.loads(next(iter(decoding))),
            thresholds=prior[0].get('thresholds'),source_hashes={str(p):sha(p) for p in paths}))
    atomic(root/'earlier_implementation_comparison.json',legacy)
    lines+=['','## Earlier implementation control','',
        'These pre-Hopper cached results use the same130 prompt hashes, seeds, budgets and frozen thresholds. They are historical scientific controls, not contemporary timing measurements. '
        'The earlier Gaussian32 path already required approximately four times the dense iterations. The large iteration increase therefore predates the optimized Hopper kernel. '
        'This does not establish numerical equivalence: outputs/accuracy differ across implementations.','',
        '|Earlier implementation|N|Steps|Accuracy|Backend|','|---|---:|---:|---:|---|']
    for r in legacy:
        lines.append(f"|{r['method']}|{r['n']}|{r['steps']}|{100*r['accuracy']:.2f}%|{', '.join(r['backends'])}|")
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    from .trajectory_interpret import interpret
    interpret(root,summary,sample_rows,step_rows,aligned,paired,historical,audit)
    supplement_path=root/'supplement_reference.json'
    if supplement_path.exists():
        supplement=Path(json.loads(supplement_path.read_text())['root'])
        control=json.loads((supplement/'summary.json').read_text())
        if not control['complete']:
            raise ValueError('Referenced same-schedule supplement is incomplete')
        with (root/'report.md').open('a') as f:
            f.write('\n## Same48-step schedule supplement\n\n'+(supplement/'report.md').read_text())
    if violations:
        raise AssertionError('Reproduction gate failed; investigate before fixed controls: '+repr(violations))
    return audit


def plots(root,rows,aligned,attention_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'plots';out.mkdir(exist_ok=True)
    adaptive=[r for r in rows if r['regime']=='adaptive']
    metrics=[('confidence_vs_step','confidence_mean','Mean top-1 probability'),
             ('raw_confidence_vs_step','raw_confidence_mean','Mean top-1 probability before temperature'),
             ('entropy_vs_step','entropy_mean','Native processed entropy (stop threshold 0.005)'),
             ('tokens_committed_vs_step','accepted','Accepted positions (reversible)'),
             ('remaining_masked_tokens_vs_step','remaining_renoised','Positions renoised'),
             ('attention_output_error_vs_step','attention_error','Shared-QKV relative output error')]
    def curve(data,field,name,ylabel,xfield='step'):
        fig,ax=plt.subplots(figsize=(8,4))
        for method in sorted({r['method'] for r in data}):
            buckets=defaultdict(list)
            for r in data:
                if r['method']==method and r[field] is not None:
                    x=r[xfield] if xfield=='step' else round(r[xfield]*20)/20
                    buckets[x].append(r[field])
            xs=sorted(buckets)
            ax.plot(xs,[np.mean(buckets[x]) for x in xs],label=method)
        ax.set(xlabel=xfield.replace('_',' '),ylabel=ylabel);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=160);plt.close(fig)
    for name,field,label in metrics:curve(adaptive,field,name,label)
    fixed=[r for r in rows if r['regime']=='fixed512']
    if fixed:
        curve(fixed,'draft_score','fixed512_draft_accuracy_vs_step','Mean official draft score (completed prompts)')
        curve(fixed,'confidence_mean','fixed512_confidence_vs_step','Mean top-1 probability')
    curve(adaptive,'confidence_mean','confidence_vs_normalized_step','Mean top-1 probability','normalized_step')
    carried=[]
    paths=defaultdict(list)
    for r in adaptive:paths[r['method'],r['id']].append(r)
    for path in paths.values():
        path.sort(key=lambda r:r['step'])
        carried.extend(path)
        for step in range(path[-1]['step']+1,49):
            carried.append(dict(path[-1],step=step))
    curve(carried,'confidence_mean','confidence_vs_step_carryforward',
          'Mean confidence; terminal observations carried forward')
    curve([r for r in aligned if r['regime']=='adaptive' and r['alignment']=='step' and r['baseline']=='native_dense'],
          'disagreement','dense_token_disagreement_vs_step','Top-1 disagreement with native dense')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,field in zip(axes,('confidence_p10','confidence_median')):
        for method in sorted({r['method'] for r in adaptive}):
            buckets=defaultdict(list)
            for r in adaptive:
                if r['method']==method:buckets[r['step']].append(r[field])
            xs=sorted(buckets);ax.plot(xs,[np.mean(buckets[x]) for x in xs],label=method)
        ax.set(xlabel='step',ylabel=field);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'confidence_quantiles_vs_step.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,kind in zip(axes,('local','global')):
        buckets=defaultdict(lambda:defaultdict(lambda:[0.,0.]))
        for r in attention_rows:
            if r['regime']=='adaptive' and r['attention_type']==kind:
                pair=buckets[r['method']][r['step']]
                pair[0]+=r['error_sq'];pair[1]+=r['dense_sq']
        for method,values in buckets.items():
            xs=sorted(values)
            ax.plot(xs,[(values[x][0]/max(1e-12,values[x][1]))**.5 for x in xs],label=method)
        ax.set(xlabel='step',ylabel='Relative output L2',title=kind);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'attention_error_local_global.png',dpi=160);plt.close(fig)
    lookup={(r['method'],r['id'],r['step']):r for r in adaptive}
    fig,ax=plt.subplots(figsize=(7,4))
    for method in sorted({r['method'] for r in aligned}):
        groups=defaultdict(list)
        for r in aligned:
            if r['regime']=='adaptive' and r['method']==method and r['alignment']=='step' and r['baseline']=='kernel_dense':
                s=lookup.get((method,r['id'],r['step']))
                if s and s['attention_error'] is not None:
                    groups[r['id']].append((s['attention_error'],-r['confidence_delta']))
        points=[np.mean(v,axis=0) for v in groups.values()]
        if points:
            points=np.asarray(points);ax.scatter(points[:,0],points[:,1],s=10,alpha=.5,label=method)
    ax.set(xlabel='Prompt mean sampled attention error',ylabel='Prompt mean confidence loss vs unpruned kernel')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'attention_error_vs_confidence_loss.png',dpi=160);plt.close(fig)
