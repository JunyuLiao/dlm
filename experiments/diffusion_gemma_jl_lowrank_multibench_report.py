"""Generalized30/100 raw-shard reporter, reusing existing metrics and NeMo.

The frozen predecessor reporter assumes50 LongBench examples and identical
condition sets per benchmark. This adapter changes those orchestration/count
assumptions only; scorers, sparsity/mass/agreement/error formulas are reused.
"""
from collections import defaultdict
import csv
import gzip
import io
from pathlib import Path
from unittest.mock import patch

from experiments import diffusion_gemma_jl_lowrank_multibench as study
from experiments.diffusion_gemma_jl_focused import report as core
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_value_aware.report_metrics import aggregate,accumulate_marginals,SUM_FIELDS,correlation
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write


def validate_conditions(conditions,policies):
    """Connect each deployed threshold/config to its audited policy explicitly."""
    by_key={(p['benchmark'],p['name']):p for p in policies}
    if len(by_key)!=len(policies):raise ValueError('Duplicate policy identity')
    for label,c in conditions.items():
        name='dense' if label=='dense' else label.rsplit('_s',1)[0]
        if c['name']!=name or c['config']!=({} if name=='dense' else study.CONFIGS[name]) or c['target']!=(0. if name=='dense' else .5):
            raise ValueError('Frozen final condition operator/target changed')
        if c['expected_per_benchmark']!={b:study.COUNTS[b] for b in c['thresholds']}:
            raise ValueError('Final condition denominators changed')
        for b,value in c['thresholds'].items():
            if label not in study.CONDITIONS_BY_BENCHMARK[b]:raise ValueError('Unexpected benchmark condition')
            if name=='dense':
                if value is not None:raise ValueError('Dense unexpectedly has a routing policy')
            elif (b,name) not in by_key or value!=by_key[b,name]['policy'] or c['config']!=by_key[b,name]['config']:
                raise ValueError('Final condition threshold differs from audited policy')


def summarize_raw(raw,conditions):
    groups=defaultdict(list);taskgroups=defaultdict(list);subgroups=defaultdict(list)
    for r in raw:
        b=r['benchmark'];groups[b,'full',r['condition']].append(r)
        split='calibration6' if r['calibration'] else ('noncalibration24' if b=='aime26' else 'noncalibration94')
        groups[b,split,r['condition']].append(r);taskgroups[b,r['task'],r['condition']].append(r)
        if r.get('sub_domain'):subgroups[b,r['sub_domain'],r['condition']].append(r)
    rows=[]
    for (b,split,label),group in sorted(groups.items()):
        required=study.COUNTS[b] if split=='full' else (6 if split=='calibration6' else study.COUNTS[b]-6)
        if len(group)!=required or len({r['id'] for r in group})!=required:continue
        c=conditions[label]
        rows.append(dict(benchmark=b,split=split,condition=label,name=c['name'],target=c['target'],config=c['config'],
            method=c['config'].get('method','centered' if c['name'] in study.ALL else 'dense'),thresholds=c['thresholds'][b],**core.summarize(group)))
    tasks=[dict(benchmark=b,task=t,condition=n,**core.summarize(g)) for (b,t,n),g in sorted(taskgroups.items())]
    subtasks=[dict(benchmark=b,sub_domain=t,condition=n,**core.summarize(g)) for (b,t,n),g in sorted(subgroups.items())]
    return rows,tasks,subtasks,groups


def direct_comparisons(rows,groups):
    result=[]
    for a in rows:
        if a['name'] not in study.NEW_BY_BENCHMARK[a['benchmark']]:continue
        for b in rows:
            if (a['benchmark'],a['split'])!=(b['benchmark'],b['split']) or a['condition']==b['condition']:continue
            ga={r['id']:r['accuracy'] for r in groups[a['benchmark'],a['split'],a['condition']]}
            gb={r['id']:r['accuracy'] for r in groups[b['benchmark'],b['split'],b['condition']]}
            if set(ga)!=set(gb):raise ValueError('Unpaired direct comparison')
            gaps={k:a[k+'_physical_sparsity']-b[k+'_physical_sparsity'] for k in ('overall','global','local')}
            result.append(dict(benchmark=a['benchmark'],split=a['split'],candidate=a['condition'],reference=b['condition'],
                count=len(ga),candidate_correct=a['correct'],reference_correct=b['correct'],delta_pp=100*(a['accuracy']-b['accuracy']),
                paired_ci95_pp=[100*x for x in paired_bootstrap_ci([ga[i]-gb[i] for i in sorted(ga)])],
                **{k+'_sparsity_gap_pp':100*v for k,v in gaps.items()},within2pp_all_types=all(abs(v)<=.02+1e-12 for v in gaps.values())))
    return result


def audit_diagnostics(root,setup):
    diag,sources=core.common.diagnostic_summary(root)
    selected=study.shared_analysis.selected_sources(study.read(root/'shared_state_index.json'))
    expected={(s['path'],label.rsplit('_s',1)[0],.5) for s in selected.values()
              for label in study.CONDITIONS_BY_BENCHMARK[s['id'].split('/')[0]] if label!='dense'}
    actual=[]
    for item in study.read(root/'shared_diagnostics_index.json'):
        ident=study.read(item['path'])['identity'];actual.append((ident['source']['path'],ident['name'],ident['target']))
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError(f'Incomplete shared diagnostic coverage {len(actual)}/{len(expected)}')
    return diag,sources


def write_report(root,setup,rows,tasks,direct,diag,audit):
    display=[dict(benchmark=r['benchmark'],split=r['split'],condition=r['condition'],threshold=core.threshold_text(r),
        correct=r['correct'],count=r['count'],accuracy=100*r['accuracy'],delta_pp=100*r['delta'],ci95_pp=[100*v for v in r['paired_ci95']],
        whole=100*r['overall_physical_sparsity'],global_s=100*r['global_physical_sparsity'],local_s=100*r['local_physical_sparsity'],
        mass=100*r['overall_mass'],agreement=100*r['token_agreement'],local_operator_error=r['overall_relative_error'],length_limited=r['length_terminated']) for r in rows]
    core.csv_write(root/'main_results.csv',display)
    text=f"# Gaussian low-rank centered routing: AIME26 and LongBench v2-100\n\nStatus: {'COMPLETE' if audit['complete'] else 'INCOMPLETE'}; {audit['completed']}/{study.EXPECTED} audited outputs.330 new final slots plus930 reused comparison slots.\n\n"
    text+='## Setup and exact sample selection\n\nUnchanged30 AIME26 questions and all100 questions from the existing newer LongBench v2 NeMo study; no new sample selection. AIME retains2048 output tokens and its existing scorer/prompt; LongBench retains4096 output tokens,32K input cap and the pinned actual NeMo prompt/MCQ evaluator. '+str(sum(r.get('truncated',False) for r in setup['final'] if r['benchmark']=='longbench_v2'))+'/100 LongBench inputs are context-truncated; these are not full-context leaderboard results.\n\n'
    text+=core.table([dict(domain=k,count=v) for k,v in setup['selection']['longbench_domains'].items()],('domain','count'))+'\n\n'
    text+='BF16 DiffusionGemma revision '+setup['revision']+',seed42,canvas256,up to48 denoising steps,thinkingFalse,unchanged native0.4–0.8 sampling schedule (temperature0 sentinel is not greedy).128-query×64-KV physical tiles, prefix+canvas eligible, native structural masks/GQA. New runs: Gaussian centered rank1 on AIME and ranks1/2/8 on LongBench. All earlier compatible comparisons, including both50-question Gaussian32/full-reference cohorts, are imported without numerical changes.\n\n'
    text+='## Projection and threshold rules\n\nThe existing online attention-weighted projected centered update, valid-KV RMS reference, strict worst-query physical gate, tie/first-support retention and unchanged state on skip are reused. Gaussian matrices use seed1729 and unchanged layer/native-head/rank-dependent construction/hashes. Token-value sketches and routing arithmetic are FP32; retained attention uses original V with ordinary renormalization. Rank1 only extends a process-local dimension allow-list. Existing generic GPU kernels are unchanged and validated against the trusted reference. No projection direction is selected using final scores.\n\n'
    text+='Each new benchmark/rank starts with that benchmark\'s existing Gaussian32 scalars as an unverified trial. Rank-specific empirical CDFs come from540 historical calibration QKV snapshots; no dense generation is replayed. The unchanged local/global proposal/scalar fallback verifies up to12 points on the same six full-budget questions per benchmark, accepting only overall/global/local48–52% calibration sparsity. Policies freeze before final inference and are never retuned using final results. Baseline/full/Gaussian32 calibration conventions remain those of their sources; this asymmetry is retained and documented, not silently normalized away.\n\n'
    for b in study.COUNTS:text+=b+' calibration IDs: '+', '.join(setup['calibration_ids'][b])+'.\n\n'
    text+='Both headline sets include these six calibration members. Separate noncalibration24/94 results follow. LongBench verification IDs are the existing score-blind one-per-domain selection from the earlier50, not a new selection from100. Historical LongBench proposal/diagnostic states come from an older calibration set, as in the prior studies. All final examples were previously examined; no fresh held-out confirmation claim.\n\n'
    text+='## Per-domain results before aggregation\n\n'+core.table([dict(domain=r['task'],condition=r['condition'],correct=r['correct'],count=r['count'],accuracy=100*r['accuracy']) for r in tasks if r['benchmark']=='longbench_v2'],('domain','condition','correct','count','accuracy'))+'\n\n'
    text+='## Main results\n\n'
    for b in study.COUNTS:
        for split in ('full','noncalibration24' if b=='aime26' else 'noncalibration94','calibration6'):
            group=[r for r in display if r['benchmark']==b and r['split']==split]
            text+=f'### {b}: {split}\n\n'+core.table(group,tuple(display[0]) if display else ())+'\n\n'
    text+='## New ranks versus Gaussian32/full-dimensional\n\n'+core.table([r for r in direct if r['split']=='full' and r['reference'] in ('jl_gaussian_r32_s50','full_centered_s50')],
        ('benchmark','candidate','reference','candidate_correct','reference_correct','delta_pp','paired_ci95_pp','overall_sparsity_gap_pp','global_sparsity_gap_pp','local_sparsity_gap_pp','within2pp_all_types'))+'\n\n'
    text+='All direct comparisons (including dense, BLASST and mass) are exported. Actual sparsity and local/global allocations, not target labels, define comparability. Original BLASST\'s capped lambda1 may be unattainable at50%; its lower attained point is not relabeled. Aggressive BLASST allows lambda>1 under its existing rule. Mass is the existing max-based mass bound, not exact mass-only routing.\n\n'
    text+='## Measurements, diagnostics and limitations\n\nPhysical sparsity is SUM(skipped eligible tiles)/SUM(eligible tiles), excluding unchanged dense prefill. Retained dense mass and full-dimensional local output error use corresponding QKV in each execution; they are not aligned dense-generation states after divergence. Shared-QKV diagnostics separately use32 early AIME and34 early LongBench states per method, one head/query tile per sampled state, with error=sqrt(SUM(error squared)/SUM(dense output squared)). Risk underestimation, overlapping type statistics and layer/head/step/type/region breakdowns are exported. These early states do not exhaust final generation.\n\n'
    text+='Token agreement compares all generated positions after divergence; missing/extra positions disagree and EOS is included. Benchmark accuracy is micro, with domain breakdowns and equal-task macro exported. Paired prompt-bootstrap95% intervals use the existing20000-draw procedure, condition on fixed policies/seeds and are exploratory, not multiplicity-corrected. Rank changes also change random directions; one seed does not establish dimension/seed robustness.\n\n'
    text+='QK, block softmax and projected PV remain computed. Original full-dimensional PV is diagnostic-only for projected routing; retained native PV remains a masked dense-shaped matmul. Sketch projection/storage/refresh/reuse is recorded. No hardware-speedup/FlashAttention claim. Cached native and rank2/8 smoke outputs are raw-audited; rank1 receives fresh model-level unpruned-native and pruned-trusted checks.\n\n'
    for b in study.COUNTS:text+=f'![{b} tradeoffs](figures/{b}_tradeoffs.png)\n\n![{b} target versus actual](figures/{b}_target_actual.png)\n\n'
    text+=f"Missing={len(audit['missing'])}; violations={len(audit['violations'])}. Failures remain preserved and independent configurations continue. Reproduce from raw shards without inference: `python -m {study.MODULE} report`, followed by `verify`.\n"
    (root/'report.md').write_text(text)


def regenerate(root=study.ROOT):
    setup,contract=study.prepare(root),study.execution(root)
    conditions={};missing=[];violations=[];sources=dict(contract['sources'])
    for label in setup['conditions']:
        p=root/'final_configs'/f'{label}.json'
        if not p.exists():missing.append(dict(condition=label,error='No frozen condition'));continue
        c=study.read(p)
        if c['fingerprint']!=contract['fingerprint']:raise ValueError('Final fingerprint mismatch')
        core.evidence.check_sources(c['sources']);sources.update(c['sources']);core.source(sources,p);conditions[label]=c
    policies=[]
    for p in sorted((root/'policies').glob('*/*.json')):
        policy=study.read(p);study.audit_policy(root,policy,setup,contract);policies.append(policy);core.source(sources,p)
    validate_conditions(conditions,policies)
    try:sources.update(study.smoke_audit(root,setup,contract))
    except Exception as e:violations.append(dict(stage='smoke',error=str(e)))
    raw=[];dimensions={};matrices={};bank=Projections();work=[];grades=[]
    temp=root/'per_layer_head_step.csv.gz.tmp'
    with study.dimension_scope(),temp.open('wb') as binary,gzip.GzipFile(filename='',fileobj=binary,mode='wb',compresslevel=3,mtime=0) as gz:
        with io.TextIOWrapper(gz,encoding='utf-8',newline='') as stream:
            writer=None
            for row in setup['final']:
                outputs={};b=row['benchmark']
                for label in setup['conditions_by_benchmark'][b]:
                    if label not in conditions or b not in conditions[label]['thresholds']:
                        missing.append(dict(id=row['id'],condition=label,error='Benchmark policy/config absent'));continue
                    c=conditions[label];stage='dense' if label=='dense' else 'final'
                    try:
                        out=study.runner.cached(None,root,row,stage,label,c['name'],c['config'],c['thresholds'][b],contract)
                        with patch.object(core.common,'PROJECTED',study.ALL):core.common.audit_matrices(out,c['name'],c['config'],bank,matrices)
                        core.source(sources,study.shard_path(root,stage,label,row['id']));sources[out['records_source']['path']]=out['records_source']['sha256']
                        if out.get('imported_source'):sources[out['imported_source']['path']]=out['imported_source']['sha256']
                        outputs[label]=out
                    except FileNotFoundError as e:missing.append(dict(id=row['id'],condition=label,error=str(e)))
                    except Exception as e:violations.append(dict(id=row['id'],condition=label,error=str(e)))
                if 'dense' not in outputs:continue
                try:
                    if b=='longbench_v2':paired,g=core.nemo_pairs(row,outputs);grades.extend(g)
                    else:paired=[dict(condition=n,**core.evidence.pair(row,o,outputs['dense'])) for n,o in outputs.items()]
                except Exception as e:violations.append(dict(id=row['id'],stage='pair',error=str(e)));continue
                for p in paired:
                    label=p['condition'];out=outputs[label]
                    raw.append(dict(**p,sub_domain=row.get('sub_domain'),difficulty=row.get('difficulty'),input_tokens=len(row['prompt_tokens']),
                        truncated=row.get('truncated',False),prediction=out['prediction'],completion_tokens=out['completion_tokens'],backend=out['backend']))
                    if out.get('work_accounting'):work.append(dict(id=row['id'],condition=label,**out['work_accounting']))
                    records=[r for r in out['records'] if r['probe']=='execution'];accumulate_marginals(dimensions,records,b,label)
                    for r in records:
                        item=dict(id=row['id'],benchmark=b,condition=label,**r)
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(item));writer.writeheader()
                        writer.writerow(item)
    temp.replace(root/'per_layer_head_step.csv.gz')
    diag=[]
    try:diag,ds=audit_diagnostics(root,setup);sources.update(ds)
    except Exception as e:violations.append(dict(stage='shared_diagnostics',error=str(e)))
    rows,tasks,subtasks,groups=summarize_raw(raw,conditions)
    with patch.object(core.common,'PROJECTED',study.ALL):compared=core.common.compare(rows,groups)
    direct=direct_comparisons(rows,groups)
    correlations=[dict(benchmark=b,metric=key,pearson_with_accuracy=correlation(
        [r[key] for r in rows if r['benchmark']==b and r['split']=='full' and r['name']!='dense'],
        [r['accuracy'] for r in rows if r['benchmark']==b and r['split']=='full' and r['name']!='dense']))
        for b in study.COUNTS for key in ('overall_mass','token_agreement','overall_relative_error')]
    audit=dict(complete=len(raw)==study.EXPECTED and not missing and not violations,completed=len(raw),expected=study.EXPECTED,
        missing=missing,violations=violations,sources=sources,fingerprint=contract['fingerprint'])
    exports=(('per_sample',raw),('summary',rows),('thresholds',policies),('comparisons',compared),('dimension_pairwise',direct),
        ('shared_operator',diag),('work_accounting',work),('projection_matrices',matrices),('nemo_grades',grades),('correlations',correlations))
    for name,data in exports:
        _write(root/f'{name}.json',data)
        if isinstance(data,list):core.csv_write(root/f'{name}.csv',data)
    core.csv_write(root/'per_task.csv',tasks);core.csv_write(root/'per_subtask.csv',subtasks)
    core.csv_write(root/'routing_marginals.csv',[dict(benchmark=b,condition=n,axis=a,index=i,attention_type=k,
        **aggregate([dict(zip(SUM_FIELDS,v))])) for (b,n,a,i,k),v in sorted(dimensions.items())])
    write_report(root,setup,rows,tasks,direct,diag,audit);study.accounting(root)
    from matplotlib.figure import Figure
    title=Figure.suptitle
    with patch.object(Figure,'suptitle',lambda fig,s,*a,**kw:title(fig,s.replace('50/75% targets','50% target; Gaussian low-rank sweep'),*a,**kw)):
        figures=core.plots(root,rows) if rows else []
    names=['per_layer_head_step.csv.gz','per_task.csv','per_subtask.csv','routing_marginals.csv','main_results.csv','report.md','inference_accounting.json']
    for name,data in exports:names.extend([name+'.json']+([name+'.csv'] if isinstance(data,list) else []))
    audit['artifacts']={n:study.sha((root/n).read_bytes()) for n in names+figures};_write(root/'audit.json',audit);return audit
