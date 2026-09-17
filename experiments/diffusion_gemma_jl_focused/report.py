"""Focused raw-only reporting, shared-QKV diagnostics, and independent audit."""
from collections import defaultdict
import csv
import gzip
import io
import json
from pathlib import Path
from unittest.mock import patch

from experiments.diffusion_gemma_jl_output_aware import report as common
from experiments.diffusion_gemma_jl_output_aware.runner import cached,load_output
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_jl_output_aware.shared_analysis import selected_sources
from experiments.diffusion_gemma_value_aware_followup import evidence
from experiments.diffusion_gemma_value_aware.report_metrics import accumulate_marginals,aggregate,SUM_FIELDS,correlation
from experiments.diffusion_gemma_value_aware_gpu.report import summarize,plots
from experiments.diffusion_gemma_value_aware.scientific_report import threshold_text
from experiments.diffusion_gemma_longbench_v2_100.report import pairs as nemo_pairs
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .protocol import ROOT,OLD,PROJECTED,BASELINES,TARGETS,prepare,execution,read,sha
from .reuse import audit_policy

csv_write=common.csv_write
table=common.table


def source(sources,path):
    sources[str(path)]=sha(Path(path).read_bytes())


def smoke_audit(root,setup,contract):
    folder=root/'validation'/contract['fingerprint'];p=folder/'smoke.json'
    proof=read(p);sources={str(p):sha(p.read_bytes())}
    if not proof['passed'] or proof['fingerprint']!=contract['fingerprint'] or len(proof['cases'])!=12:
        raise ValueError('Incomplete focused smoke')
    examples=[next(r for r in setup['calibration'] if r['benchmark']=='aime26'),
        max(setup['development'],key=lambda r:len(r['prompt_tokens']))]
    for original in examples:
        row=dict(original,generation_budget=16)
        native_path=folder/(sha(row['id'])+'.native.json');native=read(native_path);source(sources,native_path)
        for name,cfg in PROJECTED.items():
            for unpruned in (True,False):
                thresholds={k:dict(log_threshold=-1000.,unpruned=True,tau=0.) if unpruned else dict(log_threshold=-2.) for k in ('local','global')}
                label=f'{name}/{"unpruned" if unpruned else "pruned"}'
                out=cached(None,folder,row,'smoke',label,name,cfg,thresholds,contract)
                path=shard_path(folder,'smoke',label,row['id']);source(sources,path)
                sources[out['records_source']['path']]=out['records_source']['sha256']
                if not out['kernel_validation'] or not out['finite_calls']:
                    raise ValueError('Smoke lacks real finite CUDA validation')
                if unpruned:
                    if out['completion_tokens']!=native['completion_tokens'] or any(r['skipped'] for r in out['records']):
                        raise ValueError('Unpruned native parity failed')
                else:
                    ref=cached(None,folder,row,'reference',label,name,cfg,thresholds,contract)
                    path=shard_path(folder,'reference',label,row['id']);source(sources,path)
                    sources[ref['records_source']['path']]=ref['records_source']['sha256']
                    if out['completion_tokens']!=ref['completion_tokens']:
                        raise ValueError('Trusted pruned generation parity failed')
                    for kind in ('local','global'):
                        for key in ('eligible','skipped'):
                            if sum(r[key] for r in out['records'] if r['attention_type']==kind)!=sum(r[key] for r in ref['records'] if r['attention_type']==kind):
                                raise ValueError('Trusted physical counts mismatch')
                for key in evidence.DECODING_FIELDS:
                    if out['generation_metadata'].get(key)!=native['generation_metadata'].get(key):
                        raise ValueError('Smoke changed unrelated decoding settings')
    return sources


def findings(rows,compared):
    text=[]
    for b in ('aime26','longbench_v2'):
        group=[r for r in rows if r['benchmark']==b and r['split']=='full']
        dense=next((r for r in group if r['name']=='dense'),None)
        if dense is None:continue
        text.append(f"{b}: dense {dense['correct']:g}/{dense['count']}. ")
        for r in group:
            if r['name'] not in PROJECTED:continue
            lo,hi=r['paired_ci95']
            text.append(f"{r['condition']}: {r['correct']:g}/{r['count']}, actual sparsity {100*r['overall_physical_sparsity']:.1f}% "
                f"(global {100*r['global_physical_sparsity']:.1f}%, local {100*r['local_physical_sparsity']:.1f}%); "
                f"dense accuracy delta {100*r['delta']:+.1f} pp, paired 95% CI [{100*lo:+.1f}, {100*hi:+.1f}] pp.")
        unmatched=sum(not r['within3pp_all_types'] for r in compared if r['benchmark']==b and r['split']=='full')
        text.append(f'{unmatched} nearest-point comparisons are not within3 percentage points in all overall/global/local sparsity measures; do not infer a matched-budget winner from those comparisons.')
    return '\n\n'.join(text)


def write_report(root,setup,rows,tasks,compared,diag,audit,policies):
    display=[dict(benchmark=r['benchmark'],split=r['split'],condition=r['condition'],target=100*r['target'],
        threshold=threshold_text(r),score=f"{r['correct']:g}/{r['count']}",accuracy=100*r['accuracy'],delta_pp=100*r['delta'],
        whole=100*r['overall_physical_sparsity'],global_s=100*r['global_physical_sparsity'],local_s=100*r['local_physical_sparsity'],
        mass=100*r['overall_mass'],agreement=100*r['token_agreement'],trajectory_local_error=r['overall_relative_error'],
        unparsed=r['unparsed_answers'],length_limited=r['length_terminated']) for r in rows]
    csv_write(root/'main_results.csv',display)
    text='# Focused JL output-aware routing: AIME26 + LongBench v2\n\n'
    text+=f"Status: {'COMPLETE' if audit['complete'] else 'INCOMPLETE'}; {audit['completed']}/1040 audited final outputs. Three new routers, two targets; dense + original/aggressive BLASST + mass-only cached baselines.\n\n"
    text+='## Setup and selection\n\n'
    text+='All30 AIME26 problems,2048 output tokens;50 LongBench v2 questions selected deterministically from the completed newer100 NeMo study,4096 output tokens and32K context cap. Rows are copied verbatim, preserving tokenized prompts, seed42 and official scoring. NeMo is the pinned actual LongBench prompt/MCQ scorer, not a bespoke imitation; AIME retains the existing scorer and prompts. The old50-ID cohort is superseded.\n\n'
    text+=table([dict(domain=d,count=n) for d,n in setup['selection']['domains'].items()],('domain','count'))+'\n\n'
    text+='Selection: Hamilton domain then subtask quotas, SHA256(seed20260915,ID) ordering; no scores or predictions used. All final samples have previous exposure. AIME full30 overlaps the6 calibration problems; noncalibration24 is separate, not fresh held-out confirmation. All selected LongBench final IDs/hashes are disjoint from both studies’ calibration/development sets. See selection.json, dataset_audit.json and the verbatim manifests.\n\n'
    text+='Pinned BF16 DiffusionGemma revision '+setup['revision']+';128-query×64-KV physical tiles; prefix+canvas eligible, native masks and GQA. Canvas256/up to48 denoising steps, thinkingFalse, unchanged native0.4–0.8 schedule (temperature0 sentinel is not greedy). The same output budgets/settings/seeds are used for every method. Dense outputs are imported once, never rerun for final comparison.\n\n'
    text+='## Algorithms and calibration\n\n'
    text+='Gaussian centered: online candidate attention mass fraction times the difference between attention-weighted projected block mean and retained running projected output. Uncentered control: the same fraction times projected block contribution magnitude. Both use fixed Gaussian rank32, seed1729, per-layer/native-KV-head projection hashes, FP32 routing arithmetic. Full centered uses identity/full-dimensional values and is an expensive reference, not a deployable winner. All use valid-KV RMS normalization, strict worst-valid-query physical gate, retain ties/first support, and leave state unchanged on skips. Original V computes ordinary renormalized retained attention, without compensation. No new type-aware guard or additional projection families/ranks are included.\n\n'
    text+='The unchanged BLASST implementation uses exact token QK and per-row KV-block maxima; a physical tile skips only when all valid rows vote skip. Original BLASST caps lambda at1; aggressive BLASST allows lambda>1 using the preceding-seen-maximum rule. Source inverse-valid-KV-length thresholds are lambda=exp(log_scale)/L with the recorded cap; scalar thresholds are exp(log_threshold). Verified unattainable original types use constant lambda1. Local/global policies remain distinct and shared across layers/heads/steps.\n\n'
    text+='Cached AIME baseline policies are unchanged. LongBench baseline policies retain the newer100 study’s12 calibration questions spanning6 domains. Centered JL32/full-reference policies reuse the completed older JL6-per-benchmark512-token calibration; its LongBench calibration covers3 domains. Only uncentered rank32 is newly calibrated, on those same6-per-benchmark512-token runs with the existing scalar empirical-CDF/rank refinement, at most3 verified joint points. Shared proposals reuse540 calibration QKV snapshots. This asymmetric calibration coverage is a limitation; actual sparsity transfer is measured, never repaired using final scores. thresholds.json includes exact source policies and attainable-boundary annotations.\n\n'
    text+='## Main results\n\n'+findings(rows,compared)+'\n\n'
    cols=('condition','target','threshold','score','accuracy','delta_pp','whole','global_s','local_s','mass','agreement','trajectory_local_error','unparsed','length_limited')
    for b,split in (('aime26','full'),('aime26','noncalibration24'),('longbench_v2','full')):
        text+=f'### {b}: {split}\n\n'+table([r for r in display if r['benchmark']==b and r['split']==split],cols)+'\n\n'
    text+='### LongBench v2 domain results\n\n'+table([dict(domain=r['task'],condition=r['condition'],score=f"{r['correct']:g}/{r['count']}",accuracy=100*r['accuracy']) for r in tasks if r['benchmark']=='longbench_v2'],('domain','condition','score','accuracy'))+'\n\n'
    text+='## Measurements, uncertainty, and limitations\n\n'
    text+='Physical sparsity is SUM(skipped eligible tiles)/SUM(eligible tiles), not an average of sample/layer percentages. Decoder denoising attention only; dense prefill is excluded. Region counts separate prefix, canvas and boundary tiles. Main retained mass and local error compare dense/sparse attention on each run’s corresponding QKV; they are not a replay of cached dense-generation states after token divergence. Shared-QKV diagnostics separately compare every method on identical saved dense calibration states, with error=sqrt(sum error²/sum dense-output²). Sampling is one query head/query tile per selected layer/step state, not exhaustive; shared_operator.csv records coverage, full-dimensional risk underestimation and overlapping type statistics by layer/head/step/attention type/region. Full PV is diagnostic-only for projected routing.\n\n'
    text+='Token agreement compares every generated position, including after the first divergence; missing/extra positions disagree and EOS is included. Accuracy is micro, with domain macro also exported. Paired prompt-bootstrap95% intervals are exploratory, not multiplicity-corrected. Comparisons select the nearest observed actual-sparsity baseline and flag unmatched global/local budgets; no target-based interpolation or unsupported winner. Correlations across dependent configurations are descriptive, not causal. No Gaussian/sign or rank8/16 comparison and no end-to-end seed-sensitivity conclusion is claimed in this user-narrowed study.\n\n'
    text+='projection_seed_sensitivity.json/csv reuse the prior JL32 calibration-only shared-state checks for the fixed seeds1729/2718/31415 at unchanged thresholds. They compare worst-row projected/full-dimensional risks with all previous blocks retained; this is a support-controlled distortion check, not final-generation sparsity or accuracy evidence. No seed is selected from these results.\n\n'
    text+='QK, block softmax and projected PV remain computed. Sketch caching refreshes changed canvas/boundary values; work_accounting.json records projected multiply-adds, reuse/refresh and storage. The retained native BF16 PV is still a dense-shaped masked matmul in this reference evaluation. Physical deletion is not proof of hardware-compute avoidance; runtime engineering is not a FlashAttention/system-speedup claim. Full-dimensional error may correlate differently with downstream accuracy; report both.\n\n'
    text+='## Failures and reproducibility\n\n'
    text+=f"Missing entries: {len(audit['missing'])}; violations: {len(audit['violations'])}. See audit.json and preserved failures.jsonl. Superseded jobs were canceled by explicit user approval, not excluded for scientific failure; their artifacts remain unchanged. No method is eliminated based on a tiny accuracy screen.\n\n"
    text+='Regenerate without inference: `CUDA_VISIBLE_DEVICES=\'\' PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_focused.workflow report`; independently repeat with `verify`. Final tables use completed raw final shards only; incomplete conditions never appear as complete scores.\n'
    (root/'report.md').write_text(text)


def regenerate(root=ROOT):
    setup,contract=prepare(root),execution(root)
    conditions={};missing=[];violations=[];sources=dict(contract['sources'])
    for label in setup['conditions']:
        path=root/'final_configs'/f'{label}.json'
        if not path.exists():missing.append(dict(condition=label,error='Final contract not frozen'));continue
        c=read(path)
        if c['fingerprint']!=contract['fingerprint']:raise ValueError('Final execution fingerprint mismatch')
        evidence.check_sources(c['sources']);sources.update(c['sources']);source(sources,path);conditions[label]=c
    policies=[]
    for path in sorted((root/'policies').glob('*/*.json')):
        p=read(path);audit_policy(root,p,setup,contract);policies.append(p);source(sources,path)
    try:sources.update(smoke_audit(root,setup,contract))
    except Exception as error:violations.append(dict(stage='smoke',error=str(error)))
    raw=[];dimensions={};matrices={};bank=Projections();work=[];grades=[]
    temp=root/'per_layer_head_step.csv.gz.tmp'
    with temp.open('wb') as binary,gzip.GzipFile(filename='',fileobj=binary,mode='wb',compresslevel=3,mtime=0) as gz:
        with io.TextIOWrapper(gz,encoding='utf-8',newline='') as stream:
            writer=None
            for row in setup['final']:
                outputs={}
                for label,c in conditions.items():
                    stage='dense' if label=='dense' else 'final'
                    try:
                        out=cached(None,root,row,stage,label,c['name'],c['config'],c['thresholds'][row['benchmark']],contract)
                        with patch.object(common,'PROJECTED',PROJECTED):common.audit_matrices(out,c['name'],c['config'],bank,matrices)
                        source(sources,shard_path(root,stage,label,row['id']))
                        sources[out['records_source']['path']]=out['records_source']['sha256']
                        if out.get('imported_source'):
                            x=out['imported_source'];sources[x['path']]=x['sha256']
                        outputs[label]=out
                    except FileNotFoundError as error:missing.append(dict(id=row['id'],condition=label,error=str(error)))
                    except Exception as error:violations.append(dict(id=row['id'],condition=label,error=str(error)))
                if 'dense' not in outputs:continue
                try:
                    if row['benchmark']=='longbench_v2':
                        paired,g=nemo_pairs(row,outputs);grades.extend(g)
                    else:paired=[dict(condition=n,**evidence.pair(row,o,outputs['dense'])) for n,o in outputs.items()]
                except Exception as error:violations.append(dict(id=row['id'],stage='pair',error=str(error)));continue
                for p in paired:
                    label=p['condition'];out=outputs[label]
                    raw.append(dict(**p,sub_domain=row.get('sub_domain'),difficulty=row.get('difficulty'),
                        input_tokens=len(row['prompt_tokens']),truncated=row.get('truncated',False),
                        prediction=out['prediction'],completion_tokens=out['completion_tokens'],backend=out['backend']))
                    if out.get('work_accounting'):work.append(dict(id=row['id'],condition=label,**out['work_accounting']))
                    records=[r for r in out['records'] if r['probe']=='execution']
                    accumulate_marginals(dimensions,records,row['benchmark'],label)
                    for r in records:
                        item=dict(id=row['id'],benchmark=row['benchmark'],condition=label,**r)
                        if writer is None:
                            writer=csv.DictWriter(stream,fieldnames=list(item));writer.writeheader()
                        writer.writerow(item)
    temp.replace(root/'per_layer_head_step.csv.gz')
    diag=[]
    try:
        diag,ds=common.diagnostic_summary(root);sources.update(ds)
        index=read(root/'shared_diagnostics_index.json')
        expected=len(selected_sources(read(root/'shared_state_index.json')))*len(TARGETS)*(len(PROJECTED)+len(BASELINES))
        if len(index)!=expected:raise ValueError(f'Shared diagnostic coverage {len(index)}!={expected}')
    except Exception as error:violations.append(dict(stage='shared_diagnostics',error=str(error)))
    with patch.object(common,'PROJECTED',PROJECTED):
        seeds,seed_sources=common.shared_seed_summary(OLD)
    sources.update(seed_sources)
    groups=defaultdict(list);taskgroups=defaultdict(list);subgroups=defaultdict(list)
    for r in raw:
        groups[r['benchmark'],'full',r['condition']].append(r)
        if r['benchmark']=='aime26':groups['aime26','calibration6' if r['calibration'] else 'noncalibration24',r['condition']].append(r)
        taskgroups[r['benchmark'],r['task'],r['condition']].append(r)
        if r['sub_domain']:subgroups[r['benchmark'],r['sub_domain'],r['condition']].append(r)
    rows=[]
    for (b,split,label),group in sorted(groups.items()):
        required={'full':30 if b=='aime26' else 50,'calibration6':6,'noncalibration24':24}[split]
        if len(group)!=required:continue
        c=conditions[label]
        rows.append(dict(benchmark=b,split=split,condition=label,name=c['name'],target=c['target'],
            config=c['config'],method=c['config'].get('method','centered' if c['name'] in PROJECTED else 'dense'),
            thresholds=c['thresholds'][b],**summarize(group)))
    tasks=[dict(benchmark=b,task=t,condition=n,**summarize(g)) for (b,t,n),g in sorted(taskgroups.items())]
    subtasks=[dict(benchmark=b,sub_domain=t,condition=n,**summarize(g)) for (b,t,n),g in sorted(subgroups.items())]
    with patch.object(common,'PROJECTED',PROJECTED):compared=common.compare(rows,groups)
    correlations=[dict(benchmark=b,metric=key,pearson_with_accuracy=correlation(
        [r[key] for r in rows if r['benchmark']==b and r['split']=='full' and r['name']!='dense'],
        [r['accuracy'] for r in rows if r['benchmark']==b and r['split']=='full' and r['name']!='dense']))
        for b in ('aime26','longbench_v2') for key in ('overall_mass','token_agreement','overall_relative_error')]
    audited=dict(complete=len(raw)==1040 and len(conditions)==13 and not missing and not violations,
        completed=len(raw),expected=1040,missing=missing,violations=violations,sources=sources,fingerprint=contract['fingerprint'])
    for name,data in (('per_sample',raw),('summary',rows),('thresholds',policies),('comparisons',compared),
            ('shared_operator',diag),('work_accounting',work),('projection_matrices',matrices),('nemo_grades',grades),
            ('correlations',correlations),('projection_seed_sensitivity',seeds)):
        _write(root/f'{name}.json',data)
        if isinstance(data,list):csv_write(root/f'{name}.csv',data)
    csv_write(root/'per_task.csv',tasks);csv_write(root/'per_subtask.csv',subtasks)
    csv_write(root/'routing_marginals.csv',[dict(benchmark=b,condition=n,axis=a,index=i,attention_type=k,
        **aggregate([dict(zip(SUM_FIELDS,v))])) for (b,n,a,i,k),v in sorted(dimensions.items())])
    write_report(root,setup,rows,tasks,compared,diag,audited,policies)
    images=plots(root,rows) if rows else []
    names=['per_layer_head_step.csv.gz','per_task.csv','per_subtask.csv','routing_marginals.csv','main_results.csv','report.md','projection_matrices.json']
    names += [f'{n}.{ext}' for n in ('per_sample','summary','thresholds','comparisons','shared_operator','work_accounting','nemo_grades','correlations','projection_seed_sensitivity') for ext in ('json','csv')]
    audited['artifacts']={n:sha((root/n).read_bytes()) for n in names+images}
    _write(root/'audit.json',audited)
    return audited


def verify(root=ROOT):
    before=read(root/'audit.json')
    if not before['complete'] or before['completed']!=1040:raise ValueError('Require complete1040-output audit')
    evidence.check_sources({str(root/p):v for p,v in before['artifacts'].items()})
    after=regenerate(root)
    if before!=after:raise ValueError('Raw-only regeneration changed audit or artifacts')
    result=dict(passed=True,completed=1040,inference_performed=False,identical_audit_and_artifact_hashes=True,
        artifact_count=len(after['artifacts']),audit_sha256=sha((root/'audit.json').read_bytes()))
    _write(root/'regeneration_verification.json',result)
    return result
