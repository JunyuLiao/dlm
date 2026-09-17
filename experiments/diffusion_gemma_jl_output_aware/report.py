"""Raw-only paired evaluation; every complete candidate and all failures remain visible."""
from collections import defaultdict
import ast
import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np

from experiments.diffusion_gemma_value_aware.report import csv_write as parent_csv_write, table
from experiments.diffusion_gemma_value_aware.report_metrics import accumulate_marginals, aggregate, SUM_FIELDS
from experiments.diffusion_gemma_value_aware_gpu.report import summarize
from experiments.diffusion_gemma_value_aware.scientific_report import threshold_text
from experiments.diffusion_gemma_value_aware_followup import evidence
from experiments.diffusion_gemma_longbench_v2_100.report import pairs as nemo_pairs
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .protocol import ROOT, prepare, execution, sha
from .config import Config, PROJECTED, METHODS, TARGETS
from .runner import cached, load_output
from .calibration import audit_policy
from .projections import Projections


def csv_write(path, rows):
    # Parent helper omits empty exports. An incomplete audit must still be
    # regenerable without stale CSVs left over from an earlier report.
    if rows:
        parent_csv_write(path, rows)
    else:
        path.write_text('')


def audit_matrices(out, name, config, bank, merged):
    if name not in PROJECTED: return
    c = Config(**config)
    matrices = out['projection_matrices']
    if c.method != 'mass_exact':
        if {r['layer'] for records in matrices.values() for r in records} != set(range(30)):
            raise ValueError('Projection matrix layer coverage incomplete')
    for key, records in matrices.items():
        layer, heads, width, family, rank, seed = ast.literal_eval(key)
        allowed_seeds = (c.projection_seed, c.guard_seed) if c.method == 'cancellation_guard' else (c.projection_seed,)
        if family != c.family or seed not in allowed_seeds or rank != (width if family=='identity' else c.rank):
            raise ValueError('Projection seed/family/rank differs from frozen candidate')
        if key not in merged:
            bank.get(layer, heads, width, family, rank, seed, 'cpu')
            if records != bank.manifest[key]:
                raise ValueError('Projection hashes do not reproduce from frozen seeds')
            merged[key] = records
        elif merged[key] != records:
            raise ValueError('Projection matrices changed across final examples')


def compare(rows, groups):
    results=[]
    for row in rows:
        if row['name'] not in PROJECTED:
            continue
        for baseline in ('blasst_original','blasst_aggressive','mass','mass_exact','full_centered'):
            if row['name']==baseline:
                continue
            pool=[r for r in rows if (r['benchmark'],r['split'],r['name'])==(row['benchmark'],row['split'],baseline)]
            if not pool:
                continue
            ref=min(pool,key=lambda r:abs(r['overall_physical_sparsity']-row['overall_physical_sparsity']))
            a={r['id']:r['accuracy'] for r in groups[row['benchmark'],row['split'],row['condition']]}
            b={r['id']:r['accuracy'] for r in groups[ref['benchmark'],ref['split'],ref['condition']]}
            if set(a)!=set(b):
                raise ValueError('Unpaired actual-sparsity comparison')
            gaps={k:row[f'{k}_physical_sparsity']-ref[f'{k}_physical_sparsity'] for k in ('overall','global','local')}
            results.append(dict(benchmark=row['benchmark'],split=row['split'],candidate=row['condition'],
                reference=ref['condition'],accuracy_delta=row['accuracy']-ref['accuracy'],
                paired_ci95=paired_bootstrap_ci([a[i]-b[i] for i in sorted(a)]),
                **{f'{k}_sparsity_gap':v for k,v in gaps.items()},
                within3pp_overall=abs(gaps['overall'])<=.03,within3pp_all_types=all(abs(v)<=.03 for v in gaps.values()),
                interpretation='Nearest observed actual-sparsity point; no interpolation; exploratory paired CI, not multiplicity-corrected'))
    return results


def plots(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures';folder.mkdir(exist_ok=True)
    result=[]
    for benchmark in ('aime26','longbench_v2'):
        for family in ('primary','controls'):
            group=[r for r in rows if r['benchmark']==benchmark and r['split']=='full']
            if family=='primary':
                group=[r for r in group if r['name'].startswith('jl_') or r['name'] in ('dense','blasst_original','blasst_aggressive','mass','risk')]
            else:
                group=[r for r in group if not r['name'].startswith('jl_') or r['name'] in ('jl_gaussian_r16','jl_sign_r16')]
            fig,axes=plt.subplots(1,3,figsize=(16,4.5))
            for method in sorted({r['name'] for r in group}):
                points=sorted((r for r in group if r['name']==method),key=lambda r:r['target'])
                for ax,key,label in zip(axes,('accuracy','overall_mass','token_agreement'),
                        ('Accuracy (%)','Retained dense mass (%)','Positional token agreement (%)')):
                    ax.plot([100*r['overall_physical_sparsity'] for r in points],[100*r[key] for r in points],
                        marker='o',label=method,linewidth=1)
                    ax.set(xlabel='Actual physical tile sparsity (%)',ylabel=label,xlim=(-2,100))
                    ax.grid(alpha=.2)
            axes[-1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.01,1))
            fig.suptitle(f'{benchmark}: {family}; complete conditions only; previously examined samples')
            fig.tight_layout()
            path=folder/f'{benchmark}_{family}.png';fig.savefig(path,dpi=150);plt.close(fig)
            result.append(str(path.relative_to(root)))
    return result


def diagnostic_summary(root):
    index_path=root/'shared_diagnostics_index.json'
    if not index_path.exists():
        return [],{}
    groups=defaultdict(list);sources={str(index_path):sha(index_path.read_bytes())}
    for source in json.loads(index_path.read_text()):
        path=Path(source['path'])
        if sha(path.read_bytes())!=source['sha256']:
            raise ValueError('Shared-state diagnostic result changed')
        d=json.loads(path.read_text());identity=d['identity']
        evidence.check_sources(identity['diagnostic_sources'])
        sources[str(path)]=source['sha256'];sources.update(identity['diagnostic_sources'])
        source_state=identity['source'];p=Path(source_state['path'])
        if sha(p.read_bytes())!=source_state['sha256']:
            raise ValueError('Shared QKV snapshot changed')
        sources[str(p)]=source_state['sha256']
        p=Path(identity['policy_path'])
        if sha(p.read_bytes()) != identity['policy_sha256']:
            raise ValueError('Diagnostic threshold provenance changed')
        sources[str(p)]=identity['policy_sha256']
        for axis,index in (('overall','all'),('attention_type',source_state['attention_type']),
                ('layer',str(source_state['layer'])),('head',str(source_state['head'])),('step',str(source_state['step']))):
            groups[source_state['id'].split('/')[0],identity['name'],identity['target'],axis,index].append(d['metrics'])
    output=[]
    for (benchmark,name,target,axis,index),rows in sorted(groups.items()):
        total=lambda key:sum(r.get(key,0) or 0 for r in rows)
        types={t:{k:sum(r['types'][t][k] for r in rows) for k in rows[0]['types'][t]} for t in rows[0]['types']}
        regions={t:{k:sum(r['regions'][t][k] for r in rows) for k in rows[0]['regions'][t]} for t in rows[0]['regions']}
        output.append(dict(benchmark=benchmark,name=name,target=target,axis=axis,index=index,sampled_states=len(rows),
            eligible_tiles=total('eligible_tiles'),skipped_tiles=total('skipped_tiles'),
            physical_sparsity=total('skipped_tiles')/max(1,total('eligible_tiles')),
            valid_row_blocks=total('valid_row_blocks'),dangerous_underestimates=total('dangerous_underestimates'),
            severe_underestimates=total('severe_underestimates'),
            matched_full_threshold_disagreement=total('matched_full_threshold_disagreement'),
            mixed_query_tiles=total('mixed_query_tiles'),
            zero_true_update_rows=total('zero_true_update_rows'),
            per_state_spearman=[r['spearman_risk'] for r in rows],
            per_state_norm_ratio_quantiles=[r['norm_ratio_quantiles'] for r in rows],
            relative_output_error=float(np.sqrt(total('error_sq')/max(total('dense_sq'),1e-30))),
            retained_mass=total('retained_mass_sum')/max(1,total('valid_query_rows')),
            error_sq=total('error_sq'),dense_sq=total('dense_sq'),types=types,regions=regions))
    return output,sources


def seed_summary(root,setup,contract,conditions):
    path=root/'development_policy.json'
    if not path.exists():
        return [],{}
    policy=json.loads(path.read_text());ids=set(policy['seed_check_ids'])
    examples={r['id']:r for r in setup['calibration']+setup['development'] if r['id'] in ids}
    rows=[];sources={str(path):sha(path.read_bytes())}
    for label,condition in conditions.items():
        if not condition['name'].startswith('jl_'):
            continue
        for seed in (1729,*policy['alternate_projection_seeds']):
            tag=label if seed==1729 else label+f'/seed{seed}'
            group=defaultdict(list)
            for row in examples.values():
                source=shard_path(root,'development',tag,row['id'])
                if not source.exists():
                    continue
                out=load_output(source)
                config=dict(condition['config'])
                if seed!=1729:config['projection_seed']=seed
                evidence.check_result(out,row,contract['fingerprint'],config,condition['thresholds'][row['benchmark']])
                sources[str(source)]=sha(source.read_bytes())
                sources[out['records_source']['path']]=out['records_source']['sha256']
                group[row['benchmark']].append(out)
            for benchmark,outputs in group.items():
                measures=aggregate([r for o in outputs for r in o['records']])
                rows.append(dict(benchmark=benchmark,condition=label,projection_seed=seed,count=len(outputs),
                    correct=sum(o['score'] for o in outputs),accuracy=float(np.mean([o['score'] for o in outputs])),
                    **measures,scope='Prespecified small development sensitivity set; primary thresholds unchanged; no seed selected'))
    return rows,sources


def shared_seed_summary(root):
    """Same dense-prefix support for all three prespecified projection seeds."""
    path=root/'shared_screen_index.json'
    if not path.exists(): return [],{}
    groups=defaultdict(lambda:dict(tiles=0, supported=0, votes=0, dangerous=0, severe=0, ratios=[]))
    sources={str(path):sha(path.read_bytes())}
    for item in json.loads(path.read_text()):
        p=Path(item['path'])
        if sha(p.read_bytes())!=item['arrays_sha256']:raise ValueError('Shared projection-seed arrays changed')
        sources[str(p)]=item['arrays_sha256']
        if item['identity']['source']['split']!='calibration':raise ValueError('Shared seed screen contamination')
        with np.load(p) as data:
            truth=data['full_centered'].astype(float)
            for name in PROJECTED:
                if not name.startswith('jl_'):continue
                for target in TARGETS:
                    policy_path=root/'policies'/item['benchmark']/f'{name}_s{int(target*100)}.json'
                    if not policy_path.exists():continue
                    threshold=json.loads(policy_path.read_text())['policy'][item['attention_type']]['log_threshold']
                    sources[str(policy_path)]=sha(policy_path.read_bytes())
                    for seed in (1729,2718,31415):
                        arr=data[name if seed==1729 else name+f'__seed{seed}'].astype(float)
                        support=~np.isposinf(truth)
                        ratios=np.exp(np.clip(arr[np.isfinite(truth)&np.isfinite(arr)]-truth[np.isfinite(truth)&np.isfinite(arr)],-80,80))[::16]
                        for kind in ('overall',item['attention_type']):
                            g=groups[item['benchmark'],name,target,seed,kind]
                            g['tiles']+=len(arr);g['supported']+=int(support.sum())
                            g['votes']+=int((arr<threshold).sum())
                            g['dangerous']+=int(((arr<threshold)&(truth>=threshold)&support).sum())
                            g['severe']+=int(((arr<truth-np.log(2))&(truth>=np.log(.05))&support).sum())
                            g['ratios'].extend(ratios.tolist())
    rows=[]
    for (benchmark,name,target,seed,kind),g in sorted(groups.items()):
        ratios=g.pop('ratios')
        rows.append(dict(benchmark=benchmark,name=name,target=target,seed=seed,attention_type=kind,**g,
            dense_support_skip_vote_fraction=g['votes']/max(1,g['tiles']),
            sampled_worst_risk_ratio_quantiles=np.quantile(ratios,[.01,.1,.5,.9,.99]).tolist() if ratios else [],
            scope='All previous blocks retained; tile-max projected/full risk; calibration-only sensitivity, not deployed sparse trajectory or final sparsity'))
    return rows,sources


def supporting_audit(root, setup, contract, conditions):
    """Completion includes promised checks, not just final output count."""
    missing, violations, sources = [], [], {}
    for name in ('smoke', 'observer'):
        path = root/'validation'/contract['fingerprint']/f'{name}.json'
        if not path.exists():
            missing.append(dict(stage='validation', proof=name)); continue
        proof = json.loads(path.read_text())
        if not proof.get('passed') or proof['fingerprint'] != contract['fingerprint']:
            violations.append(dict(stage='validation', proof=name))
        sources[str(path)] = sha(path.read_bytes())
        evidence.check_sources(proof['sources']); sources.update(proof['sources'])
        required = 4*len(PROJECTED) if name == 'smoke' else 2
        if len(proof['cases']) != required:
            missing.append(dict(stage='validation', proof=name, expected=required))
    path = root/'development_policy.json'
    if not path.exists():
        missing.append(dict(stage='development', error='Missing development policy'))
    else:
        p = json.loads(path.read_text()); sources[str(path)] = sha(path.read_bytes())
        examples = {r['id']:r for r in setup['calibration']+setup['development']}
        checks = [('dense', 'dense', {}, None, i) for i in p['seed_check_ids']]
        for label, c in conditions.items():
            if label == 'dense': continue
            ids = p['seed_check_ids'] if c['name'].startswith('jl_') else p['other_control_check_ids']
            seeds = (None, *p['alternate_projection_seeds']) if c['name'].startswith('jl_') else (None,)
            for seed in seeds:
                tag = label if seed is None else label+f'/seed{seed}'
                candidate = c['name'] if seed is None else 'seedcheck/'+c['name']+f'/seed{seed}'
                config = c['config'] if seed is None else dict(c['config'], projection_seed=seed)
                checks.extend((tag, candidate, config, c['thresholds'], i) for i in ids)
        for tag, candidate, cfg, thresholds, ident in checks:
            try:
                row = examples[ident]
                policy = thresholds[row['benchmark']] if thresholds is not None else None
                out = cached(None, root, row, 'development', tag, candidate, cfg, policy, contract)
                file = shard_path(root, 'development', tag, ident)
                sources[str(file)] = sha(file.read_bytes())
                sources[out['records_source']['path']] = out['records_source']['sha256']
            except FileNotFoundError:
                missing.append(dict(stage='development', condition=tag, id=ident))
            except Exception as error:
                violations.append(dict(stage='development', condition=tag, id=ident, error=str(error)))
    state_index, index = root/'shared_state_index.json', root/'shared_diagnostics_index.json'
    if not state_index.exists() or not index.exists():
        missing.append(dict(stage='shared_diagnostics', error='Missing shared state/diagnostic index'))
    else:
        from .shared_analysis import selected_sources
        selected = selected_sources(json.loads(state_index.read_text()))
        expected = {(s['path'], n, t) for s in selected.values() for n in METHODS for t in TARGETS}
        actual = []
        for source in json.loads(index.read_text()):
            file = Path(source['path']); raw = file.read_bytes()
            if sha(raw) != source['sha256']: raise ValueError('Diagnostic shard changed')
            ident = json.loads(raw)['identity']
            actual.append((ident['source']['path'], ident['name'], ident['target']))
        if len(actual) != len(set(actual)):
            violations.append(dict(stage='shared_diagnostics', error='Duplicate diagnostic identities'))
        if set(actual) != expected:
            missing.append(dict(stage='shared_diagnostics', expected=len(expected), completed=len(set(actual)),
                missing_count=len(expected-set(actual)), unexpected_count=len(set(actual)-expected)))
        for file in (state_index, index): sources[str(file)] = sha(file.read_bytes())
    return missing, violations, sources


def write_report(root,setup,rows,compared,policies,audit,figures,diagnostics,seeds):
    text='# JL-projected output-aware block skipping\n\n'
    text+=f"Raw final outputs: {audit['completed']}/{audit['expected']}. Complete audit: {audit['complete']}. All requested candidates and failures remain visible; no method or projection seed was selected by final accuracy.\n\n"
    text+='## Setup and limitations\n\n'
    text+='DiffusionGemma BF16, native256-token canvas/48-step limit and0.4–0.8 sampling schedule (temperature0 is a sentinel, not greedy);128×64 physical tiles, prefix+canvas eligible. Same30 AIME26 and50 LongBench v2 IDs as the historical study. AIME uses unchanged2048-token prompts/budgets/scoring. LongBench uses the actual pinned NeMo-Skills prompt/scorer and4096 output tokens. The three domains contain17/17/16 questions;25/50 contexts are truncated at32K. All80 final questions were previously examined; AIME30 includes six calibration questions. The noncalibration24 are reported separately, not called fresh confirmation.\n\n'
    text+='Primary projection seed1729 is fixed before evaluation; Gaussian/sign and ranks8/16/32 are all evaluated. Seeds2718/31415 are prespecified shared-state and small-development sensitivity checks, never used to choose a primary seed. Every layer/native KV head has a recorded matrix hash. FP32 token sketches are refreshed after underlying value/validity changes, including canvas/boundary changes. GQA expands only after native-head sketching.\n\n'
    text+='## Operators and calibration\n\n'
    text+='The primary score is exact online block mass fraction times the norm of the difference between the attention-weighted projected block mean and the retained running projected output, divided by valid-KV RMS value scale. Strict all-query-row thresholding retains ties and first support. Skipped blocks do not update retained state. This is immediate addition impact, not final deletion error or cumulative-error control. Retained output is ordinary native attention using original V, without compensation. Full block PV does not supply sketch routing; it is confined to diagnostics and explicit full-dimensional controls.\n\n'
    text+='Controls isolate exact mass, uncentered projected contribution (both families at rank16), full-dimensional weighted centering and the historical unweighted-mean-plus-radius heuristic. The two cancellation guards use an independent second rank16 sketch when κ<0.25 and online α≥0.1, taking the larger estimated centered update. The ordinary centered variants remain in the comparison. These guards are heuristics, not JL guarantees.\n\n'
    text+='Original BLASST uses the repository physical all-valid-query-row gate and an all-previous-block maximum; λ≤1. Aggressive BLASST permits λ>1 and is separately labeled. Both reuse inverse-valid-KV-length λ adjustment and local/global calibration. Historical compatible AIME baseline policies/results are reused with hashes. New policies use the same calibration question IDs with512-token calibration rollouts, empirical-CDF scalar refinement or the existing inverse-length BLASST search, at most three verified joint points. Final budgets are longer; actual sparsity drift is not retuned away.\n\n'
    threshold_rows=[]
    for p in policies:
        threshold_rows.append(dict(benchmark=p['benchmark'],method=p['name'],requested_target=p['target'],
            thresholds=threshold_text({'thresholds':p['policy']}),calibration_actual=p.get('measured'),lambda1_unattainable=p.get('cap_one_unattainable'),
            calibration_budget=p.get('calibration_budget')))
    text+=table(threshold_rows,['benchmark','method','requested_target','thresholds','calibration_actual','lambda1_unattainable','calibration_budget'])+'\n\n'
    text+='## Complete final comparisons\n\n'
    for benchmark in ('aime26','longbench_v2'):
        text+=f'### {benchmark}\n\n'
        display=[]
        for row in rows:
            if row['benchmark']!=benchmark or row['split']!='full':continue
            display.append(dict(condition=row['condition'],target_pct=100*row['target'],
                correct=f"{row['correct']:g}/{row['count']}",dense_delta_pp=100*row['delta'],
                paired_ci95_pp=[round(100*v,2) for v in row['paired_ci95']],
                actual_pct=100*row['overall_physical_sparsity'],global_pct=100*row['global_physical_sparsity'],
                local_pct=100*row['local_physical_sparsity'],mass_pct=100*row['overall_mass'],
                agreement_pct=100*row['token_agreement'],output_error=row['overall_relative_error'],
                unparsed=row['unparsed_answers'] if benchmark=='longbench_v2' else None,length_limited=row['length_terminated']))
        text+=table(display,['condition','target_pct','correct','dense_delta_pp','paired_ci95_pp','actual_pct','global_pct','local_pct','mass_pct','agreement_pct','output_error','unparsed','length_limited'])+'\n\n'
    text+='Per-domain and AIME noncalibration24 results are in `per_task.csv` and `summary.csv`. `comparisons.csv` selects nearest measured baseline points and exposes local/global budget gaps; a requested target is never substituted for achieved sparsity. Intervals are exploratory paired prompt bootstraps, not multiplicity-corrected superiority tests.\n\n'
    text+='## Diagnostics, seed sensitivity and work accounting\n\n'
    text+='`shared_diagnostics_summary.json` reports full-dimensional output error on shared QKV separately from generation divergence. Full-risk replays follow each candidate’s retained support, rather than comparing different prefixes. Sampling is one physical query tile/head per selected state; all layer/type/head/step labels and exact sources are retained. Type frequencies overlap and mixed-query tiles are reported. Squared local updates are not additive final-output-error attribution. Projected cancellation is checked against full-dimensional classification.\n\n'
    text+='`projection_seed_sensitivity.csv` reports all prespecified development seed checks with the primary thresholds unchanged. Small development scores do not establish seed-invariant downstream behavior. `work_accounting.json` records projection multiply-adds, persistent sketch/cache-copy storage, refresh/reuse and validation work. Counts are not hardware-speedup measurements. QK, block softmax and projected PV are still computed. The final native PV is a dense-shaped masked matmul; physical deletion does not imply that this arithmetic was elided. Persistent-cache storage is not total peak GPU memory.\n\n'
    text+='Sparsity is the ratio of summed skipped/eligible tiles, separately overall/global/local; dense prefill is outside the denominator. Retained mass averages exact dense probabilities over valid query rows at the corresponding sparse execution state. Token agreement compares every positional output token, including after divergence; missing/extra positions disagree. Full-sequence exact match is exported.\n\n'
    text+='## Failures and interpretation scope\n\n'
    text+=f"Missing outputs/policies: {len(audit['missing'])}; audit violations: {len(audit['violations'])}; recorded failed attempts: {audit['failed_attempts']}. Incomplete groups are omitted from accuracy curves, not presented as completed scores. See `audit.json` and `failures.jsonl` when present. No unsupported winner is selected from a small screen.\n\n"
    text+='The empirical questions are addressed by the complete primary/ablation tables, matched-actual comparisons, type diagnostics and seed checks together: operator-error improvements alone are not downstream wins; matched overall sparsity alone does not establish matched local/global allocation. A method needs a favorable paired accuracy comparison at comparable actual budgets before claiming an advantage, with this reused-sample and single-generation-seed scope kept explicit.\n\n'
    for figure in figures:text+=f'![{Path(figure).stem}]({figure})\n\n'
    text+="Regenerate from completed immutable shards without inference: `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_output_aware.workflow report`.\n"
    (root/'report.md').write_text(text)


def regenerate(root=ROOT):
    setup,contract=prepare(root),execution(root)
    conditions={};sources={};missing=[];violations=[];policies=[]
    for label in setup['conditions']:
        path=root/'final_configs'/f'{label}.json'
        if not path.exists():missing.append(dict(condition=label,error='Missing frozen condition'));continue
        c=json.loads(path.read_text())
        if c['fingerprint']!=contract['fingerprint']:raise ValueError('Condition fingerprint mismatch')
        evidence.check_sources(c['sources']);sources.update(c['sources']);sources[str(path)]=sha(path.read_bytes())
        for benchmark in ('aime26','longbench_v2'):
            if c['name']=='dense':continue
            p=root/'policies'/benchmark/f'{label}.json'
            if not p.exists():missing.append(dict(condition=label,benchmark=benchmark,error='Missing policy'));continue
            policy=json.loads(p.read_text());audit_policy(root,policy,setup,contract);policies.append(policy)
        conditions[label]=c
    raw=[];graded=[];dimensions={};work=[];completed=0
    bank,matrices=Projections(),{}
    temp=root/'per_layer_head_step.csv.gz.tmp'
    with temp.open('wb') as binary,gzip.GzipFile(filename='',fileobj=binary,mode='wb',mtime=0,compresslevel=3) as gz:
        with io.TextIOWrapper(gz,encoding='utf-8',newline='') as stream:
            writer=None
            for row in setup['final']:
                outputs={}
                for label,c in conditions.items():
                    if row['benchmark'] not in c['thresholds']:continue
                    try:
                        stage='dense' if label=='dense' else 'final'
                        out=cached(None,root,row,stage,label,c['name'],c['config'],c['thresholds'][row['benchmark']],contract)
                        audit_matrices(out,c['name'],c['config'],bank,matrices)
                        path=shard_path(root,stage,label,row['id']);sources[str(path)]=sha(path.read_bytes())
                        sources[out['records_source']['path']]=out['records_source']['sha256']
                        outputs[label]=out;completed+=1
                    except FileNotFoundError as error:missing.append(dict(condition=label,id=row['id'],error=str(error)))
                    except Exception as error:violations.append(dict(condition=label,id=row['id'],error=str(error)))
                if 'dense' not in outputs:continue
                if row['benchmark']=='longbench_v2':
                    paired,grades=nemo_pairs(row,outputs);graded.extend(grades)
                else:
                    paired=[dict(condition=label,**evidence.pair(row,out,outputs['dense'])) for label,out in outputs.items()]
                for p in paired:
                    label=p['condition'];out=outputs[label]
                    raw.append(dict(**p,prediction=out['prediction'],completion_tokens=out['completion_tokens'],
                        sub_domain=row.get('sub_domain',row['task']),truncated=row.get('truncated',False)))
                    accumulate_marginals(dimensions,out['records'],row['benchmark'],label)
                    if out.get('work_accounting'):
                        work.append(dict(id=row['id'],condition=label,**out['work_accounting']))
                    for r in out['records']:
                        item=dict(id=row['id'],benchmark=row['benchmark'],condition=label,**r)
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(item));writer.writeheader()
                        writer.writerow(item)
    temp.replace(root/'per_layer_head_step.csv.gz')
    groups=defaultdict(list);tasks=defaultdict(list)
    for row in raw:
        groups[row['benchmark'],'full',row['condition']].append(row)
        if row['benchmark']=='aime26':groups[row['benchmark'],'calibration6' if row['calibration'] else 'noncalibration24',row['condition']].append(row)
        tasks[row['benchmark'],row['task'],row['condition']].append(row)
    rows=[]
    for (benchmark,split,label),group in sorted(groups.items()):
        required=(30 if benchmark=='aime26' else 50) if split=='full' else 6 if split=='calibration6' else 24
        if len(group)!=required:
            missing.append(dict(benchmark=benchmark,split=split,condition=label,completed=len(group),expected=required));continue
        c=conditions[label];name=c['name'];cfg=Config(**c['config']) if name in PROJECTED else None
        rows.append(dict(benchmark=benchmark,split=split,condition=label,name=name,target=c['target'],
            thresholds=c['thresholds'].get(benchmark),projection_family=cfg.family if cfg and cfg.method!='mass_exact' else None,
            projection_rank=cfg.rank if cfg and cfg.family!='identity' and cfg.method!='mass_exact' else None,
            projection_seed=cfg.projection_seed if cfg and cfg.family!='identity' and cfg.method!='mass_exact' else None,
            **summarize(group)))
    taskrows=[]
    for (b,t,c),g in sorted(tasks.items()):
        required=sum(r['benchmark']==b and r['task']==t for r in setup['final'])
        taskrows.append(dict(benchmark=b,task=t,condition=c,expected=required,complete=len(g)==required,**summarize(g)))
    compared=compare(rows,groups)
    diagnostic_rows,extra=diagnostic_summary(root);sources.update(extra)
    seeds,extra=seed_summary(root,setup,contract,conditions);sources.update(extra)
    shared_seeds,extra=shared_seed_summary(root);sources.update(extra)
    extra_missing,extra_violations,extra=supporting_audit(root,setup,contract,conditions)
    missing.extend(extra_missing);violations.extend(extra_violations);sources.update(extra)
    failures=root/'failures.jsonl'
    audit=dict(complete=completed==80*len(setup['conditions']) and not missing and not violations,
        completed=completed,expected=80*len(setup['conditions']),missing=missing,violations=violations,
        failed_attempts=len(failures.read_text().splitlines()) if failures.exists() else 0,
        fingerprint=contract['fingerprint'],sources=sources)
    for name,data in (('summary',rows),('per_sample',raw),('comparisons',compared),('thresholds',policies),
            ('work_accounting',work),('shared_diagnostics_summary',diagnostic_rows),('projection_seed_sensitivity',seeds),
            ('projection_shared_seed_sensitivity',shared_seeds),('projection_matrices',matrices)):
        _write(root/f'{name}.json',data)
    csv_write(root/'summary.csv',rows);csv_write(root/'per_task.csv',taskrows);csv_write(root/'comparisons.csv',compared)
    csv_write(root/'projection_seed_sensitivity.csv',seeds)
    csv_write(root/'routing_marginals.csv',[dict(benchmark=b,condition=n,axis=a,index=i,attention_type=k,
        **aggregate([dict(zip(SUM_FIELDS,v))])) for (b,n,a,i,k),v in sorted(dimensions.items())])
    with (root/'nemo_predictions.jsonl').open('w') as f:
        for grade in graded:f.write(json.dumps(grade,sort_keys=True,ensure_ascii=False)+'\n')
    figures=plots(root,rows)
    write_report(root,setup,rows,compared,policies,audit,figures,diagnostic_rows,seeds)
    names=['summary.json','summary.csv','per_sample.json','per_task.csv','comparisons.json','comparisons.csv',
        'thresholds.json','work_accounting.json','shared_diagnostics_summary.json','projection_seed_sensitivity.json',
        'projection_seed_sensitivity.csv','projection_shared_seed_sensitivity.json','projection_matrices.json',
        'routing_marginals.csv','per_layer_head_step.csv.gz','nemo_predictions.jsonl','report.md',*figures]
    audit['artifacts']={name:sha((root/name).read_bytes()) for name in names}
    _write(root/'audit.json',audit)
    return audit
