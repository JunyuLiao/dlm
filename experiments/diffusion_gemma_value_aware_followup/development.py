"""Calibration/development quality evidence before any new final sparse run."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import traceback

from experiments.diffusion_gemma_value_aware.report import summary,csv_write,table
from experiments.diffusion_gemma_value_aware.policy_audit import point_for
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append
from .engine import contract
from .execution import cached
from .evidence import read_result,pair,raw_source,require_sparse_smoke,merge_sources
from .policies import METHODS,BENCHMARKS,inputs,calibration_rows,audit_policy
from .protocol import ROOT,MODEL,REVISION,TARGETS,prepare,sha


def validate_selection(names,targets,configs):
    if not names or len(names)!=len(set(names)) or not set(names)<=set(configs):
        raise ValueError('unknown/empty/duplicated method selection')
    if not targets or len(targets)!=len(set(targets)) or not set(targets)<=set(TARGETS):
        raise ValueError('invalid target selection')


def policy_for(root,setup,execution,benchmark,name,config,target):
    path=root/'verified_policies'/benchmark/f'{name}_s{round(target*100)}.json'
    data,source=raw_source(path)
    if (data['benchmark'],data['name'],data['target'])!=(benchmark,name,target):
        raise ValueError('verified policy path/identity mismatch')
    audit_policy(data,calibration_rows(setup,benchmark),config,execution)
    return data,source


def run(root,names,targets):
    setup=prepare(root);execution=contract(root);_,configs=inputs(root,execution)
    validate_selection(names,targets,configs)
    rows=setup['development']
    if len(rows)!=6 or any(r['split']!='development' or r['benchmark']!='longbench_v2' for r in rows):
        raise ValueError('only six disjoint v2 development examples may be executed here')
    adapter=None;failures=[]
    import torch
    from dllm.models import create_adapter
    for target in targets:
        for name in names:
            label=f'{name}_s{round(target*100)}'
            try:
                policy,_=policy_for(root,setup,execution,'longbench_v2',name,configs[name],target)
                require_sparse_smoke(root,setup,execution,{name:configs[name]})
            except Exception:
                error=dict(stage='development_policy',condition=label,traceback=traceback.format_exc())
                failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
                continue
            for row in sorted(rows,key=lambda r:(r['generation_budget'],r['id'])):
                try:
                    try:
                        out,_=read_result(root,row,'development',label,configs[name],policy['policy'],execution)
                    except FileNotFoundError:
                        if adapter is None:
                            adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',
                                precision='bfloat16',revision=REVISION).load()
                        out=cached(adapter,root,row,'development',label,configs[name],policy['policy'],execution)
                    dense,_=read_result(root,row,'screen','dense',{},None,execution,screen=True)
                    pair(row,out,dense)
                except Exception:
                    error=dict(stage='development',id=row['id'],condition=label,traceback=traceback.format_exc())
                    failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
                finally:
                    torch.cuda.empty_cache()
    result=report(root,names,targets)
    _write(root/'development_run_terminal.json',dict(finished=time.time(),complete=result['complete'],
        failed_attempts=len(failures),methods=names,targets=targets))
    if not result['complete']:
        raise RuntimeError('development incomplete; independent configurations continued')


def report(root,names,targets):
    setup=prepare(root);execution=contract(root);_,configs=inputs(root,execution)
    validate_selection(names,targets,configs)
    records=[];sources={};missing=[];violations=[]
    for target in targets:
        for name in names:
            condition=f'{name}_s{round(target*100)}'
            for benchmark in BENCHMARKS:
                try:
                    policy,source=policy_for(root,setup,execution,benchmark,name,configs[name],target)
                    sources[source['path']]=source['sha256']
                    point=point_for(policy)
                    groups=[('calibration',calibration_rows(setup,benchmark))]
                    if benchmark=='longbench_v2':
                        groups.append(('development',setup['development']))
                    for split,rows in groups:
                        for row in rows:
                            try:
                                if row['split']!=split:
                                    raise ValueError('development evidence contains a final example')
                                if split=='calibration':
                                    # policy_for has already verified every raw selected
                                    # calibration source, including exact1 reexpression.
                                    source=point['sources'][row['id']]
                                    out,observed=raw_source(source['path'])
                                    if observed['sha256']!=source['sha256']:
                                        raise ValueError('calibration quality evidence changed after policy audit')
                                else:
                                    out,source=read_result(root,row,'development',condition,configs[name],policy['policy'],execution)
                                dense,dense_source=read_result(root,row,'screen','dense',{},None,execution,screen=True)
                                records.append(dict(split=split,condition=condition,target=target,
                                    config=configs[name],thresholds=policy['policy'],**pair(row,out,dense)))
                                sources=merge_sources(sources,{source['path']:source['sha256'],
                                    dense_source['path']:dense_source['sha256']})
                            except FileNotFoundError as error:
                                missing.append(dict(condition=condition,id=row['id'],error=str(error)))
                            except Exception as error:
                                violations.append(dict(condition=condition,id=row['id'],error=str(error)))
                except FileNotFoundError as error:
                    missing.append(dict(condition=condition,benchmark=benchmark,error=str(error)))
                except Exception as error:
                    violations.append(dict(condition=condition,benchmark=benchmark,error=str(error)))
    grouped=defaultdict(list)
    for row in records:
        grouped[row['benchmark'],row['split'],row['condition']].append(row)
    summaries=[]
    for (benchmark,split,condition),group in grouped.items():
        if len(group)!=6 or len({r['id'] for r in group})!=6:
            continue
        first=group[0]
        summaries.append(dict(benchmark=benchmark,split=split,condition=condition,target=first['target'],
            method=first['config']['method'],pooling=first['config'].get('pooling'),
            thresholds=first['thresholds'],unparsed_answers=sum(r['unparsed_answer'] for r in group),**summary(group)))
    expected=len(names)*len(targets)*18
    audit=dict(complete=not missing and not violations and len(records)==expected,
        fingerprint=execution['fingerprint'],methods=names,targets=targets,expected=expected,completed=len(records),
        expected_groups=len(names)*len(targets)*3,complete_groups=len(summaries),
        missing=missing,violations=violations,sources=sources,heldout_used=False,
        raw_sparse_and_dense_checks=True,
        scope='Calibration6 per benchmark plus disjoint LongBench-v2 development6; no final output files')
    _write(root/'development_audit.json',audit);_write(root/'development_summary.json',summaries)
    _write(root/'development_per_sample.json',records);csv_write(root/'development_summary.csv',summaries)
    text='# Calibration/development evidence — not final results\n\n'
    text+=f"Complete: {audit['complete']}. {len(records)}/{expected} sample-condition pairs.\n\n"
    text+='Each row covers six tuning questions. AIME calibration accuracy is exploratory; v2 development is disjoint from calibration and final30. Thresholds use calibration physical counts, never final accuracy.\n\n'
    view=[dict(benchmark=r['benchmark'],split=r['split'],condition=r['condition'],n=r['count'],
        score=100*r['accuracy'],delta_pp=100*r['delta'],physical=100*r['overall_physical_sparsity'],
        global_s=100*r['global_physical_sparsity'],local_s=100*r['local_physical_sparsity'],
        pv_omission=100*r['overall_pv_omission'],mass=100*r['overall_denominator_mass'],
        exact_pv_mass=100*r['overall_mass'],error=r['overall_relative_error'],agreement=100*r['token_agreement'])
        for r in summaries]
    text+=table(view,['benchmark','split','condition','n','score','delta_pp','physical','global_s','local_s',
        'pv_omission','mass','exact_pv_mass','error','agreement'])+'\n\n'
    text+='Percentages except relative output error. Compensation/zero-PV target PV omission, not physical deletion; their exact denominator mass remains100%. These are same-state attention diagnostics, not a comparison of attention tensors on diverged generations.\n'
    (root/'development_report.md').write_text(text)
    return audit


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('run','report'))
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--methods',nargs='+',default=list(METHODS))
    p.add_argument('--targets',nargs='+',type=float,default=[.5]);args=p.parse_args()
    result=(run if args.command=='run' else report)(args.output,args.methods,args.targets)
    if args.command=='report':
        print(json.dumps({k:result[k] for k in ('complete','completed','expected','missing','violations')}))


if __name__=='__main__':
    main()
