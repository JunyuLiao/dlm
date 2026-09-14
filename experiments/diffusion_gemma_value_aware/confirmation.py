"""One additional paired seed on all80 prompts, without threshold retuning."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from dllm.models import create_adapter
from .protocol import ROOT,MODEL,REVISION,prepare,fingerprint,frozen_write,sha,score
from .evaluate import run_group
from .run import shard_path,smoke
from .execution import require_refinement_smoke
from .report import inspect_shard,summary,csv_write,table
from .report_metrics import aggregate
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import token_counts


def build_contract(final,fp,names,offset):
    if offset<=0:raise ValueError('confirmation uses a distinct positive seed offset')
    if not names or not set(names)<=set(final['conditions']):raise ValueError('explicit existing frozen conditions required')
    conditions={n:dict(final['conditions'][n]) for n in dict.fromkeys(['dense']+names)}
    # Same dense operator, explicit nonempty config prevents the generator's
    # original-seed legacy AIME parity check from being incorrectly applied.
    conditions['dense']['config']=dict(method='dense')
    return dict(fingerprint=fp,seed_offset=offset,conditions=conditions,
        retuned=False,expected_per_condition={'aime26':30,'longbench':50},
        selection_scope='Confirmation of existing frozen points; may be selected after primary results, never used to redesign/retune this confirmatory sweep',
        dense_config_note='Explicit method=dense is identical attention; a new seed cannot be compared to old-seed cached tokens')


def run(root=ROOT,names=None,offset=10000):
    setup=prepare(root);fp=fingerprint(root)
    assert json.loads((root/'audit.json').read_text())['complete'],'complete primary evaluation required first'
    final_path=root/'final_contract.json';final=json.loads(final_path.read_text())
    assert final['fingerprint']==fp
    contract=build_contract(final,fp,names,offset);contract['primary_contract_sha256']=sha(final_path.read_bytes())
    stage=f'paired_seed_{offset}';dest=root/stage
    frozen_write(dest/'contract.json',contract)
    require_refinement_smoke(root,fp,[c['config'] for c in contract['conditions'].values()])
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    smoke(adapter,setup,root)
    for name,condition in contract['conditions'].items():
        for benchmark in ('aime26','longbench'):
            rows=[dict(r,seed=r['seed']+offset) for r in setup['final'] if r['benchmark']==benchmark]
            run_group(adapter,root,rows,stage,name,condition['config'],condition.get('thresholds',{}).get(benchmark),fp)
    report(root,offset)


def report(root=ROOT,offset=10000):
    setup=prepare(root);fp=fingerprint(root);stage=f'paired_seed_{offset}';dest=root/stage
    contract=json.loads((dest/'contract.json').read_text());assert contract['fingerprint']==fp
    assert contract['primary_contract_sha256']==sha((root/'final_contract.json').read_bytes())
    missing=[];violations=[];raw=[];sources={}
    for original in setup['final']:
        row=dict(original,seed=original['seed']+offset)
        dense_path=shard_path(root,stage,'dense',row['id'])
        dense=json.loads(dense_path.read_text()) if dense_path.exists() else None
        for name,condition in contract['conditions'].items():
            path=shard_path(root,stage,name,row['id'])
            if not path.exists():missing.append(str(path));continue
            d,records,errors=inspect_shard(path,row,fp,condition)
            violations.extend(dict(path=str(path),error=e) for e in errors);sources[str(path)]=sha(path.read_bytes())
            if dense is None:continue
            for key in ('native_canvas_length','thinking','sampling','denoising_configuration'):
                if d['generation_metadata'].get(key)!=dense['generation_metadata'].get(key):
                    violations.append(dict(path=str(path),error=f'unrelated decoding setting changed: {key}'))
            matching,compared=token_counts(dense['completion_tokens'],d['completion_tokens'])
            raw.append(dict(id=row['id'],benchmark=row['benchmark'],task=row['task'],calibration=row['calibration'],
                seed=row['seed'],condition=name,accuracy=score(row,d['prediction']),dense_accuracy=score(row,dense['prediction']),
                matching=matching,compared=compared,exact_match=d['completion_tokens']==dense['completion_tokens'],
                output_length=len(d['completion_tokens']),termination_reason=d['termination_reason'],
                aggregates={k:aggregate([r for r in records if k=='overall' or r['attention_type']==k]) for k in ('overall','global','local')}))
    expected=80*len(contract['conditions'])
    audit=dict(complete=not missing and not violations and len(raw)==expected,expected=expected,completed=len(raw),
        missing=missing,violations=violations,sources=sources,retuned=False,fingerprint=fp)
    _write(dest/'audit.json',audit)
    if not audit['complete']:return audit
    groups=defaultdict(list)
    for row in raw:
        groups[row['benchmark'],'full',row['condition']].append(row)
        if row['benchmark']=='aime26' and not row['calibration']:groups[row['benchmark'],'heldout24',row['condition']].append(row)
    rows=[dict(benchmark=b,split=s,condition=n,seed_offset=offset,**summary(g)) for (b,s,n),g in groups.items()]
    _write(dest/'summary.json',rows);csv_write(dest/'summary.csv',rows);_write(dest/'per_sample.json',raw)
    text=f'# Paired-seed confirmation — offset{offset}\n\nAll contracted conditions cover AIME26 full30 and LongBench50. Original thresholds are unchanged; new seeds are paired across dense and sparse methods. Selection of these points can follow primary heldout results, but there is no threshold or algorithm retuning.\n\n'
    text+=table(rows,['benchmark','split','condition','count','accuracy','dense_accuracy','delta','paired_ci95','overall_physical_sparsity','global_physical_sparsity','local_physical_sparsity','overall_mass','token_agreement'])+'\n'
    (dest/'report.md').write_text(text)
    return audit


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','report']);p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--methods',nargs='+');p.add_argument('--seed-offset',type=int,default=10000);args=p.parse_args()
    if args.command=='run':run(args.output,args.methods,args.seed_offset)
    else:report(args.output,args.seed_offset)


if __name__=='__main__':main()
