"""Calibration/development-only quality evidence; never reads heldout outputs."""
import argparse
import json
from pathlib import Path
from .protocol import ROOT,prepare,fingerprint,sha,score
from .evaluate import candidate_configs
from .execution import assert_policy_provenance,provenance as operator_provenance,CONFIGS as REFINEMENTS
from .report import summary,csv_write,table,inspect_output
from .report_metrics import aggregate
from .run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import token_counts


def selected_point(policy):
    matches=[r for r in policy['trace'] if r['iteration']==policy['selected_round']]
    if len(matches)!=1:raise ValueError('ambiguous selected calibration round')
    point=matches[0]
    if point['policy']!=policy['policy']:raise ValueError('selected point differs from canonical policy')
    return point


def example(root,row,path,fp,config,policy=None,reused_boundary=False):
    out=json.loads(path.read_text())
    if out['fingerprint']!=fp or out['config']!=config:raise ValueError('incompatible development/calibration shard')
    if out.get('refinement_sha256')!=operator_provenance(config).get('refinement_sha256'):
        raise ValueError('development/calibration refinement source mismatch')
    if not reused_boundary and out['thresholds']!=policy:raise ValueError('wrong verified threshold')
    if reused_boundary:
        # A recorded exact-lambda1 point may have equivalent old metadata;
        # validate both actual type thresholds, not just the reuse label.
        from .policy_search import is_one
        if not all(is_one(out['thresholds'][k]) and is_one(policy[k]) for k in ('local','global')):
            raise ValueError('boundary reuse is not identical lambda1 for both types')
    for key in ('id','prompt_hash','seed','generation_budget'):
        if out[key]!=row[key]:raise ValueError(f'shard {key} mismatch')
    dense_path=shard_path(root,'screen','dense',row['id']);dense=json.loads(dense_path.read_text())
    if dense['fingerprint']!=fp or dense['config']:raise ValueError('wrong matched dense reference')
    for label,output,cfg in [('sparse',out,config),('dense',dense,{})]:
        # Exact-lambda1 representation equivalence was checked above. The
        # sparse operator must still pass all raw coverage/count checks; dense
        # additionally needs full prompt/seed identity, not fingerprint alone.
        thresholds=out['thresholds'] if label=='sparse' else None
        condition=dict(config=cfg,thresholds={row['benchmark']:thresholds},**operator_provenance(cfg))
        _,_,errors=inspect_output(output,row,fp,condition)
        if errors:raise ValueError(f'{label} development evidence audit: {errors}')
    matching,compared=token_counts(dense['completion_tokens'],out['completion_tokens'])
    records=[r for r in out['records'] if r['probe']=='execution']
    return (dict(id=row['id'],task=row['task'],accuracy=score(row,out['prediction']),dense_accuracy=score(row,dense['prediction']),
        matching=matching,compared=compared,exact_match=dense['completion_tokens']==out['completion_tokens'],
        output_length=len(out['completion_tokens']),termination_reason=out['termination_reason'],
        aggregates={k:aggregate([r for r in records if k=='overall' or r['attention_type']==k]) for k in ('overall','global','local')}),
        {str(path):sha(path.read_bytes()),str(dense_path):sha(dense_path.read_bytes())})


def report(root=ROOT,names=None,targets=(.5,.75)):
    setup=prepare(root);fp=fingerprint(root);configs=candidate_configs(root)
    names=names or [n for n in configs if n not in REFINEMENTS]
    summaries=[];missing=[];sources={};raw=[]
    for name in names:
        for target in targets:
            label=f'{name}_s{int(target*100)}'
            for benchmark in ('aime26','longbench'):
                path=root/'verified_policies'/benchmark/f'{label}.json'
                if not path.exists():missing.append(str(path));continue
                policy=json.loads(path.read_text());assert_policy_provenance(policy,configs[name],fp)
                assert not policy['heldout_used']
                rows=[r for r in setup['calibration'] if r['benchmark']==benchmark]
                assert set(policy['calibration_ids'])=={r['id'] for r in rows}
                point=selected_point(policy);sources[str(path)]=sha(path.read_bytes())
                stage=point.get('verification_stage',policy.get('verification_stage','calibration'))
                groups=[('calibration',rows,stage,point['source_condition'])]
                if benchmark=='longbench':groups.append(('development',setup['development'],'development',label))
                for split,examples,source_stage,condition in groups:
                    outputs=[]
                    for row in examples:
                        shard=shard_path(root,source_stage,condition,row['id'])
                        if not shard.exists():missing.append(str(shard));continue
                        out,provenance=example(root,row,shard,fp,configs[name],policy['policy'],
                            reused_boundary=split=='calibration' and bool(point.get('reused_exact_lambda1_source')))
                        outputs.append(out);sources.update(provenance)
                        raw.append(dict(benchmark=benchmark,split=split,condition=label,**out))
                    if len(outputs)!=len(examples):continue
                    summaries.append(dict(benchmark=benchmark,split=split,condition=label,target=target,
                        method=configs[name]['method'],pooling=configs[name].get('pooling'),
                        policy=policy['policy'],**summary(outputs)))
    audit=dict(complete=not missing,fingerprint=fp,methods=names,targets=list(targets),missing=missing,sources=sources,
        source_scope='Only calibration16 and disjoint LongBench development5; never final output shards',
        raw_sparse_and_dense_checks=True,heldout_used=False)
    _write(root/'development_audit.json',audit);_write(root/'development_summary.json',summaries)
    csv_write(root/'development_summary.csv',summaries);_write(root/'development_per_sample.json',raw)
    text='# Calibration/development evidence — not final results\n\n'
    text+=('Complete for the explicitly requested methods and targets.\n\n' if audit['complete'] else f'Incomplete: {len(missing)} missing artifacts. No candidate selection may call this a complete comparison.\n\n')
    text+='AIME calibration6 accuracy is exploratory. LongBench development5 is disjoint from calibration10 and final50. Threshold fitting uses calibration sparsity, never heldout accuracy.\n\n'
    view=[dict(benchmark=r['benchmark'],split=r['split'],condition=r['condition'],n=r['count'],score=r['accuracy'],dense_delta=r['delta'],
        actual=r['overall_physical_sparsity'],global_s=r['global_physical_sparsity'],local_s=r['local_physical_sparsity'],
        mass=r['overall_denominator_mass'],error=r['overall_relative_error'],agreement=r['token_agreement']) for r in summaries]
    text+=table(view,['benchmark','split','condition','n','score','dense_delta','actual','global_s','local_s','mass','error','agreement'])+'\n'
    replacements=[dict(benchmark=r['benchmark'],split=r['split'],condition=r['condition'],n=r['count'],
        target_pv_omission=r['target'],physical=r['overall_physical_sparsity'],
        pv_omission=r['overall_pv_omission'],global_pv=r['global_pv_omission'],local_pv=r['local_pv_omission'],
        denominator_mass=r['overall_denominator_mass'],exact_pv_mass=r['overall_mass'],
        error=r['overall_relative_error'],score=r['accuracy']) for r in summaries
        if r['method'] in ('compensate','zero_pv')]
    if replacements:
        text+='\n## PV-replacement controls — not physical deletion\n\n'
        text+='All values below are fractions. These targets calibrate omitted full-PV tiles / eligible tiles. Softmax and the denominator remain exact; physical deletion stays zero. Denominator mass is therefore1 even when the exact value contribution is replaced.\n\n'
        text+=table(replacements,['benchmark','split','condition','n','target_pv_omission','physical',
            'pv_omission','global_pv','local_pv','denominator_mass','exact_pv_mass','error','score'])+'\n'
    (root/'development_report.md').write_text(text)
    return audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--methods',nargs='+');p.add_argument('--targets',nargs='+',type=float,default=[.5,.75])
    args=p.parse_args();report(args.output,args.methods,args.targets)


if __name__=='__main__':main()
