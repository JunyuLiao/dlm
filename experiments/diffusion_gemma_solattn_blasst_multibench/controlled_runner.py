"""Resumable, fingerprinted controlled sweep; completed samples are immutable."""
from __future__ import annotations
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch
from dllm.models import create_adapter
from dllm.attention.blasst.core import BLASST_MASK_SEMANTICS, blasst_2d_attention_forward
from experiments.diffusion_attention_threshold_modeling.routing import FreshRoutingAttention
from .config import REVISION, condition_map
from .controlled_dataset import audit_manifest, code_from_prediction, digest
from .diagnostics import DenseDiagnostics, CheckedAttention, sol_config
from .runner import (_rows, _write, _append, _fingerprint, _install_dense,
    _install_blasst, _set_context, _request, _policy_for_target, POLICY)


def code_hashes():
    paths = [Path('src/dllm/attention/blasst/core.py'),Path('src/dllm/attention/blasst/integration.py'),
        Path('src/dllm/models/adapters/diffusion_gemma.py'),
        Path('experiments/diffusion_attention_threshold_modeling/routing.py'),
        Path('experiments/diffusion_attention_threshold_modeling/proxy.py'),
        Path(__file__), Path(__file__).with_name('diagnostics.py')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def provenance(manifest_path, model_path, revision):
    return dict(schema='controlled_multibench_v1',manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        model_path=model_path,revision=revision,precision='bfloat16',temperature=0.,seed='per manifest row',
        block_size=256,thinking=False,denoising='pinned native model defaults',tile_size=[64,64],
        sol_region='prefix_plus_canvas_combined_population',blasst_mask_semantics=BLASST_MASK_SEMANTICS,
        policy_sha256=hashlib.sha256(POLICY.read_bytes()).hexdigest(),code_sha256=code_hashes(),
        torch_version=torch.__version__,transformers_version=__import__('transformers').__version__)


def shard_path(output_dir,name,identity):
    return output_dir/'conditions'/name/'shards'/(digest(identity)+'.json')


def load_completed(output_dir,name):
    return [json.loads(p.read_text()) for p in sorted((output_dir/'conditions'/name/'shards').glob('*.json'))]


def install(adapter,condition):
    if condition.method=='blasst':
        stats,binding,_=_install_blasst(adapter,condition)
        checked=CheckedAttention(blasst_2d_attention_forward)
        binding.runtime.attention_override=checked
        return binding,stats,checked
    binding=_install_dense(adapter)
    diagnostic=DenseDiagnostics() if condition.method=='dense' else FreshRoutingAttention(sol_config(condition.target_sparsity))
    checked=CheckedAttention(diagnostic)
    binding.runtime.attention_override=checked
    return binding,diagnostic,checked


def snapshot(condition,stats):
    if condition.method=='dense': return dict(calls=stats.calls['dense'],dense_counterfactual=stats.calls)
    if condition.method=='sol': return dict(calls=stats.stats.per_call,routing_summary=stats.stats.summary())
    return dict(calls=list(stats.per_step.values()),per_layer=list(stats.per_layer.values()),
        blasst_summary=stats.summary())


def smoke(manifest_path,output_dir,*,model_path,revision=REVISION):
    config=provenance(manifest_path,model_path,revision)
    path=output_dir/'smoke'/'audit.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('fingerprint')==_fingerprint(config) and old.get('passed'): return old
    manifest=_rows(manifest_path)
    rows=[next(r for r in manifest if r['benchmark']=='ruler16k'),
          next(r for r in manifest if r['benchmark']=='longbench' and r['task']=='trec')]
    adapter=create_adapter('diffusion_gemma',model_path,device='cuda',precision='bfloat16',revision=revision).load()
    records=[]
    for original in rows:
        row=dict(original,generation_budget=32)
        binding=_install_dense(adapter)
        try:
            _set_context(binding,row); dense=adapter.generate(_request(row))
        finally: binding.close()
        binding,stats,checked=install(adapter,condition_map()['dense'])
        try:
            _set_context(binding,row); probe=adapter.generate(_request(row)); calls=stats.calls
        finally: binding.close()
        binding=_install_dense(adapter); unpruned=FreshRoutingAttention(sol_config(0.))
        binding.runtime.attention_override=CheckedAttention(unpruned)
        try:
            _set_context(binding,row); unmasked=adapter.generate(_request(row))
        finally: binding.close()
        routed={}
        for name in ('sol_gaussian_s90','blasst_length_aware_s25'):
            binding,stats,checked=install(adapter,condition_map()[name])
            try:
                _set_context(binding,row); generated=adapter.generate(_request(row))
                routed[name]=dict(snapshot(condition_map()[name],stats),finite_attention_calls=checked.calls,
                    completion_tokens=generated.completion_tokens)
            finally: binding.close()
        coverage=calls['dense']
        record=dict(id=row['id'],dense_diagnostic_exact_parity=probe.completion_tokens==dense.completion_tokens,
            unpruned_sol_exact_parity=unmasked.completion_tokens==dense.completion_tokens,
            zero_prune_skipped=unpruned.stats.physical_skipped_tiles,
            layer_ids=sorted({c['layer'] for c in coverage}),head_ids=sorted({h for c in coverage for h in c['head_ids']}),
            attention_types=sorted({c['attention_type'] for c in coverage}),
            denoising_steps=sorted({c['denoising_step'] for c in coverage}),
            prefix_skipped=sum(c['region_counts']['prefix']['skipped_tiles'] for c in calls['sol_gaussian_s90']),
            canvas_skipped=sum(c['region_counts']['canvas']['skipped_tiles'] for c in calls['sol_gaussian_s90']),
            dense_eligible=sum(c['eligible_tiles'] for c in coverage),
            physical_counts_match=all(sum(c['eligible_tiles'] for c in v)==sum(c['eligible_tiles'] for c in coverage) for v in calls.values()),
            finite_outputs=all(v['finite_attention_calls']>0 and len(v['completion_tokens'])>0 for v in routed.values()))
        _write(output_dir/'smoke'/(digest(row['id'])+'.json'),dict(record=record,dense_calls=calls,routed=routed))
        records.append(record); print('SMOKE',row['id'],record,flush=True)
    passed=all(r['dense_diagnostic_exact_parity'] and r['unpruned_sol_exact_parity'] and r['zero_prune_skipped']==0
        and r['physical_counts_match'] and r['finite_outputs'] and r['prefix_skipped']>0 and r['canvas_skipped']>0
        and r['attention_types']==['global','local'] and len(r['layer_ids'])==adapter.model.config.text_config.num_hidden_layers
        for r in records)
    result=dict(passed=passed,fingerprint=_fingerprint(config),records=records,setup=config)
    _write(path,result)
    if not passed: raise RuntimeError('CUDA smoke audit failed')
    return result


def run(manifest_path,output_dir,*,model_path,revision=REVISION,selected=None):
    audit_manifest(output_dir)
    shared=provenance(manifest_path,model_path,revision)
    smoke_audit=json.loads((output_dir/'smoke'/'audit.json').read_text())
    if not smoke_audit.get('passed') or smoke_audit['fingerprint']!=_fingerprint(shared): raise RuntimeError('matching passing smoke required')
    _write(output_dir/'calibration_policy.json',json.loads(POLICY.read_text()))
    # Small generation budgets first; dense always precedes sparse conditions.
    rows=sorted(_rows(manifest_path),key=lambda r:(r['generation_budget'],r['prompt_tokens'],r['id']))
    names=list(selected or condition_map())
    if 'dense' in names: names.remove('dense'); names.insert(0,'dense')
    adapter=create_adapter('diffusion_gemma',model_path,device='cuda',precision='bfloat16',revision=revision).load()
    _write(output_dir/'model_runtime.json',dict(adapter=adapter.runtime_metadata(),model_config=adapter.model.config.to_dict()))
    failures=[]; summaries=[]
    for name in names:
        condition=condition_map()[name]
        config=dict(shared,condition=condition.to_dict(),blasst_length_aware_policy=
            _policy_for_target(condition.target_sparsity) if condition.method=='blasst' else None)
        fingerprint=_fingerprint(config); config['fingerprint']=fingerprint
        directory=output_dir/'conditions'/name
        if (directory/'run_config.json').exists() and json.loads((directory/'run_config.json').read_text())!=config:
            raise RuntimeError(f'incompatible resume: {directory}')
        _write(directory/'run_config.json',config)
        done={r['id']:r for r in load_completed(output_dir,name)}
        if any(r['fingerprint']!=fingerprint for r in done.values()): raise RuntimeError('incompatible completed shards')
        for index,row in enumerate(rows,1):
            if row['id'] in done: continue
            if name!='dense' and not shard_path(output_dir,'dense',row['id']).exists():
                failures.append(dict(condition=name,id=row['id'],error='missing dense baseline'));continue
            binding=None
            started=time.time()
            _write(output_dir/'progress.json',dict(condition=name,index=index,total=len(rows),id=row['id'],started=started))
            try:
                binding,stats,checked=install(adapter,condition)
                _set_context(binding,row); generated=adapter.generate(_request(row))
                if not generated.completion_tokens or checked.calls==0: raise RuntimeError('empty generation or missing attention coverage')
                result=dict(row,fingerprint=fingerprint,condition=name,prediction=generated.text,
                    completion_tokens=generated.completion_tokens,termination_reason=generated.termination_reason,
                    finite_attention_calls=checked.calls,**snapshot(condition,stats),
                    code=code_from_prediction(generated.text) if row['benchmark']=='livecodebench_v6' else None)
                # Runtime retained only for stall diagnostics, never reported as speedup.
                result['diagnostic_wall_seconds']=time.time()-started
                _write(shard_path(output_dir,name,row['id']),result);done[row['id']]=result
                print(time.strftime('%FT%TZ',time.gmtime()),name,f'{len(done)}/{len(rows)}',row['id'],flush=True)
            except Exception as exc:
                error=dict(condition=name,id=row['id'],error=repr(exc),traceback=traceback.format_exc(),time=time.time())
                _append(output_dir/'failures.jsonl',error);failures.append(error)
                print('FAILED',name,row['id'],repr(exc),flush=True)
            finally:
                if binding is not None: binding.close()
                torch.cuda.empty_cache()
        summaries.append(dict(condition=name,completed=len(done),expected=len(rows)))
        _write(output_dir/'run_status.json',dict(conditions=summaries,failures=failures,finished=False))
    result=dict(conditions=summaries,failures=failures,finished=True)
    _write(output_dir/'run_status.json',result)
    return result
