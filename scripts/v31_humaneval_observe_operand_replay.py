"""Bounded exact-input FA4 observation replay; zero native generation calls."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import torch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True)


def main(root):
    p=json.loads((root/'protocol.json').read_text())
    for name,digest in p['files_sha256'].items():assert sha(root/name)==digest
    from scripts.v31_humaneval_operand_restore import tensor
    from scripts.v31_humaneval_last_global_compare import tensor_difference
    native=runpy.run_path(p['native_base_worker'])
    patched=native['apply_fa4_local_fix']()
    from experiments.numerical_qk_reuse import v31_fa4_observe as observe
    from vllm.vllm_flash_attn.cute import interface,flash_fwd_sm90 as sm90
    assert sha(Path(interface.__file__))==p['interface_sha256']
    fwd=interface._flash_attn_fwd;original_mask=sm90.AttentionMask
    original_config=interface._get_fwd_config;configurations=[]
    def config(*args,**kwargs):
        result=original_config(*args,**kwargs)
        configurations.append(dict(num_splits=result.num_splits,tile_m=result.m_block_size,tile_n=result.n_block_size,
            overlap=result.intra_wg_overlap,pack_gqa=kwargs['pack_gqa']))
        return result
    interface._get_fwd_config=config
    captures={'count':0};capture=torch.cuda.CUDAGraph.capture_begin
    def begin(graph,*a,**k):captures['count']+=1;return capture(graph,*a,**k)
    torch.cuda.CUDAGraph.capture_begin=begin
    result=dict(status='operator_diagnostic_only',native_model_forwards=0,native_sampler_calls=0,
        runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            matmul_precision=torch.get_float32_matmul_precision(),tf32=torch.backends.cuda.matmul.allow_tf32,
            guard=torch.are_deterministic_algorithms_enabled(),cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            cublas_bytes=torch._C._cuda_getCublasWorkspaceSize(),fa4_local_fix_sha256=sha(Path(patched))),cases=[])
    with torch.inference_mode():
        for case in p['cases']:
            path=Path(case['raw_path']);assert sha(path)==case['raw_sha256']
            rows=torch.load(path,map_location='cpu',weights_only=True)
            raw=next(x for x in rows if x['kind']=='fa4' and x['sampler_index']==case['sampler_index'])
            assert raw['layout']=='contiguous' and raw['consumer']=='observing_dense_fwd'
            q,k,v=(tensor(raw[name]) for name in ('q','k','v'))
            kwargs=dict(raw['scalar_kwargs']);kwargs.update({key:tensor(value) for key,value in raw['tensor_kwargs'].items()})
            assert raw.get('block_sparse_tensors') is None
            aux=[]
            for i,item in enumerate(raw['aux']):
                if i==0:
                    aux.append(torch.empty_strided(item['shape'],item['stride'],device='cuda',dtype=getattr(torch,item['dtype'].removeprefix('torch.'))))
                else:aux.append(tensor(item['input']))
            saved=[x.detach().cpu().clone() for x in (q,k,v)]
            reference=raw['output']['value']
            item=dict(label=case['label'],sampler_index=case['sampler_index'],shape=dict(q=list(q.shape),k=list(k.shape)),
                input_scope='complete captured contiguous Q/K/V and named arguments; auxiliary0 is write-only',variants=[])
            # FP64 oracle is diagnostic only and is never used for native decoding.
            qq=q.double().permute(0,2,1,3);kk=k.double().permute(0,2,1,3);vv=v.double().permute(0,2,1,3)
            factor=qq.shape[1]//kk.shape[1];kk=kk.repeat_interleave(factor,1);vv=vv.repeat_interleave(factor,1)
            scores=qq@kk.transpose(-1,-2)*float(kwargs['softmax_scale'])
            oracle=(scores.softmax(-1)@vv).permute(0,2,1,3).float().cpu()
            del qq,kk,vv,scores
            for name,splits,observing,poison in (
                ('native_auto_observe',0,True,None),('native_auto_observe_scratch_zero',0,True,0.),
                ('native_auto_observe_scratch_finite',0,True,37.),('native_auto_observe_scratch_nan',0,True,float('nan')),
                ('single_split_observe',1,True,None),('native_auto_plain',0,False,None),('single_split_plain',1,False,None)):
                sm90.AttentionMask=observe.ObservingMask if observing else original_mask
                options=dict(kwargs,num_splits=splits)
                if observing:options['aux_tensors']=aux
                trials=[];baseline=None;native_matches=0;repeat_matches=0;first_change=None;first_output=None
                for repeat in range(p['repeats']):
                    # Deliberately vary freed scratch contents, recording actual poison scope.
                    # This is a separate operator diagnostic, never native timing.
                    scratch=[]
                    if poison is not None and configurations:
                        count=configurations[-1]['num_splits']
                        if count>1:
                            for shape in ((count,*q.shape),(count,q.shape[0],q.shape[2],q.shape[1])):
                                buffer=torch.empty(shape,device=q.device,dtype=torch.float32);buffer.fill_(poison);scratch.append(buffer)
                            torch.cuda.synchronize();del buffer;scratch.clear()
                    if observing:aux[0].fill_(float('nan') if repeat%2 else 0.)
                    out=fwd(q,k,v,**options)[0]
                    torch.cuda.synchronize();host=out.detach().cpu().contiguous()
                    if baseline is None:baseline=host.clone();first_output=host.clone()
                    same=tensor_difference(baseline,host);match=tensor_difference(reference,host)
                    repeat_matches+=int(same['equal']);native_matches+=int(match['equal'])
                    if not same['equal'] and first_change is None:first_change=dict(repeat=repeat,delta=same)
                    if observing:
                        # Require every true prefix log-mass slot to have been written.
                        finite=bool(torch.isfinite(aux[0]).all().item())
                    else:finite=None
                    trials.append(dict(repeat=repeat,equal_first=same['equal'],equal_captured_native=match['equal'],aux_all_finite=finite))
                error=(first_output.float()-oracle).double()
                item['variants'].append(dict(name=name,repeats=p['repeats'],repeat_matches=repeat_matches,native_matches=native_matches,
                    first_changed_repeat=first_change,trials=trials,actual_config=configurations[-1],
                    fp64_oracle_error=dict(max_absolute=float(error.abs().max()),rmse=float((error.square().mean()).sqrt())),
                    captured_native_delta=tensor_difference(reference,first_output)))
                del baseline,first_output,host,out
            for before,after in zip(saved,(q,k,v)):assert tensor_difference(before,after.detach().cpu())['equal'],'native operator mutated input'
            item['qkv_preserved']=True;result['cases'].append(item)
            del rows,raw,q,k,v,kwargs,aux,saved,reference,oracle
            torch.cuda.empty_cache()
    sm90.AttentionMask=original_mask
    result['unexpected_cuda_captures']=captures['count'];assert captures['count']==0
    write(root/'private/operator_summary.json',result)
    print(json.dumps(dict(status=result['status'],cases=len(result['cases']),unexpected_cuda_captures=captures['count'])),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
