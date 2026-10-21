"""Charged synthetic resident sparse consumers; frozen masks, current Q/K/V."""
import argparse,hashlib,json,os,subprocess,time,statistics
from pathlib import Path
import torch
from experiments.numerical_qk_reuse import v27_fa4
from experiments.numerical_qk_reuse.resident_sparse import ResidentMask,PagedContext,FlexConsumer,GraphPagedBucket
from scripts.v2_sparse_refresh_bench import graph_functions,timed

def host_timing(functions,rounds,repeats=30):
 """Steady-state Python/module cost, including launches and final synchronization.

 CUDA graphs are not used here. One synchronization per batch is charged; these
 measurements complement graph replay timings instead of treating the latter as
 attention-module latency.
 """
 samples={name:[] for name in functions};names=list(functions)
 for r in range(rounds):
  for name in names[r%len(names):]+names[:r%len(names)]:
   torch.cuda.synchronize();start=time.perf_counter()
   for _ in range(repeats):functions[name]()
   torch.cuda.synchronize();samples[name].append((time.perf_counter()-start)*1000/repeats)
 return {name:dict(median_ms=statistics.median(values),samples_ms=values) for name,values in samples.items()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--keys',type=int,nargs='+',default=[1024,4096,8192,16384]);p.add_argument('--sparsities',type=float,nargs='+',default=[.4,.5,.6]);p.add_argument('--flex',action='store_true');p.add_argument('--rounds',type=int,default=7);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[];captures=[0];original=torch.cuda.CUDAGraph.capture_begin
 def capture(self,*args,**kwargs):captures[0]+=1;return original(self,*args,**kwargs)
 torch.cuda.CUDAGraph.capture_begin=capture
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('experiments/numerical_qk_reuse/resident_sparse.py')]}
 (a.out/'source').mkdir()
 for path in hashes:(a.out/'source'/Path(path).name).write_bytes(Path(path).read_bytes())
 def checkgpu():
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
  if any(int(x.split(',')[0])!=os.getpid() for x in apps.splitlines() if x.strip()):raise RuntimeError('foreign GPU worker')
  return apps
 for nk in a.keys:
  for sparsity in a.sparsities:
   before=checkgpu();g=torch.Generator(device='cuda').manual_seed(nk+int(sparsity*100));n=256;h=16;hk=2;d=512;kt=nk//64;pt=(nk-n)//64
   q=torch.randn((1,h,n,d),device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn((1,hk,nk,d),device='cuda',dtype=q.dtype,generator=g);v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=g)
   keep=torch.ones((1,h,2,kt),device='cuda',dtype=torch.bool);keep[...,:pt]=False
   count=max(1,round(kt*(1-sparsity))-(kt-pt));chosen=torch.rand((h,2,pt),device='cuda',generator=g).argsort(-1)[...,:count]
   keep[0,:,:,:pt].scatter_(-1,chosen,True);lists=v27_fa4.block_sparse_tensors(keep);scale=d**-.5
   # Interleaved native-like K/V cache with a nontrivial logical page table.
   pages=nk//128;table=torch.randperm(pages,device='cuda',generator=g).int();cache=torch.empty((pages,128,hk,2*d),device='cuda',dtype=q.dtype);key=cache[...,:d];value=cache[...,d:]
   key[table.long()]=k[0].transpose(0,1).reshape(pages,128,hk,d);value[table.long()]=v[0].transpose(0,1).reshape(pages,128,hk,d)
   contexts={s:PagedContext(key,value,table,nk,splits=s) for s in [1,2,4]};masks={s:ResidentMask.create(keep,splits=s,prefix_tiles=pt,audit=True) for s in contexts}
   def old_paged(s=2):
    from experiments.numerical_qk_reuse.v31_local_sparse import alias64
    pk,pv,tab=alias64(key,value,table,nk);qs=q.transpose(1,2).expand(s,-1,-1,-1);tab=tab.expand(s,-1).contiguous();used=torch.full((s,),nk,device=q.device,dtype=torch.int32)
    out,lse=v27_fa4.load()(qs,pk,pv,softmax_scale=scale,causal=False,page_table=tab,seqused_k=used,block_sparse_tensors=masks[s].lists,num_splits=1,return_lse=True)[:2]
    from experiments.numerical_qk_reuse.resident_sparse import _merge
    result=torch.empty((1,n,h,d),device=q.device,dtype=q.dtype);_merge[(n,h)](out,lse,result,n,h,d,s,d,num_warps=4);return result
   from experiments.numerical_qk_reuse.v29_paged_copy import copy_paged_kv
   copied_k=torch.empty_like(k);copied_v=torch.empty_like(v)
   def copy_canvas():copy_paged_kv(key,value,table,copied_k,copied_v,nk-n,n)
   def copy_prefix():copy_paged_kv(key,value,table,copied_k,copied_v,0,nk)
   dynamic=torch.tensor([False],device=q.device);native_used=torch.tensor([nk],device=q.device,dtype=torch.int32)
   def native_paged_dense():
    return v27_fa4.load()(q.transpose(1,2),key,value,page_table=table[None],seqused_k=native_used,
     softmax_scale=scale,causal=True,dynamic_causal=dynamic,pack_gqa=False,num_splits=0,intra_wg_overlap=False)[0]
   funcs=dict(dense_plain=lambda:v27_fa4.dense_plain(q,k,v,scale),native_dense_paged=native_paged_dense,allkept_fa4=lambda:v27_fa4.dense(q,k,v,scale),contiguous_sparse=lambda:v27_fa4.sparse_lists(q,k,v,lists,scale),previous_paged_step=lambda:(copy_canvas(),old_paged())[1],copy_canvas=copy_canvas,copy_full=copy_prefix,mask_pack_s2=lambda:ResidentMask.create(keep,splits=2),page_metadata=lambda:contexts[2].update_table())
   for s in contexts:
    funcs[f'resident_paged_s{s}']=lambda s=s:contexts[s](q,masks[s],scale)
    if s==1:funcs['resident_paged_overlap']=lambda:contexts[1](q,masks[1],scale,overlap=True)
   native_out=torch.empty((1,n,h,d),device=q.device,dtype=q.dtype)
   bucket=GraphPagedBucket(key,value,table,nk,n,h,scale,splits=1 if nk<=4096 else 2)
   bucket.prepare(masks[bucket.context.splits],table,nk)
   funcs['held_bucket_charged']=lambda:bucket(q,out=native_out)
   if a.flex:
    flex=FlexConsumer(keep,n,nk);funcs['flex_resident']=lambda:flex(q,k,v,scale)
    def flex_copy():copy_canvas();return flex(q,copied_k,copied_v,scale)
    copy_prefix();funcs['flex_canvas_copy']=flex_copy
   for f in funcs.values():
    for _ in range(2):f()
   expected=v27_fa4.sparse_lists(q,k,v,lists,scale);errors={}
   dense_expected=v27_fa4.dense_plain(q,k,v,scale);dense_got=native_paged_dense()
   errors['native_dense_paged_vs_plain']=float((dense_got.float()-dense_expected.float()).norm()/dense_expected.float().norm())
   if not torch.isfinite(dense_got).all() or errors['native_dense_paged_vs_plain']>.01:raise RuntimeError('native paged dense parity')
   for name in ['previous_paged_step','held_bucket_charged',*[k for k in funcs if k.startswith('resident_')],*(['flex_resident','flex_canvas_copy'] if a.flex else [])]:
    got=funcs[name]();rel=float((got.float()-expected.float()).norm()/expected.float().norm());errors[name]=rel
    if not torch.isfinite(got).all() or rel>.01:raise RuntimeError('consumer parity '+name+' '+str(rel))
   torch.cuda.synchronize();fns,owners=graph_functions({k:v for k,v in funcs.items() if k!='held_bucket_charged'})
   # CUDA graph replay itself cannot be nested inside another capture. Time
   # the already warmed bucket directly, charging Q/output copies and launches.
   fns['held_bucket_charged']=funcs['held_bucket_charged'];cap=captures[0];timing=timed(fns,rounds=a.rounds,repeats=30)
   host=host_timing(funcs,rounds=a.rounds)
   assert captures[0]==cap;after=checkgpu()
   row=dict(keys=nk,target_sparsity=sparsity,global_eligible_tiles=keep.numel(),global_kept_tiles=int(keep.sum()),actual_sparsity=1-float(keep.float().mean()),timing=timing,host_module_timing=host,qualification_relative_l2=errors,before_gpu=before,after_gpu=after,captures_during_timing=0)
   rows.append(row);(a.out/'receipt.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2)+'\n');print(json.dumps(dict(keys=nk,sparsity=row['actual_sparsity'],us={k:round(v['median_ms']*1000,2) for k,v in timing.items()})),flush=True)
 (a.out/'receipt.json').write_text(json.dumps(dict(status='complete_synthetic',rows=rows,source_sha256=hashes,reservation_seconds=time.monotonic()-start,torch=torch.__version__,cuda=torch.version.cuda),indent=2)+'\n')
if __name__=='__main__':main()
