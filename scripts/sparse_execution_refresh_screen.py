"""Frozen synthetic refresh/component matrix; never a benchmark accuracy panel.

All V2 thresholds are calibrated on actual online masks for these synthetic
inputs. Two-pass discovery remains explicitly diagnostic. No projection or
packing is omitted from complete refresh/module variants.
"""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
 p.add_argument('--keys',type=int,nargs='+',default=[1024,4096,8192,16384]);p.add_argument('--sparsities',type=float,nargs='+',default=[.4,.5,.6]);p.add_argument('--rounds',type=int,default=5);p.add_argument('--schedule',type=int,default=2);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();frozen=a.out/'source';frozen.mkdir()
 files=[Path(__file__),*[Path('experiments/numerical_qk_reuse')/s for s in ['v2_online_warpspec.py','v2_online_pipeline.py','v2_online_sm90.py','fused_value_pack.py','refresh_v31_fast.py','resident_sparse.py','v2_sparse_refresh.py','v31_value_projection.py']],Path('scripts/resident_sparse_screen.py'),Path('scripts/v2_sparse_refresh_bench.py')]
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
 for x in files:shutil.copyfile(x,frozen/x.name)
 shutil.copyfile(a.library.parent/'build_receipt.json',frozen/'build_receipt.json')
 from experiments.numerical_qk_reuse import v27_fa4
 from experiments.numerical_qk_reuse.v2_online_warpspec import PackedKernel,PackedRefresh,FusedPackedRefresh,CanvasPackedRefresh,pack_z
 from experiments.numerical_qk_reuse.v2_online_pipeline import pack_lists
 from experiments.numerical_qk_reuse.v31_value_projection import refresh as project
 from experiments.numerical_qk_reuse.fused_value_pack import project as fused_project
 from experiments.numerical_qk_reuse.resident_sparse import ResidentMask,PagedContext
 from experiments.numerical_qk_reuse.refresh_v31_fast import Observer
 from experiments.numerical_qk_reuse.v2_sparse_refresh import refresh as two_pass
 from experiments.numerical_qk_reuse.v29_paged_copy import copy_paged_kv
 from experiments.diffusion_gemma_jl_output_aware.projections import Projections
 from scripts.v2_sparse_refresh_bench import graph_functions,timed
 from scripts.resident_sparse_screen import host_timing
 kernel=PackedKernel(a.library);old=PackedRefresh(a.library,a.schedule);complete=FusedPackedRefresh(a.library,a.schedule);canvas_cached=CanvasPackedRefresh(a.library,a.schedule);observer=Observer(a.out/'private/fa4')
 captures=[0];original=torch.cuda.CUDAGraph.capture_begin
 def capture(self,*args,**kwargs):captures[0]+=1;return original(self,*args,**kwargs)
 torch.cuda.CUDAGraph.capture_begin=capture
 def gpu():
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
  if any(int(x.split(',')[0])!=os.getpid() for x in apps.splitlines() if x.strip()):raise RuntimeError('foreign GPU worker')
  return apps
 rows=[]
 for nk in a.keys:
  g=torch.Generator(device='cuda').manual_seed(nk);n=256;h=16;hk=2;d=512;pt=(nk-n)//64;scale=d**-.5
  q=torch.randn((1,h,n,d),device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn((1,hk,nk,d),device='cuda',dtype=q.dtype,generator=g);v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=g)
  matrix=Projections().get(5,hk,d,'gaussian',32,1729,q.device);z=torch.empty((1,hk,nk,32),device=q.device);norm=torch.empty((1,hk,nk),device=q.device);valid=torch.ones_like(norm,dtype=torch.bool)
  nu=project(v,matrix,z,norm,valid,0);packed=pack_z(z)
  pages=nk//128;table=torch.randperm(pages,device='cuda',generator=g).int();cache=torch.empty((pages,128,hk,2*d),device=q.device,dtype=q.dtype);key,value=cache[...,:d],cache[...,d:]
  key[table.long()]=k[0].transpose(0,1).reshape(pages,128,hk,d);value[table.long()]=v[0].transpose(0,1).reshape(pages,128,hk,d)
  copied_k=torch.empty_like(k);copied_v=torch.empty_like(v);native_q=q.transpose(1,2).contiguous().transpose(1,2)
  context=PagedContext(key,value,table,nk,splits=2);used=torch.tensor([nk],device=q.device,dtype=torch.int32);dynamic=torch.tensor([False],device=q.device)
  def kv_copy():copy_paged_kv(key,value,table,copied_k,copied_v,0,nk)
  def canvas_copy():copy_paged_kv(key,value,table,copied_k,copied_v,nk-n,n)
  def dense_native():return v27_fa4.load()(q.transpose(1,2),key,value,page_table=table[None],seqused_k=used,softmax_scale=scale,causal=True,dynamic_causal=dynamic,pack_gqa=False,num_splits=0,intra_wg_overlap=False)[0]
  for target in a.sparsities:
   before=gpu();lo,hi=0.,2.;trials=[]
   for _ in range(15):
    tau=(lo+hi)/2;chosen=kernel(q,k,v,z,nu,scale,tau,prefix_tiles=pt,schedule=a.schedule,prepacked=packed)[1];count=int(chosen.sum());trials.append(dict(threshold=tau,kept=count,eligible=chosen.numel()))
    if count>chosen.numel()*(1-target):lo=tau
    else:hi=tau
   winner=min(trials,key=lambda r:abs(r['kept']/r['eligible']-(1-target)));tau=winner['threshold']
   out,keep,*_=kernel(q,k,v,z,nu,scale,tau,prefix_tiles=pt,schedule=a.schedule,prepacked=packed);resident=ResidentMask.create(keep,splits=2,prefix_tiles=pt,audit=True)
   expected=context(q,resident,scale);relative=float((out.float()-expected.float()).norm()/expected.float().norm())
   fused=complete(q,k,v,matrix,scale,pt,tau);old_out=old(q,k,v,matrix,scale,pt,tau)
   if not torch.isfinite(out).all() or relative>.01 or not torch.equal(keep,fused[1]) or not torch.equal(out,fused[0]) or not torch.equal(out,old_out[0]):raise RuntimeError('online/fused projection qualification')
   # Synthetic matched-work K, distinct from the fixed K128 native budget.
   budget=max(64,round((nk//64*(1-target)-4))*64)
   def v2_module():
    kv_copy();result=complete(native_q.contiguous(),copied_k,copied_v,matrix,scale,pt,tau)
    return result,ResidentMask.create(result[1],splits=2)
   def v31_module():
    kv_copy();result=observer.refresh(native_q,copied_k,copied_v,scale,nk-n,128,overlap=False,num_splits=1)
    return result,ResidentMask.create(result[1],splits=2)
   def cached_refresh():return canvas_cached(q,k,v,matrix,scale,pt,tau,cache_key=5,canvas=nk,prefix=nk-n)
   funcs=dict(dense_plain=lambda:v27_fa4.dense_plain(q,k,v,scale),native_dense_paged=dense_native,allkept_fa4=lambda:v27_fa4.dense(q,k,v,scale),
    v31_k128_refresh=lambda:observer.refresh(q,k,v,scale,nk-n,128,overlap=False,num_splits=1),
    v31_matched_refresh=lambda:observer.refresh(q,k,v,scale,nk-n,budget,overlap=False,num_splits=1),
    v31_dynamic_k128_refresh=lambda:observer.refresh(q,k,v,scale,nk-n,128,overlap=False,num_splits=0,dynamic_causal=True),
    v31_direct_paged_k128_refresh=lambda:observer.refresh_paged(q,key,value,table,nk,scale,nk-n,128),
    v2_online_core=lambda:kernel(q,k,v,z,nu,scale,tau,prefix_tiles=pt,schedule=a.schedule,prepacked=packed),
    v2_allkept_core=lambda:kernel(q,k,v,z,nu,scale,0.,prefix_tiles=pt,schedule=a.schedule,prepacked=packed),
    v2_refresh=lambda:old(q,k,v,matrix,scale,pt,tau),v2_fused_pack_refresh=lambda:complete(q,k,v,matrix,scale,pt,tau),
    v2_canvas_cached_refresh=cached_refresh,
    projection_rms=lambda:project(v,matrix,z,norm,valid,0),projection_rms_fused_pack=lambda:fused_project(v,matrix,z,norm,valid),sketch_pack=lambda:pack_z(z),
    compact_list=lambda:pack_lists(keep),resident_metadata=lambda:ResidentMask.create(keep,splits=2),kv_copy_full=kv_copy,kv_copy_canvas=canvas_copy,
    held_resident=lambda:context(q,resident,scale),two_pass_diagnostic=lambda:two_pass(q,k,v,matrix,scale,pt,tau,backend='parallel',fast_scan=True),
    v2_attention_module=v2_module,v31_attention_module=v31_module)
   for f in funcs.values():
    for _ in range(2):f()
   torch.cuda.synchronize();fns,owners=graph_functions(funcs);cap=captures[0];timing=timed(fns,rounds=a.rounds,repeats=30);host=host_timing(funcs,rounds=a.rounds)
   if captures[0]!=cap:raise RuntimeError('capture during timing')
   after=gpu();row=dict(keys=nk,target_sparsity=target,actual_sparsity=1-float(keep.float().mean()),global_eligible_tiles=keep.numel(),global_kept_tiles=int(keep.sum()),threshold=tau,synthetic_v31_matched_budget=budget,calibration=trials,relative_l2=relative,fused_projection_output_bitwise_equal=True,timing=timing,host_module_timing=host,before_gpu=before,after_gpu=after,captures_during_timing=0)
   rows.append(row);(a.out/'receipt.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2)+'\n');print(json.dumps(dict(keys=nk,sparsity=row['actual_sparsity'],us={k:round(v['median_ms']*1000,2) for k,v in timing.items()})),flush=True)
 result=dict(status='complete_synthetic',rows=rows,source_sha256=hashes,library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),observer_sha256=observer.source_sha256,reservation_seconds=time.monotonic()-started,torch=torch.__version__,cuda=torch.version.cuda,captures_during_timing=0,captures_outside_timing=captures[0],schedule=a.schedule,scope='synthetic only; thresholds not native calibration',component_limits='QK/softmax/projected-PV/risk and conditional PV share one kernel and are not separately timed. All-kept core is an ablation, not an isolated PV timer. End-to-end/request measurements belong to native qualification.')
 (a.out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
