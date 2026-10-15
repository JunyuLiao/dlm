"""Generate a resumable one-GPU command queue for the full value-method matrix."""
import argparse,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--python',required=True); ap.add_argument('--model',required=True); a=ap.parse_args(); root=a.root.resolve(); out=a.out.resolve(); methods=['current_v31_control','value_v1_online_discard_mass','value_v2_online_preserve_mass','value_v3a_singleton_delete','value_v3b_approx_batch8','value_v2_online_jcgate','value_mass_gated_online_swap8','value_mass_gated_online_cgate_swap16']; jobs=[]
 cfg=json.loads((root/'config.json').read_text())
 # AIME is the short/native 9k-token panel; RULER and LongBench use the
 # long-context thresholds.  The earlier provisional queue accidentally sent
 # AIME through threshold_long, so corrected campaigns must use this mapping.
 def t(reg,suite,method): return cfg['regimes'][reg]['threshold_short' if suite in ('aime26','quick_short') else 'threshold_long'][method]
 def add(name,cmd,attempt): jobs.append({'id':name,'command':cmd,'attempt':str(attempt)})
 # AIME and RULER use the frozen 3-seed / 104-cell panels, sharded for restartability.
 for suite,manifest,cells,shards in [('aime26','private/aime26','private/aime26/cells_aime26_seed42_43_44.json',3),('ruler128k','private/ruler128k','private/ruler128k/cells_ruler128k_v33ofc_subset.json',4)]:
  for reg in ('s50','s75'):
   for method in methods:
    for shard in range(shards):
     attempt=root/suite/reg/'audit'/method/f'shard{shard+1:03d}'/'attempt001'; cmd=[a.python,'scripts/v31_value_performance_campaign.py','--python',a.python,'--model',a.model,'--suite',suite,'--manifests',str(root/manifest),'--cells',str(root/cells),'--attempt',str(attempt),'--arm',method,'--purpose','audit','--sparsity-target','50' if reg=='s50' else '75','--shard-index',str(shard),'--shards',str(shards)]
     if method!='current_v31_control': cmd += ['--threshold',str(t(reg,suite,method))]
     add(f'{suite}_{reg}_{method}_shard{shard+1:03d}',cmd,attempt)
 # Auxiliary short and LongBench pools use about 100 cells per benchmark group.
 for reg in ('s50','s75'):
  for method in methods:
   for group,cells,maxlen,mage in [('short','private/quick100/cells_short.json',16384,cfg['regimes'][reg]['short_mage_k']),('longbench','private/quick100/cells_longbench.json',32768,cfg['regimes'][reg]['long_mage_k'])]:
    attempt=root/'quick100'/reg/'audit'/f'{method}_{group}'/'attempt001'; cmd=[a.python,'scripts/v31_value_campaign.py','--python',a.python,'--model',a.model,'--manifests',str(root/'private/quick100'),'--cells',str(root/cells),'--attempt',str(attempt),'--arm',method,'--purpose','audit','--mage-k',str(mage),'--max-model-len',str(maxlen),'--gpu-memory-utilization','0.85','--result-root',str(root)]
    if method!='current_v31_control': cmd += ['--threshold',str(t(reg,'quick_short' if group=='short' else 'longbench',method))]
    add(f'quick100_{reg}_{method}_{group}',cmd,attempt)
 out.parent.mkdir(parents=True,exist_ok=True); out.write_text('\n'.join(json.dumps(j) for j in jobs)+'\n'); print(json.dumps({'jobs':len(jobs),'queue':str(out)}))
if __name__=='__main__':main()
