"""Prepare a new frozen multi-benchmark attempt using the best V2 library."""
from pathlib import Path
import os
import json,hashlib,shutil,datetime
from scripts.optimized_refresh_campaign import ROOT,MODEL,write,sha,utc

def prepare(name,index=None):
 old=ROOT/'results/optimized_refresh_benchmarks_20261021/attempt004'
 study=ROOT/'results/optimized_refresh_benchmarks_20261022'/name;study.mkdir(parents=True,exist_ok=False)
 ex=study/'private/execution';shutil.copytree(old/'private/execution',ex,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for rel in ['experiments/numerical_qk_reuse/resident_sparse.py','experiments/numerical_qk_reuse/resident_execution_runtime.py','experiments/numerical_qk_reuse/refresh_v31_fast.py','experiments/numerical_qk_reuse/optimized_refresh_runtime.py','experiments/numerical_qk_reuse/v2_online_warpspec.py','experiments/numerical_qk_reuse/v2_online_sm90.py','experiments/numerical_qk_reuse/v2_online_pipeline.py','experiments/numerical_qk_reuse/fused_value_pack.py','experiments/numerical_qk_reuse/v31_value_projection.py','scripts/optimized_refresh_worker.py','scripts/optimized_refresh_campaign.py','scripts/optimized_refresh_native_campaign.py','scripts/optimized_refresh_report.py','scripts/v31_v2_local_threshold_sweep.py','scripts/v31_humaneval_cfgdet_base_worker.py','scripts/v31_humaneval_repro_policy_worker.py']:
  shutil.copyfile(ROOT/rel,ex/rel)
 for label in ['pools','schedules','calibration_pools']:
  if label=='schedules':shutil.copytree(old/'private'/label,study/'private'/label)
  else:(study/'private'/label).symlink_to((old/'private'/label).resolve(),target_is_directory=True)
 # These are newly created schedule copies, never frozen historical inputs.
 # Bound recovery to 128 requests while amortizing the minute-long engine load.
 for folder in (study/'private/schedules').iterdir():
  all_cells=json.loads((folder/'cells.json').read_text())
  for shard in folder.glob('shard???.json'):shard.unlink()
  for start in range(0,len(all_cells),128):
   write(folder/f'shard{start//128+1:03d}.json',all_cells[start:start+128])
 if index is not None:
  cells=[]
  for path in (study/'private/schedules/humaneval').glob('shard*.json'):
   cells.extend([x for x in json.loads(path.read_text()) if x['index']==index])
  if len(cells)!=1:raise RuntimeError('one known diagnostic index required')
  write(study/'private/schedules/humaneval/smoke.json',cells)
 library=ROOT/'results/sparse_execution_20261021/attempt002/build/200354cb919fb6b8/v2_online_sm90.so'
 kernel=study/'private/kernel';kernel.mkdir(parents=True)
 for source in [library,library.parent/'build_receipt.json',ROOT/'results/sparse_execution_20261021/attempt018/qualified_source/source_manifest.json']:
  shutil.copyfile(source,kernel/source.name)
 (study/'private/cache').mkdir(parents=True)
 spec='''# Optimized V2/V31 native panel, 2026-10-22\n\nUse the best qualified SM90 interior/tail online V2 library from sparse_execution attempt018. Native V2 uses fused current-V Gaussian32 projection and sketch packing, online full-QK/softmax/risk, conditional full PV, and compact-mask construction. The refresh kernel is unchanged between audit and clean passes.\n\nUse the resident direct-paged FA4 consumer on V2 and V31 held/carry steps. Held calls do no selector work, list construction, or K/V copy. Keep graph-bucket capture disabled for the timed panel; the worker's ordinary model warmup precedes measured cells. V31 uses FA4 score observation and fused top-k, with direct-paged refresh above 4K keys.\n\nCompare V2, V31, corrected FULL dense FA4, PIECEWISE dense FA4, and all-kept FA4 under the native sampler. Thinking is on; canvas=256, maximum denoising calls=48, entropy bound=0.1, confidence=0.005, stability=1, temperature=0.8 to 0.4. GLOBAL is sparse; LOCAL and prefill remain dense.\n\nV31 budgets: AIME26 K128 (30 questions x seeds 42/43/44), MATH500 K128 (500 x seed 42), HumanEval K128 (164 x seed 42), LongBench-v2 K8192 (503 x seed 42). Calibrate V2 separately per workload against actual held-phase GLOBAL eligible/kept Q128 x KV64 tile counts. Target held physical sparsity is 40-50 percent. Preserve every calibration candidate and report refresh, carried, first-canvas, held, and whole-request work separately.\n\nGate native panel launch on synthetic parity/changing-input graph checks, native audit/clean trajectory agreement, exact kernel/selector receipts, zero extra model forwards, zero LOCAL skips, and zero captures or JIT events in clean timing. Keep audit and clean timing as separate passes. Score with the official benchmark scorers and report paired/item-clustered uncertainty, accuracy, N, N/C, T, S/N, S, W, refresh/held latency, physical tile work, and source/runtime identities.\n'''
 (study/'spec.md').write_text(spec)
 p=json.loads((old/'protocol.json').read_text())
 p.update(created_utc=utc(),source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),execution_sha256={str(x.relative_to(ex)):sha(x) for x in ex.rglob('*') if x.is_file()},spec_sha256=sha(study/'spec.md'),purpose='gated full multi-benchmark comparison with best V2 interior/tail library',resident_sparse=True,resident_execution_options={'RESIDENT_SPARSE_EXECUTION':'1','RESIDENT_SPARSE_GRAPH_BUCKETS':'0','RESIDENT_V31_PAGED_REFRESH':'1'},kernel_sha256=sha(kernel/'v2_online_sm90.so'),kernel_source_manifest_sha256=sha(kernel/'source_manifest.json'))
 p['calibration']=dict(output_cap='native full request generation budget',strata=4,objective='actual held-phase GLOBAL physical Q128 x KV64 eligible/kept tiles over full native calibration requests',target_sparsity=[0.40,0.50],center_sparsity=0.45,candidate_thresholds=[0.05,0.1,0.2,0.4,0.8,0.025,0.0125,0.0],refinements=8,accuracy_used=False)
 inventory=ROOT/'results/v31_value_v2_equal_work_20261011/model_source_fingerprints.json'
 shutil.copyfile(inventory,study/'model_source_fingerprints.json')
 for entry in json.loads(inventory.read_text())['files']:
  source=MODEL/entry['file']
  if source.stat().st_size!=entry['bytes']:raise RuntimeError('model size identity drift')
  if sha(source)!=entry['sha256']:raise RuntimeError('model content identity drift')
 p['model_inventory_sha256']=sha(study/'model_source_fingerprints.json')
 p['qualification']=dict(universal_request_jit_counter=True,per_input_warm_request=True,counter_scope='all arms; request timing only; no startup or warm request cost included in clean S/W',warm_request_cost_in_resource_receipt=True)
 for suite in ['aime26','math500','humaneval']:p['suites'][suite]['mage_k']=128
 p['suites']['longbench_v2']['mage_k']=8192
 with (study/'spec.md').open('a') as specfile:specfile.write('\nCalibration uses the full native generation budget on four deterministic prompt-length strata per workload. The same calibration questions remain in mandatory full coverage and are disclosed as transductive. Every measured request is preceded by a separate untimed warm request with the same input and seed; all warm request cost remains in resource accounting. Universal request JIT events and actual CUDA capture counters apply to every arm, including native dense and resident all-kept. All-kept FA4 uses the same resident direct-paged consumer with full physical support. No final-panel launch unless calibration held GLOBAL physical sparsity is within 40-50 percent.\n')
 p['arms']=['v2_online_optimized','v31_optimized','dense_full_fix51994','dense_piecewise','allkept_fa4']
 p['source_identity_note']='base Git commit plus exact frozen execution hashes; current study integration changes included in execution hashes'
 p['sharding']=dict(max_requests_per_shard=128,reason='amortize engine startup while bounding recovery; native per-request timing and order unchanged')
 p['v31_selector']=dict(dispatch='fused through 512 KV64 tiles; bounded row-LSE, tile-score, compact top-k above 512',
    warmup='prefix tiles never exceed floor((max_model_len-query_tokens)/64); compile keys ignore runtime-only contiguous prefix stride',
    accounting='all selector launches and temporary tensors charged to refresh; no held selector')
 with (study/'spec.md').open('a') as specfile:specfile.write('\nV31 selector storage is bounded at long contexts: above 512 KV64 tiles, row normalizers and tile scores precede the unchanged deterministic fixed-budget top-k/list construction. All launches and temporary tensors are charged to refresh. Attention observation/PV kernels and the short-context fused selector are unchanged. Warm-up never synthesizes a prefix exceeding the configured model context. This corrects the preserved attempt005 qualification OOM; it does not reuse that attempt as a completed panel.\n')
 p['spec_sha256']=sha(study/'spec.md')
 p['input_sha256']={str(x.relative_to(study)):sha(x) for label in ['pools','schedules','calibration_pools'] for x in (study/'private'/label).rglob('*') if x.is_file()}
 if index is not None:
  p['input_sha256']['private/schedules/humaneval/smoke.json']=sha(study/'private/schedules/humaneval/smoke.json')
  p['diagnostic_qualification']=dict(index=index,previously_examined=True,accuracy_panel=False,workload_calibration=False)
 from scripts.v31_v2_local_threshold_sweep import runtime_identity
 p['runtime']=runtime_identity()
 write(study/'protocol.json',p);write(study/'state.json',dict(status='prepared',updated_utc=utc()))
 print(study)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--name',default='native002');p.add_argument('--humaneval-index',type=int);a=p.parse_args();prepare(a.name,a.humaneval_index)
