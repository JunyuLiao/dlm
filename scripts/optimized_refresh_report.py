"""Sanitized paired/item-clustered comparison for the optimized-refresh panel."""
from pathlib import Path
import json
import math
import statistics
import numpy as np
from scripts.v31_official import cell_key

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def paired_report(out,scores,clean,audit):
 ref='v31_optimized';rng=np.random.default_rng(20261021);paired={}
 keys=sorted(scores[ref]);clusters=sorted(set('|'.join(k.split('|')[:2]) for k in keys));groups={c:[k for k in keys if '|'.join(k.split('|')[:2])==c] for c in clusters}
 cleanmap={arm:{cell_key(x):x for x in rows} for arm,rows in clean.items()}
 for arm in scores:
  if set(scores[arm])!=set(keys):raise RuntimeError('unpaired score coverage')
  d=np.array([statistics.mean(int(scores[arm][k])-int(scores[ref][k]) for k in groups[c]) for c in clusters])
  # Arithmetic mean accuracy differences; paired geometric request/step ratios.
  logs={metric:np.array([statistics.mean(math.log(cleanmap[arm][k][field]/cleanmap[ref][k][field]) for k in groups[c]) for c in clusters]) for metric,field in [('request_W','wall_s'),('generation_S','decode_s'),('denoise_N','denoise_forwards')]}
  boot=rng.integers(0,len(clusters),size=(10000,len(clusters)))
  ac=np.mean(d[boot],axis=1);entry=dict(paired_cells=len(keys),problem_clusters=len(clusters),accuracy_difference_pp=float(d.mean()*100),accuracy_difference_ci95_pp=[float(x*100) for x in np.percentile(ac,[2.5,97.5])],arm_vs_reference=ref)
  for metric,v in logs.items():entry[metric]=dict(geometric_ratio=float(np.exp(v.mean())),ci95=[float(x) for x in np.exp(np.percentile(np.mean(v[boot],axis=1),[2.5,97.5]))])
  paired[arm]=entry
 write(out/'paired_vs_v31.json',paired)

def write_suite_markdown(out,suite,summary,cal):
 lines=[f'# {suite}: optimized refresh comparison','',f"V2 threshold {cal['threshold']:.12g}; calibration absolute first-canvas kept-work ratio {cal['absolute_work_ratio']:.5f}; matched={cal['matched']}. Calibration uses computation only. All timings below are the separate clean pass.",'','| Method | Correct | Accuracy | Mean W (s) | N/C | S/N (ms) | GLOBAL sparsity | Mean refresh (audit ms) |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for arm,x in summary.items():
  refresh=f"{x['mean_refresh_gpu_ms']:.3f}" if x['mean_refresh_gpu_ms'] is not None else '—'
  lines.append(f"| {arm} | {x['correct']}/{x['cells']} | {100*x['accuracy']:.2f}% | {x['mean_W']:.4f} | {x['N_per_C']:.3f} | {1000*x['S_per_N']:.3f} | {100*x['global_sparsity']:.2f}% | {refresh} |")
 ref=summary['v31_optimized'];v2=summary['v2_online_optimized'];ratio=v2['first_canvas_kept']/ref['first_canvas_kept'];frac=(1-v2['first_canvas_sparsity'])-(1-ref['first_canvas_sparsity'])
 lines+=['',f'Full-panel absolute first-canvas kept-work V2/V31 ratio: {ratio:.5f}; retained-fraction difference {frac*100:+.4f}pp. This does not match total FLOPs: V2 repeats QK/routing across four channel groups and charges projection/packing.','', 'LOCAL/prefill stay dense. Inherited held/carry consumer and native adaptive stopping are preserved. Refresh-event measurements are audit-only; request/generation timings are clean. S/N is an amortized generation cost, not a measured single-forward price. Paired problem-clustered confidence intervals are in paired_vs_v31.json. Calibration and previously examined questions are not held-out evidence.']
 (out/'summary.md').write_text('\n'.join(lines)+'\n')

def campaign_report(study):
 from scripts.optimized_refresh_campaign import SUITES,read,sha,utc
 rows={};reservations=[]
 for suite in SUITES:
  p=study/f'analysis/{suite}/attempt001';assert (p/'analysis_complete.json').exists();rows[suite]=read(p/'summary.json')
 for p in study.rglob('receipt.json'):
  try:x=read(p)
  except Exception:continue
  if 'worker_reservation_seconds' in x:reservations.append(dict(attempt=str(p.parent.relative_to(study)),seconds=x['worker_reservation_seconds']))
 write(study/'summary.json',rows);write(study/'resource_accounting.json',dict(worker_reservation_seconds=sum(x['seconds'] for x in reservations),workers=reservations,meaning='process elapsed reservations incl loading/JIT/warmup; not active GPU time'))
 lines=['# Optimized V2/V31 and dense baselines','', 'Complete native comparisons with current optimized kernels, thinking-on, one H100. AIME26 uses seeds42/43/44; MATH500, HumanEval164 and LongBench-v2 full503 use seed42. Fixed V31 prefix budgets:1728/1728/128/8192 tokens respectively. V2 thresholds were selected only from first-canvas computation work.','']
 for suite in SUITES:lines.append(f'- [{suite} results](analysis/{suite}/attempt001/summary.md)')
 lines+=['','Dense FULL includes fix51994; matched PIECEWISE and all-kept FA4 are also retained. V2 is an experimental online sparse refresh output, with full PV skipped after full QK/softmax/Gaussian32 projected PV. The inherited Yuhan cross-step reuse and Junyu value-direction router remain separately attributed. Read each suite receipt for actual work matching and confidence intervals.']
 (study/'summary.md').write_text('\n'.join(lines)+'\n')
 write(study/'final_complete.json',dict(status='complete',utc=utc(),protocol_sha256=sha(study/'protocol.json'),summary_sha256=sha(study/'summary.json'),all_suites=list(SUITES),native_sampler_unchanged=True))
