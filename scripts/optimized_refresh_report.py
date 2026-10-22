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
 if cal.get('status')=='fixed_threshold_by_user':
  threshold_text=f"V2 threshold {cal['threshold']:.12g}, fixed by explicit user instruction without new calibration. Provenance: {cal['provenance']['source']}. Current physical work is measured below; prior matching does not establish current work equivalence or 40-50% held sparsity."
 else:
  threshold_text=f"V2 threshold {cal['threshold']:.12g}; held calibration sparsity {100*cal['held_sparsity']:.2f}% from {cal['held_eligible_tiles']} eligible and {cal['held_kept_tiles']} kept physical GLOBAL tiles across {cal['calibration_cells']} calibration cells. Target range was 40-50%; in range={cal['in_target_range']}. Calibration uses computation only."
 lines=[f'# {suite}: optimized refresh comparison','',threshold_text+' Timings below are the separate clean pass.','','| Method | Correct | Accuracy | Mean W (s) | N/C | S/N (ms) | GLOBAL sparsity | Held sparsity | Refresh (audit ms) | Held (audit ms) |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for arm,x in summary.items():
  refresh=f"{x['mean_refresh_gpu_ms']:.3f}" if x['mean_refresh_gpu_ms'] is not None else 'n/a'
  held=x.get('phase_work',{}).get('held',{}).get('sparsity')
  held_text=f'{100*held:.2f}%' if held is not None else 'n/a'
  held_ms=x.get('resident_phase_timing',{}).get('held',{}).get('mean_gpu_ms')
  held_time=f'{held_ms:.3f}' if held_ms is not None else 'n/a'
  lines.append(f"| {arm} | {x['correct']}/{x['cells']} | {100*x['accuracy']:.2f}% | {x['mean_W']:.4f} | {x['N_per_C']:.3f} | {1000*x['S_per_N']:.3f} | {100*x['global_sparsity']:.2f}% | {held_text} | {refresh} | {held_time} |")
 v2=summary['v2_online_optimized'];phases=v2.get('phase_work',{})
 phase_summary='; '.join(f"{name}: {row['kept_tiles']}/{row['eligible_tiles']} kept ({100*row['sparsity']:.2f}% sparse)" for name,row in sorted(phases.items()) if row['eligible_tiles'])
 calibration_note=('This workload uses a previously examined/fixed threshold and is not fresh calibration or held-out confirmation.' if cal.get('status')=='fixed_threshold_by_user' else 'The full mandated coverage includes the four transductive calibration questions; their prior exposure is disclosed and they are not held-out evidence.')
 lines+=['',f"V2 physical GLOBAL work by phase: {phase_summary}. First-canvas work is {v2['first_canvas_kept']}/{v2['first_canvas_eligible']} kept ({100*v2['first_canvas_sparsity']:.2f}% sparse); whole-request work is {v2['global_kept_tiles']}/{v2['global_eligible_tiles']} kept ({100*v2['global_sparsity']:.2f}% sparse).",'', 'LOCAL and prefill stay dense. Inherited native adaptive stopping and the resident direct-paged held consumer are preserved. Refresh and held event timing are audit-only; request/generation timing is clean. S/N is an amortized per-step cost, not a measured single-forward price. Paired problem-clustered confidence intervals are in paired_vs_v31.json. '+calibration_note]
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
 lines=['# Optimized V2/V31 and dense baselines','', 'Complete native comparisons with the frozen optimized kernels, thinking-on, one H100. AIME26 uses 30 questions x seeds42/43/44; MATH500 uses all500 questions, HumanEval all164 tasks, and LongBench-v2 all503 problems at seed42. Fixed V31 prefix budgets: K128/K128/K128/K8192 respectively. V2 calibrates AIME26/MATH500 to actual held GLOBAL physical sparsity; HumanEval reuses its saved work-matched threshold and LongBench-v2 uses 0.021 by user instruction, without new calibration.','']
 for suite in SUITES:lines.append(f'- [{suite} results](analysis/{suite}/attempt001/summary.md)')
 lines+=['','Dense FULL includes fix51994; matched PIECEWISE and all-kept FA4 are retained. V2 is an experimental online sparse refresh output, with full PV skipped after full QK/softmax/Gaussian32 projected PV. The inherited Yuhan cross-step reuse and Junyu value-direction router remain separately attributed. Each suite report lists per-phase physical work, accuracy, clean latency, audit event latency, paired uncertainty, and calibration evidence.']
 (study/'summary.md').write_text('\n'.join(lines)+'\n')
 write(study/'final_complete.json',dict(status='complete',utc=utc(),protocol_sha256=sha(study/'protocol.json'),summary_sha256=sha(study/'summary.json'),all_suites=list(SUITES),native_sampler_unchanged=True))
