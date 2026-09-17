"""User-requested scheduling gate; frozen algorithms and final contracts unchanged.

Run only full_centered_s50/s75, then report the720-result reference/baseline
subset. Projected candidates remain paused. This module is deliberately outside
the frozen focused package; it changes scheduling/report scope, not inference.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_jl_focused import protocol as protocol, report as report_backend
from experiments.diffusion_gemma_jl_focused.gate import require as require_tests
from experiments.diffusion_gemma_jl_focused.reuse import dispatch
from experiments.diffusion_gemma_jl_output_aware import runner
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from experiments.diffusion_gemma_value_aware_followup.run import gpu_idle
from experiments.diffusion_gemma_value_aware.scientific_report import threshold_text
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append

ROOT=protocol.ROOT
MODULE='experiments.diffusion_gemma_jl_reference_first'
LABELS=('full_centered_s50','full_centered_s75')
BASELINES=('dense','blasst_original_s50','blasst_aggressive_s50','mass_s50',
           'blasst_original_s75','blasst_aggressive_s75','mass_s75')
REPORT_LABELS=BASELINES+LABELS
read,sha,frozen_write=protocol.read,protocol.sha,protocol.frozen_write


def control(root):
    return root/'reference_first'


def scheduling_contract(root,freeze=False):
    setup=protocol.prepare(root);contract=protocol.execution(root)
    require_tests(root,contract)
    paths=[Path(__file__),Path('tests/test_jl_reference_first.py'),control(root)/'tests.xml',
           root/'execution_contract.json',root/'setup.json',root/'test_gate.json']
    paths += [root/'final_configs'/f'{label}.json' for label in REPORT_LABELS]
    suites=ET.parse(control(root)/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=3,failures=0,errors=0,skipped=0):
        raise ValueError(f'Reference-first scheduling/report tests incomplete: {counts}')
    data=dict(schema='reference_first_schedule_v1',parent_execution_fingerprint=contract['fingerprint'],
        labels=list(LABELS),expected_new_reference=160,expected_reference_with_baselines=720,
        sources={str(p):sha(p.read_bytes()) for p in paths},
        decision='Full80 matched examples at both targets before deciding whether to resume projected methods. No automatic projected launch; inconclusive small differences are not declared failure.',
        scientific_change=False,threshold_refitting=False,projected_methods_paused=True)
    check_sources(data['sources'])
    path=control(root)/'scheduling_contract.json'
    if path.exists() and read(path)!=data:raise ValueError('Frozen reference-first scheduler changed')
    if freeze:frozen_write(path,data)
    return setup,contract,data


def link(path,target):
    """Read-only report view; raw generations remain in the original bundle."""
    target=target.resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():
        if path.resolve()!=target:raise ValueError('Report alias changed')
    elif path.exists():raise ValueError('Refuse to overwrite report-view input')
    else:path.symlink_to(target,target_is_directory=target.is_dir())


def report_view(root,setup,contract):
    view=control(root)/'report';view.mkdir(parents=True,exist_ok=True)
    selected=deepcopy(setup);selected['conditions']=list(REPORT_LABELS)
    for label in REPORT_LABELS:
        link(view/'final_configs'/f'{label}.json',root/'final_configs'/f'{label}.json')
        stage='dense' if label=='dense' else 'final'
        link(view/stage/label,root/stage/label)
        if label!='dense':
            for benchmark in ('aime26','longbench_v2'):
                link(view/'policies'/benchmark/f'{label}.json',root/'policies'/benchmark/f'{label}.json')
    for name in ('validation','shared_state_index.json'):
        link(view/name,root/name)
    allowed={'full_centered',*protocol.BASELINES}
    index=[]
    for entry in read(root/'shared_diagnostics_index.json'):
        path=Path(entry['path'])
        if sha(path.read_bytes())!=entry['sha256']:raise ValueError('Shared diagnostic source changed')
        if read(path)['identity']['name'] in allowed:index.append(entry)
    frozen_write(view/'shared_diagnostics_index.json',index)
    frozen_write(view/'scope.json',dict(conditions=list(REPORT_LABELS),expected=720,
        source_setup=str(root/'setup.json'),setup_sha256=sha((root/'setup.json').read_bytes()),
        projected_shards_excluded=True,parent_execution_fingerprint=contract['fingerprint']))
    return view,selected


def write_reference_report(view,setup,rows,tasks,compared,diag,audit,policies):
    # This hook changes only the declared report scope/completion requirement.
    # Pairing, official scoring, raw audits and metrics remain the existing ones.
    audit['expected']=720
    audit['complete']=audit['completed']==720 and not audit['missing'] and not audit['violations']
    display=[dict(benchmark=r['benchmark'],split=r['split'],condition=r['condition'],target=100*r['target'],
        threshold=threshold_text(r),correct=f"{r['correct']:g}/{r['count']}",accuracy=100*r['accuracy'],
        dense_delta_pp=100*r['delta'],ci95_pp=[100*x for x in r['paired_ci95']],
        whole=100*r['overall_physical_sparsity'],global_s=100*r['global_physical_sparsity'],
        local_s=100*r['local_physical_sparsity'],mass=100*r['overall_mass'],agreement=100*r['token_agreement'],
        local_operator_error=r['overall_relative_error'],unparsed=r['unparsed_answers']) for r in rows]
    report_backend.csv_write(view/'main_results.csv',display)
    text='# Full-dimensional centered reference: feasibility gate\n\n'
    text+=f"Status: {'COMPLETE' if audit['complete'] else 'INCOMPLETE'}, {audit['completed']}/720 audited outputs:160 new reference outputs plus560 reused baselines. The projected sweep is paused, not completed or scientifically excluded.\n\n"
    text+='The user requested full-dimensional centered routing first. Both50% and75% targets use all30 AIME26 and the same50 LongBench v2 IDs sampled from the completed100-run cohort. AIME output budget2048; LongBench actual pinned NeMo prompt/scorer and4096 tokens,32K input cap. Domain quotas17 Single-Document QA,13 Multi-Document QA,8 Long In-context Learning,5 Code Repository Understanding,4 Long-dialogue History Understanding,3 Long Structured Data Understanding.43/50 LongBench contexts inherit source truncation. Previously examined data, not fresh confirmation; AIME includes6 calibration questions and separately reports noncalibration24.\n\n'
    text+='## Unchanged operator and calibration\n\n'
    text+='The reference uses attention-weighted full-dimensional block means: online candidate mass fraction times the difference from the retained running output, divided by valid-KV RMS. Strict worst-valid-row128×64 physical-tile threshold, ties/first support retained, prefix+canvas eligible, native structural masks/GQA; skipped blocks do not update retained state. Actual attention uses originalV and ordinary renormalization with no compensation. It removes projection distortion but is not a globally optimal selector or a proven upper bound on projected routing accuracy.\n\n'
    text+='BF16 DiffusionGemma revision '+setup['revision']+'; seed42, canvas256, up to48 denoising steps, thinkingFalse, unchanged native0.4–0.8 sampling schedule (temperature0 is a sentinel, not greedy). Full-centered calibrated local/global scalar thresholds are reused unchanged from the completed JL calibration:6 questions/benchmark,512-token verification; LongBench calibration spans3 domains. Cached LongBench baselines retain their12-question,6-domain calibration. Calibration-coverage asymmetry and sparsity transfer are limitations; no thresholds are tuned on these final scores.\n\n'
    text+='Original BLASST caps lambda at1; aggressive BLASST permits>1 with the preceding-seen-maximum convention. Existing inverse-valid-KV-length policies and unattainable-at1 annotations are retained. Cached mass-only uses the max-based candidate mass bound, not mass_exact; differences versus it alone do not isolate direction from mass estimation. Thresholds and raw traces remain linked to their original execution fingerprints.\n\n'
    cols=('condition','target','threshold','correct','accuracy','dense_delta_pp','whole','global_s','local_s','mass','agreement','local_operator_error','unparsed')
    for b,split in (('aime26','full'),('aime26','noncalibration24'),('longbench_v2','full')):
        text+=f'## {b}: {split}\n\n'+report_backend.table([r for r in display if r['benchmark']==b and r['split']==split],cols)+'\n\n'
    text+='## Reference versus actually measured baselines\n\n'
    text+=report_backend.table([r for r in compared if r['split']=='full'],
        ('benchmark','candidate','reference','accuracy_delta','paired_ci95','overall_sparsity_gap','global_sparsity_gap','local_sparsity_gap','within3pp_all_types'))+'\n\n'
    text+='Comparisons use the nearest observed actual-sparsity point, not merely equal targets. Gaps/deltas in this comparison table are fractions. Paired prompt-bootstrap95% intervals are exploratory and not multiplicity-corrected. A small or inconclusive difference is not categorical failure. If the full-dimensional reference consistently fails to improve a comparable actual-sparsity baseline, that weakens the motivation for spending on projected variants; it does not mathematically prove every projection must fail. Projected runs remain paused for an evidence-based decision.\n\n'
    text+='## LongBench per-domain scores\n\n'+report_backend.table([dict(domain=r['task'],condition=r['condition'],correct=f"{r['correct']:g}/{r['count']}",accuracy=100*r['accuracy']) for r in tasks if r['benchmark']=='longbench_v2'],('domain','condition','correct','accuracy'))+'\n\n'
    text+='Sparsity=SUM(skipped eligible physical tiles)/SUM(eligible tiles), overall/global/local separately; not averaged percentages. Dense prefill is excluded. Retained mass/local error compare each run’s corresponding QKV; shared_operator.json separately compares methods on identical saved dense calibration QKV, with sampling coverage and error=sqrt(sum error²/sum dense-output²). Token agreement compares all positions after divergence, counting missing/extra tokens as disagreements. No generation divergence is mislabeled local operator error.\n\n'
    text+='Raw per-sample, per-layer/head/step/region metrics, domain/subtask scores, retained mass, error, token agreement, exact-sequence matches, paired uncertainty and work accounting are exported. Full-dimensional routing is an expensive reference. Native PV remains a dense-shaped masked matmul; no hardware speedup versus FlashAttention is claimed.\n\n'
    text+=f"Audit missing entries:{len(audit['missing'])}; violations:{len(audit['violations'])}. Independent report regeneration uses completed shards only, never inference. The39 earlier projected outputs and all prior logs are preserved outside this scoped report.\n\n"
    text+='Reproduce: `CUDA_VISIBLE_DEVICES=\'\' PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_reference_first report` (or `verify`).\n'
    (view/'report.md').write_text(text)


def regenerate(root=ROOT):
    setup,contract,schedule=scheduling_contract(root)
    view,selected=report_view(root,setup,contract)
    with patch.object(report_backend,'prepare',lambda unused:selected),\
         patch.object(report_backend,'execution',lambda unused:contract),\
         patch.object(report_backend,'PROJECTED',{'full_centered':protocol.PROJECTED['full_centered']}),\
         patch.object(report_backend.common,'shared_seed_summary',lambda unused:([],{})),\
         patch.object(report_backend,'write_report',write_reference_report):
        audit=report_backend.regenerate(view)
    audit['sources'].update(schedule['sources'])
    audit['sources'][str(control(root)/'scheduling_contract.json')]=sha((control(root)/'scheduling_contract.json').read_bytes())
    audit['sources'][str(view/'scope.json')]=sha((view/'scope.json').read_bytes())
    _write(view/'audit.json',audit)
    return audit


def verify(root=ROOT):
    view=control(root)/'report';before=read(view/'audit.json')
    if not before['complete'] or before['completed']!=720:raise ValueError('Require complete720-result reference report')
    check_sources({str(view/p):v for p,v in before['artifacts'].items()})
    after=regenerate(root)
    if before!=after:raise ValueError('Reference-first raw-only regeneration changed artifacts/audit')
    proof=dict(passed=True,completed=720,inference_performed=False,
        audit_sha256=sha((view/'audit.json').read_bytes()),artifact_count=len(after['artifacts']))
    _write(view/'regeneration_verification.json',proof)
    return proof


def phase(root,stage,**details):
    _write(control(root)/'phase.json',dict(stage=stage,started=time.time(),**details))
    print('reference-first phase',stage,details,flush=True)


def work(root):
    setup,contract,schedule=scheduling_contract(root,freeze=True)
    conditions={label:read(root/'final_configs'/f'{label}.json') for label in LABELS}
    for label,c in conditions.items():
        if c['name']!='full_centered' or c['config']!={'family':'identity'} or c['unavailable_benchmarks']:
            raise ValueError('Reference-first must use complete existing identity-projection conditions')
        check_sources(c['sources'])
    torch.backends.cuda.matmul.allow_tf32=False
    phase(root,'load_model')
    adapter=create_adapter('diffusion_gemma',protocol.MODEL,device='cuda',precision='bfloat16',revision=protocol.REVISION).load()
    original=runner.generate
    def tagged(*args,**kwargs):
        out=original(*args,**kwargs)
        path=control(root)/'scheduling_contract.json'
        out['scheduling_source']=dict(path=str(path),sha256=sha(path.read_bytes()))
        return out
    failures=[]
    with dispatch(),patch.object(runner,'generate',tagged):
        for label in LABELS:
            c=conditions[label];phase(root,'final',condition=label,expected=80)
            for row in sorted(setup['final'],key=lambda r:(len(r['prompt_tokens']),r['id'])):
                try:
                    runner.cached(adapter,root,row,'final',label,'full_centered',c['config'],c['thresholds'][row['benchmark']],contract)
                except Exception:
                    failure=dict(stage='final',condition=label,id=row['id'],traceback=traceback.format_exc())
                    failures.append(failure);_append(control(root)/'failures.jsonl',failure);print(failure,flush=True)
                    torch.cuda.empty_cache()
    del adapter;torch.cuda.empty_cache();phase(root,'reference_report')
    audit=regenerate(root)
    if audit['complete']:
        phase(root,'independent_report_verification');verify(root)
    _write(control(root)/'terminal.json',dict(complete=audit['complete'],completed=audit['completed'],expected=720,
        failed_attempts=len(failures),projected_methods_paused=True,finished=time.time()))
    phase(root,'await_reference_decision',complete=audit['complete'])


def supervise(root):
    folder=control(root);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_idle()
        with (folder/'run.log').open('a',buffering=1) as log:
            child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'work','--output',str(root)],stdout=log,stderr=subprocess.STDOUT)
            _write(folder/'job.json',dict(pid=child.pid,supervisor_pid=os.getpid(),started=time.time()))
            while True:
                try:code=child.wait(timeout=1800);break
                except subprocess.TimeoutExpired:
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
                    state={k:read((root if k=='progress' else folder)/f'{k}.json') if ((root if k=='progress' else folder)/f'{k}.json').exists() else None for k in ('phase','progress')}
                    _append(folder/'monitor.jsonl',dict(time=time.time(),pid=child.pid,gpu=gpu.stdout.strip(),disk_free_bytes=shutil.disk_usage(root).free,**state))
            _write(folder/'supervisor_terminal.json',dict(exit_code=code,pid=child.pid,finished=time.time()))
            if code:raise SystemExit(code)


def launch(root):
    scheduling_contract(root,freeze=True);gpu_idle()
    with (control(root)/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'supervise','--output',str(root)],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid,reference_runs=160,projected_methods_paused=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('launch','supervise','work','report','verify'))
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args()
    if args.command=='report':print(json.dumps(regenerate(args.output)))
    elif args.command=='verify':print(json.dumps(verify(args.output)))
    else:globals()[args.command](args.output)
