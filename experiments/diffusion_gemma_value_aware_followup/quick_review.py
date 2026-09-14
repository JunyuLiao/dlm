"""User-requested 25-question, 50%-target review; never starts later targets.

This is a new subset contract, not a modification of the frozen broad60 study.
The original operator, policies, raw-shard cache and single-H100 supervisor are
reused. Completed compatible outputs remain useful to either study.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from experiments.diffusion_gemma_value_aware.protocol import frozen_write
from experiments.diffusion_gemma_value_aware.evaluate import final_condition_order
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append
from .protocol import ROOT, MODEL, REVISION, prepare as base_setup, sha
from .engine import contract
from .evidence import read_result, pair, check_sources
from .execution import cached
from .final import load_contract as base_contract, reuse_equivalent
from . import jobs

DOMAINS = ('Single-Document QA', 'Multi-Document QA', 'Code Repository Understanding')
EXPECTED = {'aime26': 10, 'longbench_v2': 15}
REFERENCES = ('blasst_original_s50', 'blasst_aggressive_s50')


def destination(root):
    return root/'quick50_review'


def select_rows(rows):
    """Fixed seed-hash stratification; no predictions/cache availability inputs."""
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate source identities')
    def take(pool, n):
        if len(pool) < n:
            raise ValueError('insufficient stratum for fixed quick-review subset')
        return sorted(pool, key=lambda r: (sha('quick50/42/'+r['id']), r['id']))[:n]
    selected = []
    # Five per AIME exam, excluding all six calibration questions. The other
    # 24 were previously exposed; this is exploratory, not fresh heldout data.
    for exam in (0, 1):
        pool = [r for r in rows if r['benchmark']=='aime26' and not r['calibration']
                and (int(r['source_id'])-1)//15 == exam]
        selected += take(pool, 5)
    # Five/domain cannot be evenly split by difficulty. Alternate 3/2, 2/3,
    # 3/2 easy/hard: eight easy and seven hard overall, without score selection.
    for index, domain in enumerate(DOMAINS):
        for difficulty, n in (('easy', 2 if index==1 else 3), ('hard', 3 if index==1 else 2)):
            selected += take([r for r in rows if r['benchmark']=='longbench_v2'
                and r['task']==domain and r['difficulty']==difficulty], n)
    selected = sorted(selected, key=lambda r: (r['generation_budget'], r['id']))
    validate_rows(selected)
    return selected


def validate_rows(rows):
    if (Counter(r['benchmark'] for r in rows) != Counter(EXPECTED)
            or len({r['id'] for r in rows}) != 25
            or len({r['prompt_hash'] for r in rows}) != 25):
        raise ValueError('quick review requires exactly10 AIME and15 distinct v2 prompts')
    if Counter(r['task'] for r in rows if r['benchmark']=='longbench_v2') != Counter({d:5 for d in DOMAINS}):
        raise ValueError('quick review requires exactly five per v2 domain')
    if any(r['split']!='final' or r['calibration'] or r['seed']!=42 for r in rows):
        raise ValueError('quick review excludes calibration and preserves final seeds')


def prepare(root):
    setup=base_setup(root); execution=contract(root); broad=base_contract(root,'broad50',execution)
    dest=destination(root); dest.mkdir(parents=True,exist_ok=True)
    rows=select_rows(setup['final'])
    excluded={r['id'] for r in setup['calibration']+setup['development']}
    if excluded & {r['id'] for r in rows}:
        raise ValueError('quick review overlaps calibration/development')
    selection=dict(rows=rows,counts=EXPECTED,seed=42,selected_before_reading_scores=True,
        rule='Seed42 SHA256 order within AIME exam (5 each, exclude calibration) and v2 domain/difficulty (3/2,2/3,3/2 easy/hard)',
        aime_previously_exposed=True,source_manifest_sha256=sha((root/'final.json').read_bytes()),
        domain_counts={d:5 for d in DOMAINS},
        truncated_v2=sum(r.get('truncated',False) for r in rows if r['benchmark']=='longbench_v2'),
        preserved_settings=True)
    frozen_write(dest/'selection.json',selection)
    sources={str(root/'final_contracts'/'broad50.json'):sha((root/'final_contracts'/'broad50.json').read_bytes()),
        str(dest/'selection.json'):sha((dest/'selection.json').read_bytes())}
    for name in ('quick_review.py','quick_report.py'):
        p=Path(__file__).with_name(name); sources[str(p)]=sha(p.read_bytes())
    from experiments.diffusion_gemma_value_aware import report as base_report, report_metrics, scientific_report
    for module in (base_report,report_metrics,scientific_report):
        p=Path(module.__file__); sources[str(p)]=sha(p.read_bytes())
    data=dict(schema='quick50_user_review_v1',fingerprint=execution['fingerprint'],
        conditions=broad['conditions'],expected_per_condition=EXPECTED,expected_shards=325,
        selection=selection,sources=sources,scope='User-authorized quick25 review, superseding the running broad60 sweep',
        next_action='PAUSE for user review after complete50%; no automatic later targets or full study',
        provisional_keep_rule=dict(max_extra_wrong_per_benchmark=1,
            reference='better point accuracy of original and aggressive BLASST separately for each benchmark',
            user_specified_numeric_tolerance=False,not_statistical_equivalence=True,
            actual_budget_mismatch_must_be_disclosed=True))
    frozen_write(dest/'contract.json',data)
    frozen_write(dest/'review_gate.json',dict(allowed_run='quick50_only',later_targets_authorized=False,
        action_after_complete='present all13 conditions, then await explicit user review',
        previous_full_contracts_preserved=True))
    return data


def load(root):
    execution=contract(root); broad=base_contract(root,'broad50',execution)
    dest=destination(root); data=json.loads((dest/'contract.json').read_text())
    check_sources(data['sources']); validate_rows(data['selection']['rows'])
    if (data['fingerprint']!=execution['fingerprint'] or data['conditions']!=broad['conditions']
            or data['expected_per_condition']!=EXPECTED or data['expected_shards']!=325):
        raise ValueError('quick contract changed scientific settings or requested scope')
    original=base_setup(root)
    if data['selection']['rows'] != select_rows(original['final']):
        raise ValueError('quick sample selection changed')
    return data,execution


def run(root):
    frozen,execution=load(root); dest=destination(root); rows=frozen['selection']['rows']
    # Dense results are audited, never regenerated or filtered by their scores.
    for row in rows:
        read_result(root,row,'dense','dense',{},None,execution)
    import torch
    from dllm.models import create_adapter
    adapter=None; failures=[]
    for name in final_condition_order(frozen['conditions'],None):
        if name=='dense': continue
        c=frozen['conditions'][name]
        for row in rows:
            policy=c['thresholds'][row['benchmark']]
            try:
                try:
                    result,source=read_result(root,row,'final',name,c['config'],policy,execution)
                    reused=True
                except FileNotFoundError:
                    try:
                        result=reuse_equivalent(root,row,name,frozen['conditions'],execution)
                        reused=True
                    except FileNotFoundError:
                        try:
                            result=cached(None,root,row,'final',name,c['config'],policy,execution)
                            reused=True
                        except FileNotFoundError:
                            if adapter is None:
                                adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',
                                    precision='bfloat16',revision=REVISION).load()
                            result=cached(adapter,root,row,'final',name,c['config'],policy,execution)
                            reused=False
                dense,_=read_result(root,row,'dense','dense',{},None,execution)
                pair(row,result,dense)
                _append(dest/'completed.jsonl',dict(condition=name,id=row['id'],reused=reused,time=time.time()))
                print('quick50 complete',name,row['id'],'reused' if reused else 'generated',flush=True)
            except Exception:
                error=dict(stage='quick50',condition=name,id=row['id'],traceback=traceback.format_exc())
                failures.append(error);_append(dest/'failures.jsonl',error);print(error,flush=True)
            finally:
                torch.cuda.empty_cache()
    if adapter is not None:
        del adapter
        import gc
        gc.collect();torch.cuda.empty_cache()
    _write(dest/'run_terminal.json',dict(finished=time.time(),failed_attempts=len(failures),
        next_action='Report complete50% results and PAUSE; no later stages'))
    from .quick_report import regenerate
    audit=regenerate(root)
    if not audit['complete']:
        raise RuntimeError('quick50 incomplete; independent conditions continued and errors preserved')


def supervise(root):
    command=lambda stage,path,extra:[sys.executable,'-u','-m',__package__+'.quick_review',
        'run','--output',str(path)]
    with patch.object(jobs,'worker_args',command):
        jobs.supervise(root,'quick50',[])


def launch(root):
    # The actual shared lock and idle check are performed by the supervisor.
    load(root)
    with (destination(root)/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m',__package__+'.quick_review',
            'supervise','--output',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid,stage='quick50_only',pause_after=True)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','run','launch','supervise','report'))
    p.add_argument('--output',type=Path,default=ROOT);a=p.parse_args()
    if a.command=='prepare':
        f=prepare(a.output);print(json.dumps(dict(expected=f['expected_shards'],counts=EXPECTED,
            ids=[r['id'] for r in f['selection']['rows']])))
    elif a.command=='report':
        from .quick_report import regenerate
        result=regenerate(a.output)
        print(json.dumps({k:result[k] for k in ('complete','completed','expected','missing','violations')}))
        if not result['complete']:raise SystemExit(1)
    else: globals()[a.command](a.output)


if __name__=='__main__': main()
