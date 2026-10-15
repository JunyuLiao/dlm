"""Close observed MATH/RULER decision-coverage gaps before native promotion."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
from scripts import v31_value_decision_replay as io
from scripts import v31_value_suffix_campaign as suffix
from scripts import v31_value_decision_report as report
from scripts import v31_value_decision_analysis as analysis

SUITES=('math500','ruler128k')


def prepare(root):
    tests=root/'receipts/h100_qualification/attempt001/receipt.json'
    proof=io.read(tests)
    if proof['status']!='passed':raise RuntimeError('optimized operators must pass H100 correctness first')
    for rel,digest in proof['source_sha256'].items():
        if io.sha(io.ROOT/rel)!=digest:raise RuntimeError('native source changed since H100 qualification')
    for suite in SUITES:suffix.prepare(root/f'coverage_{suite}_attempt001',suite,'sparse_coverage')
    io.write(root/'coverage_protocol.json',dict(status='frozen',
        studies={s:dict(study=f'coverage_{s}_attempt001',protocol_sha256=io.sha(root/f'coverage_{s}_attempt001/protocol.json')) for s in SUITES},
        parent_sha256=io.sha(root/'decision_batch_complete.json'),h100_receipt_sha256=io.sha(tests),
        rationale='initial MATH had no sparse frames; initial RULER correction improved KL but worsened some acceptance/stopping decisions',
        selection='purposive coverage after initial diagnostics; explicitly exploratory, not a representative or held-out estimate',timing='paused'),True)


def run(root):
    protocol=io.read(root/'coverage_protocol.json');summaries={};inputs={}
    for suite,spec in protocol['studies'].items():
        study=root/spec['study']
        if io.sha(study/'protocol.json')!=spec['protocol_sha256']:raise RuntimeError('coverage protocol changed')
        if not (study/'qualification_complete.json').exists():
            io.write(root/'coverage_state.json',dict(stage='running',suite=suite,timing='paused'))
            env=dict(os.environ,PYTHONPATH=f"{study/'private/execution/src'}:{study/'private/execution'}")
            ordinal=len(list(study.glob('coordinator_attempt*.log')))+1
            with (study/f'coordinator_attempt{ordinal:03d}.log').open('w') as log:
                proc=subprocess.run([suffix.PY,str(study/'coordinator_source.py'),'--stage','run','--study',str(study)],
                    cwd=study/'private/execution',env=env,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode:
                io.write(root/'coverage_state.json',dict(stage='failed',suite=suite,resumable=True,timing='paused'))
                raise RuntimeError('preserved failed coverage worker')
        report.report(study)
        summary=io.read(study/'report/summary.json');rows=[]
        for rel,digest in summary['input_sha256'].items():
            if io.sha(study/rel)!=digest:raise RuntimeError('completed coverage records changed')
            inputs[str((study/rel).relative_to(root))]=digest
            if rel.startswith('repeat01/'):rows.extend(json.loads(x) for x in (study/rel).read_text().splitlines())
        summaries[suite]=analysis.summarize(rows)
    io.write(root/'coverage_analysis.json',dict(status='complete',workloads=summaries,input_sha256=inputs,
        scope=protocol['selection'],timing='paused',native_accuracy=False),True)
    io.write(root/'coverage_complete.json',dict(status='complete',protocol_sha256=io.sha(root/'coverage_protocol.json'),
        report_sha256={f"{spec['study']}/report/summary.json":io.sha(root/spec['study']/'report/summary.json') for spec in protocol['studies'].values()},
        analysis_sha256=io.sha(root/'coverage_analysis.json'),timing='paused'),True)
    io.write(root/'coverage_state.json',dict(stage='complete',timing='paused'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','run'],required=True)
    p.add_argument('--study',type=Path,default=io.DEFAULT);a=p.parse_args()
    {'prepare':prepare,'run':run}[a.stage](a.study.resolve())
