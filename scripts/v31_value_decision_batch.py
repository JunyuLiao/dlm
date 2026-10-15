"""Sequential multi-workload same-state diagnostics after eager qualification.

No task-accuracy generation or timing can be launched by this coordinator.
Every workload freezes its own source/input protocol before any worker starts.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import subprocess
from scripts import v31_value_decision_replay as io
from scripts import v31_value_suffix_campaign as suffix
from scripts import v31_value_decision_report as report

SUITES=('longbench_v2','aime26','math500','ruler128k')


def prepare(root):
    qualification=root/'suffix_qualification_attempt006'
    marker=io.read(qualification/'qualification_complete.json')
    assert marker['reproducibility_sha256']==io.sha(qualification/'reproducibility.json')
    assert io.read(qualification/'reproducibility.json')['status']=='passed'
    for suite in SUITES:
        study=root/f'decisions_{suite}_attempt001'
        suffix.prepare(study,suite,'decisions')
    io.write(root/'decision_batch_protocol.json',dict(schema='value_decision_multisuite_v1',
        suites={s:dict(study=f'decisions_{s}_attempt001',protocol_sha256=io.sha(root/f'decisions_{s}_attempt001/protocol.json')) for s in SUITES},
        qualification_sha256=io.sha(qualification/'qualification_complete.json'),
        timing='paused',native_accuracy_launched=False,
        inference='same-state eager layer29 decisions; no task-accuracy or production latency claim'),True)


def run(root):
    protocol=io.read(root/'decision_batch_protocol.json')
    for suite,spec in protocol['suites'].items():
        study=root/spec['study']
        assert io.sha(study/'protocol.json')==spec['protocol_sha256']
        if not (study/'qualification_complete.json').exists():
            io.write(root/'decision_batch_state.json',dict(stage='running',suite=suite,study=spec['study'],timing='paused'))
            env=dict(os.environ,PYTHONPATH=f"{study/'private/execution/src'}:{study/'private/execution'}")
            prior=list(study.glob('coordinator_attempt*.log'))
            with (study/f'coordinator_attempt{len(prior)+1:03d}.log').open('w') as log:
                proc=subprocess.run([suffix.PY,str(study/'coordinator_source.py'),'--stage','run','--study',str(study)],
                    cwd=study/'private/execution',env=env,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode:
                io.write(root/'decision_batch_state.json',dict(stage='failed',suite=suite,resumable=True,timing='paused'))
                raise RuntimeError('preserved failed diagnostic; inspect its state and worker log')
        report.report(study)
    io.write(root/'decision_batch_complete.json',dict(status='complete',
        protocol_sha256=io.sha(root/'decision_batch_protocol.json'),
        reports_sha256={s:io.sha(root/spec['study']/'report/summary.json') for s,spec in protocol['suites'].items()},
        native_accuracy_launched=False,timing='paused'),True)
    io.write(root/'decision_batch_state.json',dict(stage='decisions_complete',timing='paused',native_accuracy_launched=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','run'],required=True)
    p.add_argument('--study',type=Path,default=io.DEFAULT)
    a=p.parse_args();{'prepare':prepare,'run':run}[a.stage](a.study.resolve())
