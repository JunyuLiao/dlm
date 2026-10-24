"""Complete exactly the authorized finite screen; never expand or tune it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--supervisor-pid',type=int,required=True)
    ap.add_argument('--supervisor-start-ticks',required=True);args=ap.parse_args()
    here=Path(__file__).resolve().parent;root=here.parent/'attempt003'
    protocol=json.loads((root/'e2e_protocol.json').read_text())
    spec=json.loads((here/'binding.json').read_text())
    if hashlib.sha256((root/'e2e_protocol.json').read_bytes()).hexdigest()!=spec['parent_protocol_sha256']:
        raise RuntimeError('parent protocol drift')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=spec['controller_sha256']:
        raise RuntimeError('controller source drift')
    for p,h in spec['sources'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('bound source drift '+p)
    started=time.time();commands=[]
    with (here/'launch_receipt.json').open('x') as f:json.dump(dict(started_unix=started,
        supervisor_pid=args.supervisor_pid,supervisor_start_ticks=args.supervisor_start_ticks,
        parent_protocol_sha256=spec['parent_protocol_sha256'],source_checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope='same25questions2seeds9arms; no expansion; no campaign GPU ceiling'),f,indent=2)
    targets=[a for a in protocol['arms'] if a!='v31_optimized']
    while True:
        complete=[a for a in targets if (root/'runs/panel/long'/a/'final_complete.json').exists()]
        if len(complete)==len(targets):break
        for arm in targets:
            p=root/'launches/long'/arm/'exit_receipt.json'
            if p.exists() and json.loads(p.read_text())['returncode']:
                raise RuntimeError('Long worker failed; preserve and require primary review: '+arm)
        process=Path('/proc')/str(args.supervisor_pid)
        if not process.exists() or process.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]!=args.supervisor_start_ticks:
            raise RuntimeError('Original non-V31 supervisor exited before family completion')
        current=[]
        for arm in targets:
            leaf=root/'runs/panel/long'/arm
            current.append(dict(arm=arm,audit_records=len(list(leaf.glob('audit???.json'))),
                clean_records=len(list(leaf.glob('clean???.json'))),complete=arm in complete))
        with (here/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),nonv31_long=current))+'\n')
        time.sleep(30)
    environment=os.environ.copy();environment.update(PYTHONPATH='/home/exouser/ljy/dlm/src:/home/exouser/ljy/dlm',HF_HUB_OFFLINE='1')
    python='/home/exouser/dyh/v31_env_20261005/bin/python'
    jobs=[('v31_long',here.parent/'attempt005/launch_v31.py',['--suite','long'],False),
          ('official_score',root/'score_v3.py',['--root',str(root)],True),
          ('analysis',root/'analyze_v4.py',['--root',str(root)],True),
          ('paired_trajectory',root/'paired_trajectory.py',['--root',str(root)],True),
          ('native_bottlenecks',root/'trajectory_bottlenecks.py',['--root',str(root)],True),
          ('support_diversity',root/'support_diversity.py',['--root',str(root)],True),
          ('output_diagnostics',root/'output_diagnostics.py',['--root',str(root)],True),
          ('plots',root/'plot_trajectories.py',['--root',str(root)],True),
          ('verify',root/'verify.py',['--root',str(root)],True),
          ('report',root/'report.py',['--root',str(root)],True)]
    for name,script,extra,cpu in jobs:
        env=dict(environment)
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        command=[python,str(script),*extra];step_start=time.time()
        commands.append(dict(name=name,command=command,CPU_only=cpu,started_unix=step_start))
        with (here/'private'/(name+'.log')).open('x') as stream:
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        receipt=dict(name=name,returncode=result.returncode,elapsed_seconds=time.time()-step_start,command=command,CPU_only=cpu)
        with (here/(name+'_receipt.json')).open('x') as f:json.dump(receipt,f,indent=2)
        print(json.dumps(receipt),flush=True)
        if result.returncode:raise RuntimeError('Completion stage failed, preserved for primary review: '+name)
    with (here/'final_complete.json').open('x') as f:json.dump(dict(passed=True,
        scope='execution/scoring/readout controller; publication still primary-reviewed',
        parent_protocol_sha256=spec['parent_protocol_sha256'],elapsed_seconds=time.time()-started,
        commands=commands,method_promoted=False),f,indent=2)


if __name__=='__main__':
    try:main()
    except Exception as error:
        here=Path(__file__).resolve().parent
        with (here/'failed_preserved.json').open('x') as f:json.dump(dict(error_class=type(error).__name__,
            reason=str(error),unix=time.time(),automatic_retry=False),f,indent=2)
        raise
