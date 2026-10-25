"""Sequential readout/qualification relay. No GPU resource ceiling or watchdog."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

STUDY=Path(__file__).resolve().parents[1];REPO=STUDY.parents[1]
R=STUDY/'track_r/attempt001';H=STUDY/'track_h/attempt001'
PRIVATE=STUDY/'private/post_r002';PRIVATE.mkdir(exist_ok=True)
PATHS=json.loads((R/'private/runtime_paths.json').read_text())
ENV=os.environ.copy();ENV.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(REPO/'src')+':'+str(REPO),HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True)


def run(name,args):
    start=time.time()
    with (PRIVATE/(name+'.log')).open('x') as log:
        result=subprocess.run([PATHS['python'],*map(str,args)],env=ENV,stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
    save(PRIVATE/(name+'_receipt.json'),dict(command=['<pinned_python>',*map(str,args)],returncode=result.returncode,seconds=time.time()-start,model_forwards=0))
    assert result.returncode==0,name+' failed; retained private log'
    print(json.dumps(dict(job=name,passed=True)),flush=True)


def commit(message):
    # Never consume an independently staged change. User authorized task-owned commits.
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).splitlines()
    assert all(x.startswith(str(STUDY.relative_to(REPO))+'/') for x in staged),'unrelated index entries: preserve and inspect'
    subprocess.run(['git','add','--',str(STUDY.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','commit','--only','-m',message,'--',str(STUDY.relative_to(REPO))],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()


def bind_prefix_inputs():
    """Validate identity/output/file bindings without showing or selecting text."""
    old=REPO/'results/denoising_e2e_research_20261024/attempt003';rows=[]
    for scope,root,panels in [('original',old,{'short':json.loads((old/'e2e_protocol.json').read_text())['panel']['short']}),
                              ('panel',R,json.loads((R/'protocol.json').read_text())['panel'])]:
        for task,cells in panels.items():
            for arm in json.loads((R/'protocol.json').read_text())['arms']:
                score_file=root/'scoring001/official_scores.json' if scope=='original' else root/f'runs/panel/{task}/{arm}/official_scores.json'
                # Original raw and canonical clean bindings suffice; supplemental text is never gold-scored.
                for i,identity in enumerate(cells):
                    private=root/f'private/generations/panel/{task}/{arm}/clean{i:03d}.json';public=root/f'runs/panel/{task}/{arm}/clean{i:03d}.json'
                    raw=json.loads(private.read_text());record=json.loads(public.read_text())
                    assert all(raw[k]==record[k]==identity[k] for k in ['id','seed'])
                    assert raw['output_sha256']==record['output_sha256']
                    rows.append(dict(scope=scope,id=identity['id'],seed=identity['seed'],arm=arm,private_generation_sha256=sha(private),clean_sha256=sha(public)))
    assert len(rows)==900
    save(STUDY/'content_prefix_study001/input_binding001.json',dict(passed=True,rows=rows,private_text_not_exposed=True))


def main():
    while not (R/'execution_complete.json').exists():
        assert not (R/'controller_stopped.json').exists(),'R controller stopped; no H launch'
        (PRIVATE/'heartbeat.json').write_text(json.dumps(dict(waiting_for='complete_R_readout',unix=time.time())))
        time.sleep(30)
    assert json.loads((R/'execution_complete.json').read_text())['passed']
    run('R_closeout',[STUDY/'reporting002/close_track_r.py'])
    bind_prefix_inputs()
    for scope in ['original','panel']:run('prefix_'+scope,[STUDY/'content_prefix_study001/analyze.py','--scope',scope])
    r_commit=commit('Complete unchanged Response behavioral replication and frozen short-task readout')
    save(PRIVATE/'R_readout_commit.json',dict(commit=r_commit,readout_sha256=sha(R/'readout_frozen.json')))
    assert not list(H.iterdir()),'H attempt must be empty before source copy'
    shutil.copytree(STUDY/'track_h/preparation001',H,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (H/'private').mkdir(exist_ok=True)
    shutil.copyfile(R/'private/runtime_paths.json',H/'private/runtime_paths.json')
    shutil.copyfile(R/'code/verify_inputs.py',H/'code/verify_inputs.py')
    # Change only the preparation label, before any H generation/freeze.
    spec=(H/'README.md').read_text().replace('# Prepared Track H contract — not launched or frozen for execution yet','# Track H: later-canvas signed-bias stopping-headroom diagnostic')
    spec=spec.replace('This source is preparation only. A separate execution attempt/source/protocol\ncheckpoint, tests, binding and R-readout hash must precede H GPU generation.',
        'This execution source requires its own committed source/protocol checkpoint,\nCPU tests, binding and R-readout hash before H GPU generation.')
    (H/'README.md').write_text(spec)
    run('H_old_dense_census',[H/'code/census.py'])
    run('H_manifest',[H/'code/prepare.py'])
    run('H_CPU',['-m','pytest','-q',H/'tests/test_headroom.py','--junitxml='+str(H/'tests/prelaunch002.xml')])
    run('H_model_runtime',[H/'code/verify_inputs.py'])
    run('H_freeze',[H/'freeze.py'])
    h_commit=commit('Freeze separate later-canvas signed-bias pulse headroom diagnostic')
    idle=not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    assert idle,'GPU occupied; do not preempt'
    save(H/'binding_receipt.json',dict(source_checkpoint=h_commit,protocol_sha256=sha(H/'protocol.json'),R_readout_sha256=sha(R/'readout_frozen.json'),
        R_final_complete_sha256=sha(R/'final_complete.json'),CPU_tests_passed=8,model_runtime_sha256=sha(H/'model_runtime_verification001.json'),GPU_idle=idle,no_resource_ceilings=True))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    save(H/'hardware_begin.json',dict(GPU=gpu,unix=time.time(),idle=True))
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None);env['PYTHONDONTWRITEBYTECODE']='1'
    with (H/'private/controller.log').open('x') as log:
        child=subprocess.Popen([PATHS['python'],str(H/'launch.py')],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(PRIVATE/'H_controller_launch.json',dict(pid=child.pid,source_checkpoint=h_commit,protocol_sha256=sha(H/'protocol.json'),started_unix=time.time()))
    print(json.dumps(dict(H_controller_PID=child.pid,H_source_checkpoint=h_commit,H_protocol_sha256=sha(H/'protocol.json'))),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(PRIVATE/'failure.json',dict(error_class=type(error).__name__,reason=str(error),unix=time.time(),no_budget_stop=True))
        raise
