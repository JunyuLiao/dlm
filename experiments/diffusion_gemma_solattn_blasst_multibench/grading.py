"""Official LiveCodeBench scoring behind an explicit process/container boundary."""
import json
import os
import subprocess
from pathlib import Path
from .config import condition_map
from .controlled_dataset import digest
from .controlled_runner import load_completed
from .runner import _write, _append


def grade_livecodebench(output_dir,docker_image='dlm-experiment:h100'):
    image_id=subprocess.check_output(['docker','image','inspect',docker_image,'--format','{{.Id}}'],text=True).strip()
    summary={}
    for name in condition_map():
        rows=[r for r in load_completed(output_dir,name) if r['benchmark']=='livecodebench_v6']
        if not rows: continue
        directory=output_dir/'conditions'/name;target=directory/'livecodebench_grades.json'
        inputs=[dict(id=r['id'],lcb=r['lcb'],code=r['code'],code_hash=digest(r['code'])) for r in rows]
        fingerprint=digest(json.dumps(inputs,sort_keys=True)+image_id+Path(__file__).with_name('grade_worker.py').read_text())
        if target.exists() and json.loads(target.read_text()).get('fingerprint')==fingerprint:
            summary[name]={'cached':True,'graded':len(rows)};continue
        source=directory/'grading_input.json';_write(source,inputs)
        container_name='dlm-controlled-grade-'+fingerprint[:16]
        command=['docker','run','--rm','--name',container_name,'--network','none','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--user',f'{os.getuid()}:{os.getgid()}',
            '--pids-limit','128','--memory','4g','--cpus','4','--tmpfs','/tmp:rw,nosuid,size=1g,mode=1777',
            '-e','PYTHONPATH=/scorer','-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
            '-v',f'{source.resolve()}:/input.json:ro',
            '-v',f'{Path("reference/LiveCodeBench").resolve()}:/scorer:ro',
            '-v',f'{Path(__file__).with_name("grade_worker.py").resolve()}:/worker.py:ro',
            '-w','/tmp',image_id,'python','/worker.py']
        try:
            process=subprocess.run(command,capture_output=True,text=True,timeout=1800,check=True)
            line=next(x for x in process.stdout.splitlines() if x.startswith('CONTROLLED_GRADES='))
            grades=json.loads(line.split('=',1)[1]);grades.update(fingerprint=fingerprint,image_id=image_id,
                scorer_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd='reference/LiveCodeBench',text=True).strip())
            _write(target,grades);summary[name]={'graded':len(grades['grades'])}
        except Exception as exc:
            if isinstance(exc,subprocess.TimeoutExpired):
                # Kill only this invocation's isolated worker, not other jobs.
                subprocess.run(['docker','rm','-f',container_name],capture_output=True,timeout=30)
            error=dict(stage='grading',condition=name,error=repr(exc),stderr=getattr(exc,'stderr',None))
            _append(output_dir/'failures.jsonl',error);summary[name]=error
    _write(output_dir/'grading_status.json',summary)
    return summary
