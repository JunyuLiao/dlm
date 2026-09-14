"""Preserve verified source overlays before later development changes."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from .config import ROOT
from .run import fingerprint as screen_fingerprint
from .online_run import fingerprint as online_fingerprint
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write


def preserve_cost(stage):
    """Cost harness overlay on its already-preserved inference source stage."""
    setup=json.loads((stage/'setup.json').read_text());audit=json.loads((stage/'audit.json').read_text())
    source=Path(__file__).with_name('performance.py');data=source.read_bytes()
    assert audit['complete'] and hashlib.sha256(data).hexdigest()==setup['performance_source_sha256']
    parent=Path(setup['stage']);manifest=parent/'source_snapshot'/'manifest.json'
    preserved=json.loads(manifest.read_text())
    assert preserved['executed_fingerprint']==setup['code_fingerprint']==online_fingerprint(parent)
    payload=dict(performance_source_path=str(source),performance_source_sha256=setup['performance_source_sha256'],
        performance_source=data.decode(),inference_source_manifest=str(manifest),
        inference_source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        executed_fingerprint=setup['code_fingerprint'])
    path=stage/'source_snapshot.json'
    if path.exists() and json.loads(path.read_text())!=payload:raise RuntimeError('immutable cost source snapshot differs')
    _write(path,payload);return str(path)


def preserve_validation(stage):
    from .validation import fingerprint
    setup=json.loads((stage/'freeze.json').read_text());smoke=json.loads((stage/'smoke.json').read_text())
    assert smoke['passed'] and fingerprint(stage)==smoke['fingerprint']
    parent=Path(setup['parent_stage']);manifest=parent/'source_snapshot'/'manifest.json'
    assert json.loads(manifest.read_text())['executed_fingerprint']==setup['parent_code_fingerprint']
    files={}
    for name in ('validation.py','online_report.py','work_volume.py','performance.py'):
        path=Path(__file__).with_name(name);data=path.read_bytes()
        files[str(path)]=dict(sha256=hashlib.sha256(data).hexdigest(),source=data.decode())
    payload=dict(executed_fingerprint=smoke['fingerprint'],inference_source_manifest=str(manifest),
        inference_source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),files=files)
    path=stage/'source_snapshot.json'
    if path.exists() and json.loads(path.read_text())!=payload:raise RuntimeError('immutable held-out source snapshot differs')
    _write(path,payload);return str(path)


def preserve(root=ROOT,*,stages=None):
    stages=stages if stages is not None else [(root,screen_fingerprint),(root/'online_v1',online_fingerprint)]
    paths=set(Path('experiments/diffusion_gemma_preqk').glob('*.py'))
    for package in ('diffusion_gemma_oracle','diffusion_gemma_solattn_blasst_multibench'):
        paths.update((Path('experiments')/package).glob('*.py'))
    paths.update(Path('src/dllm/attention/blasst').glob('*.py'))
    paths.update((Path('src/dllm/models/adapters/diffusion_gemma.py'),
        Path('experiments/diffusion_attention_threshold_modeling/math500.py'),Path('tests/test_diffusion_gemma_preqk.py')))
    dependencies={name:version(name) for name in ('torch','transformers','numpy','scipy','datasets','accelerate','safetensors','triton')}
    outputs=[]
    for stage,fp_fn in stages:
        smoke=json.loads((stage/'smoke.json').read_text());fp=fp_fn(stage)
        assert smoke['passed'] and smoke['fingerprint']==fp,'current code no longer matches the executed stage'
        target=stage/'source_snapshot';entries={}
        for path in sorted(paths):
            data=path.read_bytes();dest=target/path;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and dest.read_bytes()!=data:raise RuntimeError(f'cannot overwrite prior source snapshot: {dest}')
            dest.write_bytes(data);entries[str(path)]=hashlib.sha256(data).hexdigest()
        metadata=dict(executed_fingerprint=fp,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256=entries,dependencies=dependencies,
            note='Source overlay on recorded repository HEAD; pinned model/datasets/protocol remain in the parent bundle. No weights are copied.')
        _write(target/'manifest.json',metadata);outputs.append(str(target))
    return outputs


if __name__=='__main__':print(json.dumps(preserve(),indent=2))
