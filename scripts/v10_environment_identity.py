"""v10: freeze the ACTUAL import/environment/storage identity (no GPU work)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha(path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def run(command: str) -> str:
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout


def main() -> None:
    out = Path(sys.argv[1])
    import torch
    import transformers
    import triton
    from transformers.models.diffusion_gemma import generation_diffusion_gemma, modeling_diffusion_gemma
    from transformers.integrations import sdpa_attention
    triton_root = Path(triton.__file__).parent
    library = '/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/value_direction_db080045f7a5fbce.so'
    bridge = ('/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/torch_4c65c048754f9fb7/'
              'value_direction_torch_4c65c048754f9fb7.so')
    identity = dict(
        schema='v10_environment_and_storage_identity_v1',
        executable=sys.executable, python=sys.version, sys_path=sys.path,
        env=dict(PYTHONPATH=os.environ.get('PYTHONPATH'), PYTHONNOUSERSITE=os.environ.get('PYTHONNOUSERSITE'),
                 TRITON_CACHE_DIR=os.environ.get('TRITON_CACHE_DIR'), CONDA_DEFAULT_ENV=os.environ.get('CONDA_DEFAULT_ENV')),
        modules=dict(
            torch=dict(version=torch.__version__, file=torch.__file__, cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version(), version_py_sha256=sha(Path(torch.__file__).parent / 'version.py')),
            triton=dict(version=triton.__version__, file=triton.__file__,
                        jit_py_sha256=sha(triton_root / 'runtime/jit.py'),
                        compiler_py_sha256=sha(triton_root / 'compiler/compiler.py'),
                        shadowed_conda_triton=sorted(p.name for p in Path(sys.prefix, 'lib/python3.10/site-packages').glob('triton-*.dist-info'))),
            transformers=dict(version=transformers.__version__, file=transformers.__file__,
                              generation_sha256=sha(generation_diffusion_gemma.__file__),
                              modeling_sha256=sha(modeling_diffusion_gemma.__file__),
                              sdpa_attention_sha256=sha(sdpa_attention.__file__))),
        fresh_junyu_T_extension=dict(library=library, library_sha256=sha(library), bridge=bridge, bridge_sha256=sha(bridge)),
        device=dict(name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    capability=list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
                    driver=run('nvidia-smi --query-gpu=driver_version --format=csv,noheader').strip()),
        flags=dict(bf16='model loaded in torch.bfloat16', tf32_matmul_set_by_runner=False, tf32_cudnn_set_by_runner=False,
                   attn_implementation='sdpa (adapter.load)', native_dispatch='ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward when unbound'),
        host=dict(hostname=run('hostname').strip(), ips=run('hostname -I').strip()),
        storage=dict(df_hT=run('df -hT / /media/volume/dllm-1'), df_i=run('df -i / /media/volume/dllm-1'),
                     findmnt=run('findmnt /media/volume/dllm-1'),
                     bulky_root='/media/volume/dllm-1/dyh/numerical_qk_overnight_runtime_20260925',
                     migration='none (root had ~57 GB free at start; nothing moved or deleted)'),
        import_policy=('same interpreter for every job: conda ljy_dlm python with user site ENABLED '
                       '(torch/triton resolve from ~/.local); PYTHONNOUSERSITE never set; no python -s'))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
    print(json.dumps(identity['modules'], indent=1)[:1500])


if __name__ == '__main__':
    main()
