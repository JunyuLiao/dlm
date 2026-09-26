"""Read-only host inventory; never imports model weights or changes the environment."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except Exception as e:
        return dict(error=str(e))


def main():
    result = dict(utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                  executable=sys.executable, python=sys.version, user_site=os.environ.get('PYTHONNOUSERSITE'),
                  modules={}, commands={})
    for name in ('torch', 'triton', 'transformers',
                 'transformers.models.diffusion_gemma.generation_diffusion_gemma',
                 'transformers.models.diffusion_gemma.modeling_diffusion_gemma'):
        try:
            m = importlib.import_module(name)
            p = Path(m.__file__)
            result['modules'][name] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                           version=getattr(m, '__version__', None))
            if name == 'torch':
                result['torch_abi'] = m._C._GLIBCXX_USE_CXX11_ABI
                result['torch_cuda'] = m.version.cuda
        except Exception as e:
            result['modules'][name] = dict(error=str(e))
    commands = [['lscpu'], ['free', '-b'], ['df', '-h'], ['df', '-i'], ['lsblk', '-f'],
                ['findmnt', '-rn'], ['nvidia-smi', '-q'], ['nvcc', '--version'], ['c++', '--version'],
                ['ps', '-eo', 'user,pid,pgid,etime,args'], ['lslocks']]
    for args in commands:
        result['commands'][' '.join(args)] = command(args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
