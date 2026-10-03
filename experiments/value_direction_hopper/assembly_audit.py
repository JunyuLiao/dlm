"""Preserve compiler warnings and static SASS, not dynamic instruction counts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def run(base,variant,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);rows=[]
    for name,library in [('frozen',base),('float_operands',variant)]:
        build=json.loads(library.with_name(library.stem.removeprefix('value_direction_')+'.json').read_text())
        warnings=[line for line in build['stderr'].splitlines() if 'C7519' in line]
        for width in (256,512):
            symbol=f'_ZN4fmha15value_direction6kernelILi{width}ELb1ELb1ELb1ELb0EEEvNS0_6ParamsENS0_7TmaMapsE'
            command=['/usr/local/cuda/bin/cuobjdump','--dump-sass','--function',symbol,str(library)]
            result=subprocess.run(command,capture_output=True,text=True);result.check_returncode()
            path=output/f'{name}_d{width}.sass';path.write_text(result.stdout)
            rows.append(dict(name=name,width=width,library=str(library),command=command,sass=str(path),
                sass_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stderr=result.stderr,
                serialization_warnings=[w for w in warnings if symbol in w],
                static_hgmma_instructions=result.stdout.count(' HGMMA.'),
                static_warpgroup_waits=result.stdout.count(' WARPGROUP.DEPBAR'),
                static_warpgroup_arrives=result.stdout.count(' WARPGROUP.ARRIVE')))
    (output/'summary.json').write_text(json.dumps(dict(cases=rows,scope='Static compiled instruction inventory; dynamic execution counts and stall cycles require hardware profiling'),indent=2)+'\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--variant',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.base,a.variant,a.output)
