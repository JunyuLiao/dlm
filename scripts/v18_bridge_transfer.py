"""Local SSH relay into a private directory; never copies SSH credentials.

Only transport: a completed stage is not a bridge qualification. Both SSH
clients retain their normal host-key verification. Rerunning skips stages
with completed transport receipts; runtime/model identity is checked separately.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = '/home/exouser/dyh/junyu_frontier_v18_20260926'
SNAP = '.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--stage', choices=['runtime', 'model'], required=True)
    a = p.parse_args()
    a.log.parent.mkdir(parents=True, exist_ok=True)
    source = ['miniconda3/envs/ljy_dlm', '.local/lib/python3.10/site-packages'] if a.stage == 'runtime' else [SNAP]
    dest = ROOT + '/bridge/' + a.stage
    # The destination is fixed, private, and no source is removed or modified.
    setup = f'mkdir -p {dest}; test ! -e {dest}/TRANSPORT_DONE'
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
    r = subprocess.run(ssh + ['exouser@149.165.159.64', setup])
    if r.returncode:
        raise SystemExit('destination already transported or setup failed; inspect receipt')
    start = time.time()
    with a.log.open('a', encoding='utf-8') as log:
        log.write(json.dumps(dict(event='start', stage=a.stage, start=start, dest=dest)) + '\n')
        log.flush()
        producer = subprocess.Popen(ssh + ['exouser@149.165.151.254',
                                    'tar -chf - -C /home/exouser ' + ' '.join(source)],
                                    stdout=subprocess.PIPE, stderr=log)
        consumer = subprocess.Popen(ssh + ['exouser@149.165.159.64', 'tar -xf - -C ' + dest],
                                    stdin=producer.stdout, stdout=log, stderr=log)
        producer.stdout.close()
        rc_out = consumer.wait()
        rc_in = producer.wait()
        log.write(json.dumps(dict(event='transport_exit', stage=a.stage, producer=rc_in,
                                  consumer=rc_out, seconds=time.time()-start)) + '\n')
        log.flush()
    if rc_in or rc_out:
        raise SystemExit(1)
    subprocess.run(ssh + ['exouser@149.165.159.64', f'touch {dest}/TRANSPORT_DONE'], check=True)


if __name__ == '__main__':
    main()
