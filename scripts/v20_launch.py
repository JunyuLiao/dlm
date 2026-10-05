"""One-shot detached launch; a durable marker forbids uncertain duplicate launch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    marker = Path(config['launch_marker'])
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open('x') as f:
        json.dump(dict(status='launch_reserved', when=time.time(), config=str(a.config)), f)
    env = dict(os.environ, **config['env'])
    with open(config['log'], 'xb') as log:
        child = subprocess.Popen(config['command'], cwd=config['cwd'], env=env,
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(status='launched', pid=child.pid, when=time.time(), config=str(a.config))
    temp = marker.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2)+'\n')
    temp.replace(marker)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
