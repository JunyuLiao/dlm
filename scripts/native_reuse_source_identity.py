"""Export source/config identities only, never prompts or generated answers."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    entries = []
    for root in args.root:
        for path in sorted(root.glob('configs/*.json')):
            content = json.loads(path.read_text())
            entries.append(dict(config_path=str(path.resolve()),
                config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                config=content))
    if not entries:
        raise ValueError('No frozen configs')
    args.output.write_text(json.dumps(dict(schema='numerical_reuse_execution_identity_v1',
        configs=entries, authority='Loaded module and binary hashes, not latest report Git HEAD'), indent=2)+'\n')


if __name__ == '__main__':
    main()
