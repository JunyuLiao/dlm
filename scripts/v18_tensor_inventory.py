"""CPU tensor-payload inventory for bridge identity; no model construction."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import torch
    from safetensors import safe_open
    index = a.model / 'model.safetensors.index.json'
    weight_map = json.loads(index.read_text())['weight_map']
    groups = defaultdict(list)
    for name, file in weight_map.items():
        groups[file].append(name)
    result = dict(model=str(a.model), index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
                  torch=torch.__version__, tensors={})
    started = time.time()
    for file, names in sorted(groups.items()):
        with safe_open(a.model / file, framework='pt', device='cpu') as reader:
            for name in sorted(names):
                t = reader.get_tensor(name).contiguous()
                raw = t.reshape(-1).view(torch.uint8).numpy()
                result['tensors'][name] = dict(shape=list(t.shape), dtype=str(t.dtype),
                                               sha256=hashlib.sha256(memoryview(raw)).hexdigest())
    result['seconds'] = time.time() - started
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temp = a.out.with_suffix('.tmp')
    temp.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    temp.replace(a.out)
    print(json.dumps(dict(tensors=len(result['tensors']), seconds=result['seconds'])))


if __name__ == '__main__':
    main()
