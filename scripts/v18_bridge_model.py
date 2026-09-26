"""Build a private pinned-snapshot bridge from exact shared export tensors.

extract (old host): copy only the 386 absent tensors into missing.safetensors
and small pinned model metadata. assemble (export host): link the existing
read-only export shards, add missing.safetensors, and write the combined index.
Neither command mutates the old snapshot or the text export. Run the separate
v18_tensor_inventory.py over the result to verify all 1047 tensor payloads.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

METADATA_SUFFIXES = {'.json', '.model', '.txt', '.jinja'}
INDEX = 'model.safetensors.index.json'
SHARD = 'missing.safetensors'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(comparison: Path, old_inventory: Path, export_inventory: Path):
    c = json.loads(comparison.read_text())
    old = json.loads(old_inventory.read_text())['tensors']
    export = json.loads(export_inventory.read_text())['tensors']
    missing = set(c['missing'])
    if c['differences'] or set(export) & missing or set(old) != set(export) | missing:
        raise ValueError('comparison is not an exact shared subset plus missing tensors')
    if c['shared'] != len(export) or len(missing) != 386 or len(old) != 1047:
        raise ValueError('unexpected frozen tensor counts')
    if any(old[name] != value for name, value in export.items()):
        raise ValueError('shared tensor inventory differs')
    return old, export, missing


def extract(old_model: Path, comparison: Path, old_inventory: Path,
            export_inventory: Path, bundle: Path) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    old, _export, missing = inputs(comparison, old_inventory, export_inventory)
    old_index = json.loads((old_model / INDEX).read_text())
    old_map = old_index['weight_map']
    if set(old_map) != set(old):
        raise ValueError('old index and tensor inventory disagree')
    if bundle.exists() and any(bundle.iterdir()):
        raise FileExistsError('bundle directory must be new or empty')
    bundle.mkdir(parents=True, exist_ok=True)
    tensors = {}
    groups = defaultdict(list)
    for name in sorted(missing):
        groups[old_map[name]].append(name)
    for filename, names in sorted(groups.items()):
        with safe_open(old_model / filename, framework='pt', device='cpu') as reader:
            for name in names:
                value = reader.get_tensor(name).contiguous()
                raw = value.reshape(-1).view(torch.uint8).numpy()
                observed = dict(shape=list(value.shape), dtype=str(value.dtype),
                                sha256=hashlib.sha256(memoryview(raw)).hexdigest())
                if observed != old[name]:
                    raise ValueError(f'old payload differs from audited inventory: {name}')
                tensors[name] = value
    save_file(tensors, bundle / SHARD)
    del tensors
    # Verify the written shard independently, including its exact key set.
    with safe_open(bundle / SHARD, framework='pt', device='cpu') as reader:
        if set(reader.keys()) != missing:
            raise ValueError('written missing shard has wrong keys')
        for name in sorted(missing):
            value = reader.get_tensor(name).contiguous()
            digest = hashlib.sha256(memoryview(value.reshape(-1).view(torch.uint8).numpy())).hexdigest()
            if digest != old[name]['sha256']:
                raise ValueError(f'written missing shard payload differs: {name}')
    metadata = bundle / 'metadata'
    metadata.mkdir()
    copied = {}
    for source in sorted(old_model.iterdir()):
        if source.is_file() and source.name != INDEX and source.suffix in METADATA_SUFFIXES:
            destination = metadata / source.name
            shutil.copy2(source, destination)
            copied[source.name] = sha(destination)
    if 'config.json' not in copied or 'tokenizer_config.json' not in copied:
        raise ValueError('pinned config/tokenizer metadata absent')
    receipt = dict(schema='v18_missing_bundle_v1', missing_tensors=len(missing), shard_sha256=sha(bundle / SHARD),
                   old_index_sha256=sha(old_model / INDEX), comparison_sha256=sha(comparison),
                   metadata_sha256=copied)
    (bundle / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def assemble(export_model: Path, bundle: Path, comparison: Path, old_inventory: Path,
             export_inventory: Path, out_model: Path) -> dict:
    from safetensors import safe_open

    old, export, missing = inputs(comparison, old_inventory, export_inventory)
    receipt = json.loads((bundle / 'receipt.json').read_text())
    if receipt['comparison_sha256'] != sha(comparison) or receipt['shard_sha256'] != sha(bundle / SHARD):
        raise ValueError('bundle receipt mismatch')
    if out_model.exists() and any(out_model.iterdir()):
        raise FileExistsError('output model directory must be new or empty')
    export_index = json.loads((export_model / INDEX).read_text())
    export_map = export_index['weight_map']
    if set(export_map) != set(export):
        raise ValueError('export index and audited inventory disagree')
    with safe_open(bundle / SHARD, framework='pt', device='cpu') as reader:
        if set(reader.keys()) != missing:
            raise ValueError('missing shard key set mismatch')
    out_model.mkdir(parents=True, exist_ok=True)
    for filename in sorted(set(export_map.values())):
        source = (export_model / filename).resolve(strict=True)
        if export_model.resolve() not in source.parents:
            raise ValueError('export shard escapes export directory')
        (out_model / filename).symlink_to(source)
    shutil.copy2(bundle / SHARD, out_model / SHARD)
    for filename, digest in receipt['metadata_sha256'].items():
        source = bundle / 'metadata' / filename
        if sha(source) != digest:
            raise ValueError(f'metadata changed: {filename}')
        shutil.copy2(source, out_model / filename)
    old_bytes = sum(__import__('math').prod(x['shape']) *
                    {'torch.float32': 4, 'torch.bfloat16': 2, 'torch.float16': 2, 'torch.int64': 8,
                     'torch.int32': 4, 'torch.uint8': 1}[x['dtype']] for x in old.values())
    combined = dict(metadata={'total_size': old_bytes},
                    weight_map={**export_map, **{name: SHARD for name in missing}})
    if set(combined['weight_map']) != set(old):
        raise AssertionError('combined index has wrong tensor keys')
    (out_model / INDEX).write_text(json.dumps(combined, indent=2, sort_keys=True) + '\n')
    result = dict(schema='v18_bridge_model_v1', old_tensors=len(old), linked_export_shards=len(set(export_map.values())),
                  missing_tensors=len(missing), index_sha256=sha(out_model / INDEX),
                  missing_shard_sha256=sha(out_model / SHARD), export_model=str(export_model.resolve()))
    (out_model / 'bridge_receipt.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('extract', 'assemble'):
        q = sub.add_parser(command)
        q.add_argument('--comparison', type=Path, required=True)
        q.add_argument('--old-inventory', type=Path, required=True)
        q.add_argument('--export-inventory', type=Path, required=True)
        if command == 'extract':
            q.add_argument('--old-model', type=Path, required=True)
            q.add_argument('--bundle', type=Path, required=True)
        else:
            q.add_argument('--export-model', type=Path, required=True)
            q.add_argument('--bundle', type=Path, required=True)
            q.add_argument('--out-model', type=Path, required=True)
    args = p.parse_args()
    common = (args.comparison, args.old_inventory, args.export_inventory)
    result = (extract(args.old_model, *common, args.bundle) if args.command == 'extract'
              else assemble(args.export_model, args.bundle, *common, args.out_model))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
