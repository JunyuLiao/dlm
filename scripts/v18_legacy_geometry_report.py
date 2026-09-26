"""CPU-only LEGACY_REPRO geometry count from the frozen T60 real states.

Counts native-legal key pairs and the historical Junyu sliding-window subset
for the same captured shapes. No kernel execution, accuracy, or speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def count_state(state: dict) -> dict:
    if state.get('derived_mask') or state.get('frontier_arm') != 'T60':
        raise ValueError('only real frozen T60 states qualify')
    if state['mask'] is not None or state['is_causal'] is not False:
        raise ValueError('this geometry-only report requires native mask=None, is_causal=False')
    b, h, nq, _ = state['q'].shape
    nk = state['k'].shape[-2]
    window = state['sliding_window']
    native = b * h * nq * nk
    if window:
        if type(window) is not int or window <= 0:
            raise ValueError('invalid historical window')
        rows = torch.arange(nq, dtype=torch.int64) + nk - nq
        keys = torch.arange(nk, dtype=torch.int64)
        historical = int((keys[None, :] >= rows[:, None] - window + 1).sum()) * b * h
    else:
        historical = native
    return dict(source_id=state['source_id'], layer=state['layer'], layer_kind=state['layer_kind'],
                nq=nq, nk=nk, sliding_window=window,
                native_legal_pairs=native, legacy_repro_legal_pairs=historical,
                native_legal_pairs_omitted_by_legacy=native - historical,
                legacy_retained_fraction=historical / native)


def report(states_path: Path) -> dict:
    states = torch.load(states_path, map_location='cpu', weights_only=True)
    real = [s for s in states if not s.get('derived_mask')]
    if len(real) != 12 or any(s.get('frontier_arm') != 'T60' for s in real):
        raise ValueError('expected all 12 predetermined real T60 states')
    rows = [count_state(s) for s in real]
    groups = {}
    for kind in ('local', 'global'):
        selected = [r for r in rows if r['layer_kind'] == kind]
        native = sum(r['native_legal_pairs'] for r in selected)
        kept = sum(r['legacy_repro_legal_pairs'] for r in selected)
        groups[kind] = dict(states=len(selected), native_legal_pairs=native,
                            legacy_repro_legal_pairs=kept, omitted=native - kept,
                            legacy_retained_fraction=None if not native else kept / native)
    return dict(schema='v18_legacy_repro_geometry_v1', claim='geometry-only LEGACY_REPRO diagnostic',
                accuracy_claim=False, speed_claim=False, states_sha256=file_sha(states_path),
                rows=rows, by_kind=groups)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = report(args.states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result['by_kind']))


if __name__ == '__main__':
    main()
