"""CPU-only audit of reused tuning shards; never opens held-out outputs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path


def projected_source(source, origin, dense):
    """The only permitted cache rewrite is the documented observer projection."""
    expected = deepcopy(source)
    if dense:
        expected.update(
            records=[r for r in source['records'] if r['probe'] == 'execution'],
            distributions=[], screen=False, reused_dense_source=origin)
    else:
        expected['reused_screening_source'] = origin
    return expected


def validate_reuse(source, cached, origin, source_bytes, dense):
    from .protocol import sha
    if sha(source_bytes) != origin['sha256']:
        raise ValueError('cached source hash mismatch')
    if cached != projected_source(source, origin, dense):
        raise ValueError('cache differs beyond the permitted provenance/observer projection')


def audit(root):
    from .efficient_iteration import matching_sources, screen_rows
    from .protocol import fingerprint, prepare, sha
    from .report import inspect_output
    from .run import shard_path

    dest = root / 'efficient_iteration'
    plan = json.loads((dest / 'iteration_plan.json').read_text())
    contract_bytes = (root / 'final_contract.json').read_bytes()
    if sha(contract_bytes) != plan['original_contract_sha256']:
        raise ValueError('original frozen contract changed')
    contract = json.loads(contract_bytes)
    setup = prepare(root)
    fp = fingerprint(root)
    if fp != plan['fingerprint'] or fp != contract['fingerprint']:
        raise ValueError('scientific fingerprint changed')
    samples = {r['id']: r for r in screen_rows(setup)}
    index = json.loads((dest / 'cache_reuse.json').read_text())
    inspected = []
    seen = set()
    for entry in index['reused']:
        name, identity = entry['condition'], entry['id']
        if (name, identity) in seen:
            raise ValueError('duplicate cache reuse entry')
        seen.add((name, identity))
        if identity not in samples:
            raise ValueError('cache reuse ID outside calibration16')
        row = samples[identity]
        condition = plan['screening_conditions'][name]
        if condition != contract['conditions'][name]:
            raise ValueError('screening configuration differs from frozen condition')
        source_path = Path(entry['source'])
        # Determine admissibility before opening the source. matching_sources
        # permits final-path reads only for the six shared AIME tuning IDs.
        allowed = matching_sources(root, row, name, condition)
        if source_path.resolve() not in {p.resolve() for p in allowed}:
            raise ValueError(f'cache source outside matching calibration provenance: {source_path}')
        path = shard_path(dest, 'screening', name, identity)
        source_bytes, cached_bytes = source_path.read_bytes(), path.read_bytes()
        source, cached = json.loads(source_bytes), json.loads(cached_bytes)
        origin = cached['reused_dense_source' if name == 'dense' else 'reused_screening_source']
        if Path(origin['path']).resolve() != source_path.resolve():
            raise ValueError('cache origin differs from reuse index')
        validate_reuse(source, cached, origin, source_bytes, name == 'dense')
        for label, data in (('source', source), ('cached', cached)):
            _, _, errors = inspect_output(data, row, fp, condition)
            if errors:
                raise ValueError(f'{label} {name}/{identity}: {errors}')
        dense = json.loads(shard_path(dest, 'screening', 'dense', identity).read_text())
        for key in ('native_canvas_length', 'thinking', 'sampling', 'denoising_configuration'):
            if cached['generation_metadata'].get(key) != dense['generation_metadata'].get(key):
                raise ValueError(f'unrelated decoding setting differs from dense: {key}')
        inspected.append(dict(id=identity, condition=name, source=str(source_path),
            source_sha256=sha(source_bytes), cached=str(path), cached_sha256=sha(cached_bytes)))
    return dict(passed=True, inspected_reused_shards=len(inspected),
        expected_reused_shards=len(index['reused']), calibration_examples=len(samples),
        fingerprint=fp, original_contract_sha256=sha(contract_bytes),
        heldout_outputs_opened=False, downstream_scores_computed=False,
        scope='Only indexed cached calibration shards; not a complete final-run audit',
        inspected=inspected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
        default=Path('results/diffusion_gemma_value_aware_128x64'))
    args = parser.parse_args()
    result = audit(args.root)
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
    output = args.root / 'efficient_iteration' / 'early_cache_provenance_audit.json'
    _write(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'inspected'}))


if __name__ == '__main__':
    main()
