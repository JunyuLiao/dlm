"""Byte-preserving deterministic export of shareable v20 profile measurements."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path


SOURCES = (
    ('screen_002', 'mpk'), ('screen_002', 'dllm'),
    ('selected_001', 'mpk'), ('selected_001', 'dllm'),
    ('ruler_selected_001', 'mpk'), ('ruler_selected_001', 'dllm'),
)
FORBIDDEN = frozenset(('prompt', 'prompt_tokens', 'input_ids', 'token_ids',
                       'completion_tokens', 'answer', 'expected_answer', 'gold',
                       'past_key_values', 'hidden_states', 'pixel_values'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit(value):
    """Reject payload keys and long free text; retain numeric work/timing rows."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN:
                raise ValueError(f'private payload field present: {key}')
            audit(child)
    elif isinstance(value, list):
        for child in value:
            audit(child)
    elif isinstance(value, str) and len(value) > 200:
        raise ValueError('long free-text field present in profile')


def deterministic_gzip(data):
    output = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0, compresslevel=9) as stream:
        stream.write(data)
    return output.getvalue()


def export(source_root, out):
    source_root, out = Path(source_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('profile record export directory must be new and empty')
    entries = []
    for stage, host in SOURCES:
        source = source_root / f'watch_{stage}' / host / f'{stage}.json'
        raw = source.read_bytes()
        profile = json.loads(raw)
        if profile.get('schema') != 'v20_direct_full_forward_v1' or not isinstance(profile.get('targets'), dict):
            raise ValueError(f'not a completed direct profile: {stage}/{host}')
        audit(profile)
        identity = profile.get('runtime_identity', {})
        if identity.get('hostname') != host or not identity.get('gpu_uuid'):
            raise ValueError(f'host/GPU identity absent or crossed: {stage}/{host}')
        compressed = deterministic_gzip(raw)
        if len(compressed) >= 50_000_000:
            raise ValueError(f'profile export exceeds 50 MB: {stage}/{host}')
        name = f'{stage}_{host}.json.gz'
        path = out / name
        with path.open('xb') as stream:
            stream.write(compressed)
        if gzip.decompress(path.read_bytes()) != raw:
            raise AssertionError('profile gzip roundtrip changed source bytes')
        missing = sum(bool(t.get('resolution', {}).get('missing')) for t in profile['targets'].values())
        unprofiled = sum(not t.get('boundaries') for t in profile['targets'].values())
        omitted_lengths = sorted({n for t in profile['targets'].values()
                                 for boundary in t.get('boundaries', {}).values()
                                 for n in ('N4', 'N16') if n not in boundary})
        entries.append(dict(stage=stage, host=host, gpu_uuid=identity['gpu_uuid'],
                            file=name, source_sha256=sha(raw), gzip_sha256=sha(compressed),
                            source_bytes=len(raw), gzip_bytes=len(compressed),
                            targets=len(profile['targets']), missing_target_resolutions=missing,
                            targets_without_profiled_boundary=unprofiled,
                            absent_sequence_lengths=omitted_lengths))
    manifest = dict(schema='v20_profile_record_export_v1', format='gzip of exact UTF-8 source JSON bytes',
                    payload_audit='no prompt, gold, answer, completion token, input ID, hidden state or raw tensor fields; strings <=200 characters',
                    omitted_fields=[], entries=entries,
                    caveats=['Captured snapshot inventory lists Python types and dimensions, not tensors or tokens.',
                             'model_forward is full model.forward with state.begin outside that timer.',
                             'denoising_step includes the native sampler, observer, T and control path.',
                             'Direct replay timing is not a generated request or answer-quality measurement.',
                             'Missing native target calls and absent N16 sequences remain missing; no padding.'])
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    with (out / 'manifest.json').open('xb') as stream:
        stream.write(payload)
    readme = ('# Direct profile records\n\n'
              'The six `.json.gz` files are byte-preserving gzip copies of the frozen direct-profile JSON. '
              'They include accepted timing repetitions, native brackets, first/cold observations, '
              'physical work counters, operator-probe numerical measurements, phase evidence, '
              'source hashes, and explicit missing target/sequence records.\n\n'
              'The source audit found no raw prompts, gold answers, completion tokens, input IDs, '
              'hidden states, or tensor payloads. No fields were removed. Input/output and QKV '
              'values appear only as digests and scalar statistics. Absolute host and model paths '
              'are provenance metadata.\n\n'
              'The profile replay uses captured native states. `model_forward` times complete '
              '`model.forward` calls with `state.begin` outside the timer; `denoising_step` '
              'times the full native step. CUDA events include stream launch gaps, and direct '
              'sequences are not natural generated requests. Missing native calls or N16 '
              'sequences were not synthesized. See `manifest.json` for SHA-256 checksums.\n')
    with (out / 'README.md').open('xb') as stream:
        stream.write(readme.encode())
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = export(args.source_root, args.out)
    print(json.dumps(dict(files=len(result['entries']), gzip_bytes=sum(e['gzip_bytes'] for e in result['entries']))))


if __name__ == '__main__':
    main()
