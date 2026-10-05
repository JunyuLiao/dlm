"""Rebind a frozen config to byte-identical sources in a new deployment.

Only source_hashes path keys and existing fingerprints may change. Hash values,
method settings, binary paths and all other fields remain frozen. This module
uses only the standard library and does not import the method or GPU runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


_SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def fingerprint(config):
    """The exact canonical JSON hash used by the v20 and v21 guards."""
    payload = {key: value for key, value in config.items() if key != 'fingerprint'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _validate_fingerprints(value, location='$'):
    if isinstance(value, dict):
        for key, child in value.items():
            if key != 'fingerprint':
                _validate_fingerprints(child, f'{location}.{key}')
        if 'fingerprint' in value and value['fingerprint'] != fingerprint(value):
            raise ValueError(f'original config fingerprint drift at {location}')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_fingerprints(child, f'{location}[{index}]')


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _check_file(path, expected, label):
    try:
        actual = _file_hash(path)
    except OSError as exc:
        raise ValueError(f'{label} source is unavailable: {path}') from exc
    if actual != expected:
        raise ValueError(f'{label} source byte identity drift: {path}')


def _relative(path, root):
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _method_fields(value):
    """Remove only the two binding fields for the final preservation guard."""
    if isinstance(value, dict):
        return {key: _method_fields(child) for key, child in value.items()
                if key not in ('source_hashes', 'fingerprint')}
    if isinstance(value, list):
        return [_method_fields(child) for child in value]
    return value


def rebind_config(config, old_root, new_root):
    """Return a new config after validating every original and rebound source.

    Run on the host that owns both deployment directories. External source
    paths are verified and retained verbatim. A changed or missing source,
    incorrect nested fingerprint, path collision or escaping symlink rejects
    the entire binding. The supplied dictionary is never modified.
    """
    if not isinstance(config, dict) or 'fingerprint' not in config:
        raise ValueError('a fingerprinted frozen config dictionary is required')
    # Validate the original nested identities before changing any path.
    _validate_fingerprints(config)
    old_root = Path(old_root).resolve(strict=True)
    new_root = Path(new_root).resolve(strict=True)
    if not old_root.is_dir() or not new_root.is_dir():
        raise ValueError('old and new deployment roots must be directories')
    checks = {}

    def check(path, expected, label):
        key = (str(path), expected)
        if key not in checks:
            _check_file(path, expected, label)
            checks[key] = True

    def rebind_sources(sources):
        if not isinstance(sources, dict):
            raise ValueError('source_hashes must be a dictionary')
        rebound = {}
        for original, expected in sources.items():
            if not isinstance(original, str) or not isinstance(expected, str) or not _SHA256.fullmatch(expected):
                raise ValueError('source_hashes requires absolute paths and frozen SHA256 values')
            source = Path(original)
            if not source.is_absolute():
                raise ValueError('source_hashes requires absolute paths')
            relative = _relative(source, old_root)
            if relative is None:
                check(source, expected, 'external')
                destination_key = original
            else:
                if '..' in relative.parts:
                    raise ValueError('source path escapes the old deployment root')
                try:
                    resolved_source = source.resolve(strict=True)
                    destination = new_root / relative
                    resolved_destination = destination.resolve(strict=True)
                except OSError as exc:
                    raise ValueError(f'deployment source is unavailable: {original}') from exc
                if _relative(resolved_source, old_root) is None:
                    raise ValueError('source symlink escapes the old deployment root')
                if _relative(resolved_destination, new_root) is None:
                    raise ValueError('source symlink escapes the new deployment root')
                check(source, expected, 'original')
                check(destination, expected, 'rebound')
                destination_key = str(resolved_destination)
            if destination_key in rebound:
                raise ValueError('rebound source path collision')
            rebound[destination_key] = expected  # Never replace a frozen hash.
        return rebound

    def rewrite(value):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                if key == 'source_hashes':
                    result[key] = rebind_sources(child)
                elif key != 'fingerprint':
                    result[key] = rewrite(child)
            if 'fingerprint' in value:
                # Children have already been rebound and re-fingerprinted.
                result['fingerprint'] = fingerprint(result)
            return result
        if isinstance(value, list):
            return [rewrite(child) for child in value]
        return value

    result = rewrite(config)
    if _method_fields(result) != _method_fields(config):
        raise ValueError('rebinding changed fields outside source paths and fingerprints')
    _validate_fingerprints(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_config', type=Path)
    parser.add_argument('old_deploy_root', type=Path)
    parser.add_argument('new_deploy_root', type=Path)
    parser.add_argument('output_config', type=Path)
    args = parser.parse_args(argv)
    original = json.loads(args.input_config.read_text(encoding='utf-8'))
    bound = rebind_config(original, args.old_deploy_root, args.new_deploy_root)
    # Exclusive creation preserves every frozen config and prior binding.
    with args.output_config.open('x', encoding='utf-8') as output:
        json.dump(bound, output, indent=2, allow_nan=False)
        output.write('\n')


if __name__ == '__main__':
    main()
