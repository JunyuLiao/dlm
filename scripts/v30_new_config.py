"""Freeze a NEW V30 wrapper identity; preserve the byte-identical v20 parent.

Unlike a byte-identical rebind, this explicitly allows reviewed v21 wrapper
source changes and records new pins. Never use it to validate/repair old runs.
"""
import copy
import hashlib
from pathlib import Path
from scripts.v27_vllm_bind import rebind_config, fingerprint, _method_fields, _validate_fingerprints


def new_config(original, old_root, new_root, mode='temporal_T'):
    from experiments.numerical_qk_reuse import v21
    if mode not in ('temporal_T', 'unit_v30', 'confidence_v30'):
        raise ValueError('Unknown named mode')
    _validate_fingerprints(original)
    if original.get('sensitivity', 'temporal_T') != 'temporal_T':
        raise ValueError('Frozen temporal-T reference required')
    old_root, new_root = Path(old_root).resolve(), Path(new_root).resolve()
    module_root = Path(v21.__file__).resolve().parent
    if module_root != new_root/'experiments/numerical_qk_reuse':
        raise ValueError('Import the exact new deployment')
    # Verify every old wrapper pin before creating an independent identity.
    old_sources = original.get('source_hashes', {})
    if not old_sources:
        raise ValueError('Original wrapper provenance missing')
    for path, pin in old_sources.items():
        src = Path(path)
        if not src.resolve().is_relative_to(old_root):
            raise ValueError('Unexpected wrapper source outside old deployment')
        if hashlib.sha256(src.read_bytes()).hexdigest() != pin:
            raise ValueError('Old wrapper bytes changed')
        dst = new_root/src.relative_to(old_root)
        if src.name != 'v21.py' and hashlib.sha256(dst.read_bytes()).hexdigest() != pin:
            raise ValueError('Unreviewed wrapper dependency change')
    out = copy.deepcopy(original)
    out['parent_config'] = rebind_config(original['parent_config'], old_root, new_root)
    out['source_hashes'] = {str(module_root/name):hashlib.sha256((module_root/name).read_bytes()).hexdigest()
                            for name in v21.SOURCES}
    if mode != 'temporal_T':
        out['sensitivity'] = mode
    out['fingerprint'] = fingerprint(out)
    want = _method_fields(original)
    if mode != 'temporal_T':
        want['sensitivity'] = mode
    if _method_fields(out) != want:
        raise ValueError('Changes beyond declared sensitivity')
    v21.validate_effective(out, out['condition'])
    return out
