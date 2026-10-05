"""Freeze a NEW V30 wrapper identity; preserve the v20 mathematical settings.

Unlike a byte-identical rebind, this explicitly allows reviewed v21 wrapper
source changes and records new pins. Never use it to validate/repair old runs.
"""
import copy
import hashlib
from pathlib import Path
from scripts.v27_vllm_bind import fingerprint, _method_fields, _validate_fingerprints

REVIEWED_CHANGES = frozenset(('experiments/numerical_qk_reuse/v21.py',
                              'experiments/numerical_qk_reuse/vllm_adapter.py'))


def migrate_source_identity(original, old_root, new_root):
    """Old pins are verified, never overwritten in place, including nested pins.

Some parent provenance also includes v21/adapter files. Their reviewed changes
must be named at every occurrence; every other old dependency stays identical.
"""
    _validate_fingerprints(original)
    old_root,new_root=Path(old_root).resolve(),Path(new_root).resolve()
    def visit(obj):
        if isinstance(obj,list):return [visit(x) for x in obj]
        if not isinstance(obj,dict):return obj
        out={}
        for key,value in obj.items():
            if key=='fingerprint':continue
            if key!='source_hashes':out[key]=visit(value);continue
            pins={}
            for name,pin in value.items():
                src=Path(name).resolve(strict=True)
                if hashlib.sha256(src.read_bytes()).hexdigest()!=pin:
                    raise ValueError('Original pinned bytes changed')
                if src.is_relative_to(old_root):
                    relative=src.relative_to(old_root);dst=(new_root/relative).resolve(strict=True)
                    if not dst.is_relative_to(new_root):raise ValueError('Destination escapes deployment')
                    actual=hashlib.sha256(dst.read_bytes()).hexdigest()
                    if actual!=pin and relative.as_posix() not in REVIEWED_CHANGES:
                        raise ValueError('Unreviewed source change: '+relative.as_posix())
                else:dst,actual=src,pin
                if str(dst) in pins:raise ValueError('Duplicate source alias')
                pins[str(dst)]=actual
            out[key]=pins
        if 'fingerprint' in obj:out['fingerprint']=fingerprint(out)
        return out
    result=visit(original)
    if _method_fields(result)!=_method_fields(original):raise ValueError('Method changed during source migration')
    return result


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
    out = migrate_source_identity(original,old_root,new_root)
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
