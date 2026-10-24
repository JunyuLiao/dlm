"""In-memory, in-place native state restoration for diagnostic branches only."""
from collections import deque
import enum
import hashlib
import types
import numpy as np
import torch


class Snapshot:
    def __init__(self, roots, shared=()):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.nodes = []
        self.storages = {}
        self.seen = set(map(id, shared))
        self.shared = {}
        self.pending = [(obj, name) for name, obj in roots.items()]
        while self.pending:
            obj, path = self.pending.pop()
            path = path[-240:]
            self.visit(obj, path)
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        self.storage_hashes = {key: self.digest(value[2]) for key, value in self.storages.items()}

    @staticmethod
    def digest(value):
        return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()

    def visit(self, obj, path):
        if id(obj) in self.seen:
            return
        self.seen.add(id(obj))
        if obj is None or isinstance(obj, (str, bytes, int, float, bool, enum.Enum,
                torch.dtype, torch.device, type, types.FunctionType, types.MethodType,
                types.BuiltinFunctionType, types.ModuleType)):
            return
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            key = (str(obj.device), storage.data_ptr(), storage.nbytes())
            if key not in self.storages and storage.nbytes():
                raw = torch.empty(0, dtype=torch.uint8, device=obj.device).set_(storage, 0, (storage.nbytes(),), (1,))
                self.storages[key] = (storage, raw, raw.cpu().clone())
            return
        if isinstance(obj, np.ndarray):
            self.nodes.append(('numpy', obj, obj.copy()))
            return
        if isinstance(obj, torch.Generator):
            self.nodes.append(('generator', obj, obj.get_state().clone()))
            return
        if isinstance(obj, dict):
            saved = list(obj.items())
            self.nodes.append(('dict', obj, saved))
            for i, (key, val) in enumerate(saved):
                self.pending.append((key, path+'.key'+str(i)))
                self.pending.append((val, path+'.'+str(key)))
            return
        if isinstance(obj, (list, deque, set)):
            saved = list(obj)
            self.nodes.append(('sequence', obj, saved))
            for i, val in enumerate(saved):
                self.pending.append((val, path+'.'+str(i)))
            return
        if isinstance(obj, tuple):
            for i, val in enumerate(obj):
                self.pending.append((val, path+'.'+str(i)))
            return
        module = type(obj).__module__
        if isinstance(obj, torch.nn.Module) or module.startswith(('vllm.config', 'tokenizers',
                'transformers', 'logging', 'threading', 'asyncio', 'weakref', 'torch.cuda')):
            self.shared[path] = type(obj).__qualname__
            return
        fields = dict(vars(obj)) if hasattr(obj, '__dict__') else {}
        for cls in type(obj).__mro__:
            slots = getattr(cls, '__slots__', ())
            if isinstance(slots, str):
                slots = (slots,)
            for key in slots:
                if key not in ('__dict__', '__weakref__') and hasattr(obj, key):
                    fields[key] = getattr(obj, key)
        if fields:
            self.nodes.append(('object', obj, fields))
            for key, value in fields.items():
                self.pending.append((value, path+'.'+key))
        else:
            self.shared[path] = type(obj).__qualname__

    def restore(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for kind, obj, saved in self.nodes:
            if kind == 'numpy':
                np.copyto(obj, saved)
            elif kind == 'generator':
                obj.set_state(saved)
            elif kind == 'dict':
                obj.clear()
                obj.update(saved)
            elif kind == 'sequence':
                obj.clear()
                obj.update(saved) if isinstance(obj, set) else obj.extend(saved)
            elif kind == 'object':
                if hasattr(obj, '__dict__'):
                    for key in set(vars(obj))-set(saved):
                        delattr(obj, key)
                for key, value in saved.items():
                    object.__setattr__(obj, key, value)
        with torch.no_grad():
            for storage, raw, saved in self.storages.values():
                raw.copy_(saved)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng:
            torch.cuda.set_rng_state_all(self.cuda_rng)

    def verify(self):
        errors = []
        for key, (_, raw, _) in self.storages.items():
            if self.digest(raw) != self.storage_hashes[key]:
                errors.append(str(key))
        return dict(passed=not errors, storage_mismatches=errors,
            storage_bytes=sum(k[2] for k in self.storages),
            storages=len(self.storages), python_nodes=len(self.nodes),
            shared_opaque_types=sorted(set(self.shared.values())))
