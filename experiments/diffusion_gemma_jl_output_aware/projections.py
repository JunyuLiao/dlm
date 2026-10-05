"""Fixed CPU-generated matrices and exact-content-validated native-head caches."""
from dataclasses import asdict
import hashlib
import math

import torch

from .config import Config


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class Projections:
    def __init__(self):
        self.matrices = {}
        self.banks = {}
        self.manifest = {}

    def get(self, layer, heads, width, family, rank, seed, device):
        key = (layer, heads, width, family, rank, seed)
        if key not in self.matrices:
            parts, records = [], []
            for head in range(heads):
                material = f'jl_output_v1/{family}/{rank}/{seed}/{layer}/{head}/{width}'
                derived = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], 'little') % (2**63-1)
                gen = torch.Generator(device='cpu').manual_seed(derived)
                if family == 'identity':
                    matrix = torch.eye(width, dtype=torch.float32)
                elif family == 'gaussian':
                    matrix = torch.randn(width, rank, generator=gen, dtype=torch.float32) / math.sqrt(rank)
                elif family == 'sign':
                    matrix = (torch.randint(0, 2, (width, rank), generator=gen).float()*2-1) / math.sqrt(rank)
                else:
                    raise ValueError(family)
                parts.append(matrix)
                records.append(dict(layer=layer, native_kv_head=head, value_width=width,
                    family=family, rank=matrix.shape[-1], seed=seed, derived_seed=derived, sha256=digest(matrix)))
            self.matrices[key] = torch.stack(parts).to(device)
            self.manifest[str(key)] = records
        matrix = self.matrices[key]
        if matrix.device != torch.device(device):
            matrix = matrix.to(device)
            self.matrices[key] = matrix
        return matrix

    def get_nested_bank(self, layer, heads, width, max_rank, seed, device,
                        family='gaussian'):
        """Return one unscaled, reproducible bank shared by all cascade ranks.

        The previous ``get`` API intentionally creates independent matrices for
        each rank and remains unchanged for historical experiments.  Adaptive
        routing calls this method and obtains genuinely nested projections by
        taking the first ``r`` columns and dividing by ``sqrt(r)``.
        """
        if family != 'gaussian':
            raise ValueError('nested cascade currently supports Gaussian banks only')
        key = (layer, heads, width, family, int(max_rank), int(seed))
        if key not in self.banks:
            parts, records = [], []
            for head in range(heads):
                material = f'jl_output_nested_v1/{family}/{max_rank}/{seed}/{layer}/{head}/{width}'
                derived = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], 'little') % (2**63-1)
                gen = torch.Generator(device='cpu').manual_seed(derived)
                # This is deliberately unscaled; each prefix receives its own
                # 1/sqrt(r) normalization at use time.
                bank = torch.randn(width, max_rank, generator=gen, dtype=torch.float32)
                parts.append(bank)
                records.append(dict(layer=layer, native_kv_head=head, value_width=width,
                    family='gaussian_nested', rank=max_rank, seed=seed,
                    derived_seed=derived, sha256=digest(bank)))
            self.banks[key] = torch.stack(parts)
            self.manifest[str(('nested_bank',) + key)] = records
        bank = self.banks[key]
        if bank.device != torch.device(device):
            bank = bank.to(device)
            self.banks[key] = bank
        return bank

    def get_nested(self, layer, heads, width, rank, max_rank, seed, device):
        """Materialize R_r=W[:,:r]/sqrt(r) from the frozen bank."""
        if not (1 <= int(rank) <= int(max_rank)):
            raise ValueError('rank must be within nested bank')
        bank = self.get_nested_bank(layer, heads, width, max_rank, seed, device)
        return bank[..., :int(rank)] / math.sqrt(int(rank))


class SketchCache:
    """Generation-local, native KV head/token cache; never reuses stale values.

    Complete unchanged prefix blocks can reuse. Boundary and canvas always
    refresh. Content equality, not tensor identity or a guessed position, is
    the authority. This exact validation has an explicit engineering cost.
    """
    def __init__(self, config=None, projections=None):
        self.config = config or Config()
        self.projections = projections or Projections()
        self.entries = {}
        self.reused_blocks = 0
        self.refreshed_blocks = 0
        self.work = dict(projected_tokens=0, reused_tokens=0, refreshed_tokens=0,
                         projection_madds=0, value_comparison_elements=0,
                         sketch_storage_bytes=0, validation_copy_bytes=0,
                         peak_sketch_storage_bytes=0, peak_validation_copy_bytes=0)

    def _project(self, layer, value, valid):
        c = self.config
        x = value.float()
        h, d = x.shape[1], x.shape[-1]
        if c.method == 'mass_exact':
            z = torch.zeros((*x.shape[:-1], 1), dtype=torch.float32, device=x.device)
        elif c.family == 'identity':
            # This explicitly expensive control is not a low-dimensional router.
            self.projections.get(layer, h, d, c.family, d, c.projection_seed, x.device)
            z = x
        elif c.method == 'adaptive':
            # Cache the largest unscaled bank once.  The adaptive route applies
            # the rank-specific 1/sqrt(r) normalization to nested prefixes.
            bank = self.projections.get_nested_bank(layer, h, d, c.rank,
                                                    c.projection_seed, x.device)
            z = torch.matmul(x, bank) / math.sqrt(c.rank)
            self.work['projection_madds'] += x.numel()*z.shape[-1]
            self.work['projected_tokens'] += x.shape[0]*h*x.shape[-2]
        else:
            matrix = self.projections.get(layer, h, d, c.family, c.rank, c.projection_seed, x.device)
            z = torch.matmul(x, matrix)
            if c.method == 'cancellation_guard':
                second = self.projections.get(layer, h, d, c.family, c.rank, c.guard_seed, x.device)
                z = torch.cat((z, torch.matmul(x, second)), -1)
            self.work['projection_madds'] += x.numel()*z.shape[-1]
            self.work['projected_tokens'] += x.shape[0]*h*x.shape[-2]
        norm_sq = x.norm(dim=-1).square().masked_fill(~valid, 0.)
        primary = z[..., :c.rank] if c.method == 'cancellation_guard' else z
        return dict(z=z, norm_sq=norm_sq, projected_norm=primary.norm(dim=-1), valid=valid)

    def get(self, layer, values, valid, prefix):
        if values.ndim != 4 or valid.shape != values.shape[:-1]:
            raise ValueError('Cache expects B,native_KVH,K,D and its token validity')
        if values.shape[0] != 1:
            raise ValueError('Experiment is batch one')
        end = min(values.shape[-2], max(0, prefix)//64*64)
        old = self.entries.get(layer)
        reuse = False
        if end and old is not None and old[0].shape == values[..., :end, :].shape:
            self.work['value_comparison_elements'] += old[0].numel()
            reuse = torch.equal(old[0], values[..., :end, :]) and torch.equal(old[1], valid[..., :end])
        if reuse:
            a = old[2]
            self.reused_blocks += end//64
            self.work['reused_tokens'] += a['z'].shape[0]*a['z'].shape[1]*end
        elif end:
            a = self._project(layer, values[..., :end, :], valid[..., :end])
            self.entries[layer] = (values[..., :end, :].clone(), valid[..., :end].clone(), a)
            self.refreshed_blocks += end//64
            self.work['refreshed_tokens'] += values.shape[0]*values.shape[1]*end
        if end < values.shape[-2]:
            b = self._project(layer, values[..., end:, :], valid[..., end:])
            self.refreshed_blocks += (values.shape[-2]-end+63)//64
            self.work['refreshed_tokens'] += values.shape[0]*values.shape[1]*(values.shape[-2]-end)
        result = b if not end else a if end == values.shape[-2] else {
            key: torch.cat((a[key], b[key]), -2 if key == 'z' else -1) for key in a}
        result = dict(result)
        result['ref'] = (result['norm_sq'].sum(-1)/valid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)
        self.work['sketch_storage_bytes'] = sum(t[2]['z'].numel()*4 for t in self.entries.values())
        self.work['validation_copy_bytes'] = sum(t[0].numel()*t[0].element_size() for t in self.entries.values())
        for field in ('sketch_storage_bytes', 'validation_copy_bytes'):
            self.work['peak_'+field] = max(self.work['peak_'+field], self.work[field])
        return result
