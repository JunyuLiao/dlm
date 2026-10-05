"""v11 D_mask diagnostic: dense attention with EXACTLY the legacy legal mask.

Same legality as fresh T / H1 / H3 (query-relative LOCAL window
key >= q + (K - Q) - window + 1; GLOBAL unrestricted), dense over all legal
keys. It exists to control the native-vs-legacy LOCAL mask gap in quality and
trajectory; it is NOT a speed baseline (it runs through the BLASST dispatcher
and a boolean-mask SDPA, and pays an idle observe wrapper). Runner plug-in.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
from experiments.value_direction_hopper.query_adaptive import State


class DenseMask:
    def __init__(self):
        self.masks, self.calls = {}, 0

    @torch.no_grad()
    def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None, is_causal=None,
                 sliding_window=None, **kwargs):
        if dropout or module.training or mask is not None or is_causal:
            raise ValueError('D_mask is qualified for the bidirectional unmasked decoder call only')
        nq, nk = q.shape[2], k.shape[2]
        legal = None
        if sliding_window:
            key = (nq, nk, int(sliding_window), q.device)
            if key not in self.masks:
                qi = torch.arange(nq, device=q.device)[:, None]
                kk = torch.arange(nk, device=q.device)[None, :]
                self.masks[key] = kk >= qi + (nk - nq) - int(sliding_window) + 1
            legal = self.masks[key]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=legal, scale=scaling,
                                             enable_gqa=q.shape[1] != k.shape[1])
        self.calls += 1
        return out.transpose(1, 2).contiguous(), None

    def counters(self):
        return dict(attention_calls=self.calls, mask='legacy query-relative LOCAL window; GLOBAL unrestricted')


@contextmanager
def install(adapter, config, condition):
    if condition != 'dense_mask':
        raise ValueError(condition)
    binding = _install_dense(adapter)
    router = DenseMask()
    try:
        binding.runtime.attention_override = router
        state = State('native_dense', None, m_ref=config['m_ref'], beta=config['beta'], gamma=config['gamma'],
                      diagnostics=False)
        yield dict(binding=binding, router=router, state=state, counters=router.counters)
    finally:
        binding.close()
