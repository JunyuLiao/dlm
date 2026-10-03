"""GLOBAL-only scope for isolated Junyu HF reproduction (not a vLLM port).

Supply the install context from the separately pinned peer export. Restricting
the adapter selector prevents LOCAL modules from being tagged by its registry
dispatcher or sketch hooks. Native encoder/commit eligibility remains with the
original adapter. No attention or sampling mathematics is changed here.
"""
from contextlib import contextmanager

GLOBAL_LAYERS = (5, 11, 17, 23, 29)


class GlobalScopeAdapter:
    def __init__(self, adapter):
        self._adapter = adapter
        selected = {}
        for name, module in adapter.model.named_modules():
            if not adapter.is_blasst_attention_module(name, module):
                continue
            layer = int(module.layer_idx)
            if layer not in GLOBAL_LAYERS:
                continue
            if layer in selected:
                raise ValueError("Duplicate GLOBAL decoder layer")
            if getattr(module, "is_sliding", False):
                raise ValueError("Expected GLOBAL layer is marked sliding")
            if getattr(module, "_blasst_2d_runtime", None) is not None:
                raise RuntimeError("GLOBAL decoder already bound")
            selected[layer] = module
        if tuple(sorted(selected)) != GLOBAL_LAYERS:
            raise ValueError("Expected the five DiffusionGemma GLOBAL decoder layers")
        self._selected = selected

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def is_blasst_attention_module(self, name, module):
        return any(module is selected for selected in self._selected.values())


class ScopeCounter:
    def __init__(self, router, adapter):
        self.router = router
        self.adapter = adapter
        self.calls = {layer: 0 for layer in GLOBAL_LAYERS}
        self.unexpected_calls = 0

    def __call__(self, module, *args, **kwargs):
        if not self.adapter.is_blasst_attention_module("", module):
            self.unexpected_calls += 1
            raise RuntimeError("Non-GLOBAL module reached peer value router")
        result = self.router(module, *args, **kwargs)
        self.calls[int(module.layer_idx)] += 1
        return result

    def receipt(self):
        return dict(effective_scope="five_GLOBAL_decoder_layers_only",
                    successful_calls_by_layer=dict(self.calls),
                    unexpected_router_calls=self.unexpected_calls,
                    scope_qualified=all(self.calls.values()) and self.unexpected_calls == 0)


@contextmanager
def install_global_value(adapter, peer_install, library, thresholds, **kwargs):
    """Keep peer install/cleanup and original call selector, with checked scope.

    LOCAL native passthrough must still receive a GPU numerical qualification.
    The returned scope receipt establishes routing only, not output accuracy.
    """
    scoped = GlobalScopeAdapter(adapter)
    with peer_install(scoped, library, thresholds, **kwargs) as (binding, router):
        counter = ScopeCounter(router, scoped)
        binding.runtime.attention_override = counter
        yield binding, router, counter
