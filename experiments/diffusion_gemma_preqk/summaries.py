"""Producer-attached value norms and explicit encoder-cache identity.

Prefix V is scanned once at cache initialization (a charged initialization cost).
New canvas V is summarized at v_norm production, before attention. Only scalar
token norms, never full V, are subsequently read to form physical block summaries.
"""
import torch
import torch.nn.functional as F
from .config import TILE


def block_norms(token_norms):
    length=token_norms.shape[-1];pad=(-length)%TILE
    x=F.pad(token_norms,(0,pad)).unflatten(-1,(-1,TILE))
    count=F.pad(torch.ones(length,device=x.device),(0,pad)).unflatten(-1,(-1,TILE)).sum(-1)
    return {'rms':(x.square().sum(-1)/count).sqrt(),'max':x.amax(-1)}


class ProducerSummaries:
    def __init__(self,adapter):
        self.handles=[];self.layers={};self.prefix_value_elements=0;self.canvas_value_elements=0
        for name,module in adapter.model.named_modules():
            if not adapter.is_blasst_attention_module(name,module):continue
            self.handles.append(module.register_forward_pre_hook(self._prefix,with_kwargs=True))
            self.handles.append(module.v_norm.register_forward_hook(self._canvas_hook(module)))

    def _prefix(self,module,args,kwargs):
        cache=kwargs.get('past_key_values',args[3] if len(args)>3 else None)
        layer=int(module.layer_idx)
        if cache is None:
            signature=(None,);prefix=None
        else:
            c=cache.layers[layer];prefix=c.values
            signature=(id(cache),c.keys.data_ptr(),c.values.data_ptr(),tuple(c.keys.shape))
        previous=self.layers.get(layer)
        if previous is None or previous['identity']!=signature:
            norms=None if prefix is None else prefix.float().square().mean(-1).sqrt()
            self.prefix_value_elements+=0 if prefix is None else prefix.numel()
            self.layers[layer]={'identity':signature,'prefix':norms}

    def _canvas_hook(self,module):
        def hook(norm,args,output):
            # v_norm output is B,K,H,D, before the attention transpose/GQA repeat.
            self.layers[int(module.layer_idx)]['canvas']=output.float().square().mean(-1).sqrt().transpose(1,2)
            self.canvas_value_elements+=output.numel()
        return hook

    def get(self,module,heads):
        entry=self.layers[int(module.layer_idx)];prefix=entry['prefix'];canvas=entry['canvas']
        norms=canvas if prefix is None else torch.cat((prefix,canvas),-1)
        norms=norms.repeat_interleave(heads//norms.shape[1],dim=1)
        return block_norms(norms),entry['identity']

    @property
    def bytes(self):
        return sum(t.numel()*t.element_size() for e in self.layers.values() for k,t in e.items() if k in ('prefix','canvas') and t is not None)

    def close(self):
        for h in self.handles:h.remove()
