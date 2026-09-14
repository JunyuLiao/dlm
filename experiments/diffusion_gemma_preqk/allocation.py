"""One coarse, paired layer-budget shift; no ranking or held-out fitting."""
from collections import defaultdict
import csv
import torch


def layer_sensitivities(path,benchmark):
    rows=[r for r in csv.DictReader(path.open()) if r['benchmark']==benchmark and
        r['method']=='last_mass_s50' and r['dimension']=='layer']
    return {int(r['group']):float(r['relative_error']) for r in rows}


def choose_groups(path,layer_types,benchmark='longbench'):
    scores=layer_sensitivities(path,benchmark);groups={}
    for kind in ('local','global'):
        ids=[i for i,t in enumerate(layer_types) if ('local' if t=='sliding_attention' else 'global')==kind]
        order=sorted(ids,key=lambda i:(scores[i],i));count=5 if kind=='local' else 1
        assert len(order)>=2*count
        groups[kind]=dict(low=order[:count],high=order[-count:])
    return dict(groups=groups,layer_kinds={str(i):'local' if t=='sliding_attention' else 'global' for i,t in enumerate(layer_types)},
        criterion='pooled development last-mass s50 same-state relative output error; observational, not causal layer sensitivity',
        sensitivities={str(i):v for i,v in scores.items()},shift_tiles=1)


class PairedLayerAllocation:
    def __init__(self,policy):
        self.policy=policy;self.kinds={int(k):v for k,v in policy['layer_kinds'].items()};self.signs={}
        for kind,g in policy['groups'].items():
            assert len(g['high'])==len(g['low']) and not set(g['high'])&set(g['low'])
            for sign,label in ((1,'high'),(-1,'low')):
                for layer in g[label]:
                    assert self.kinds[layer]==kind and layer not in self.signs
                    self.signs[layer]=sign
        assert policy['shift_tiles']==1
        self.pending={};self.audit=[];self.call=0

    def shift(self,layer,kind,eligible,sparsity,active):
        if layer==0:self.pending={}
        assert self.kinds[layer]==kind
        n=eligible.sum(-1);base=(n-torch.floor(n*sparsity).long()).clamp_min(1).minimum(n)
        entry=self.pending.setdefault(kind,dict(n=n.clone(),delta=torch.zeros_like(n),actual_delta=torch.zeros_like(n),base=0,layers=[]))
        if not torch.equal(n,entry['n']):raise RuntimeError('allocation assumes identical eligibility within attention type')
        amount=(base-1).clamp(0,1).minimum((n-base).clamp_min(0))
        delta=amount*self.signs.get(layer,0) if active else torch.zeros_like(n)
        entry['delta']+=delta;entry['base']+=int(base.sum()) if active else int(n.sum());entry['layers'].append(layer)
        return delta,base if active else n

    def observe(self,layer,kind,keep,baseline_budget):
        self.pending[kind]['actual_delta']+=keep.sum(-1)-baseline_budget
        if layer!=max(self.kinds):return
        for kind,entry in self.pending.items():
            if (entry['delta']!=0).any():raise RuntimeError('non-zero-sum layer allocation')
            actual=entry['actual_delta']
            self.audit.append(dict(forward=self.call,attention_type=kind,baseline_retained_tiles=entry['base'],
                quota_delta=0,actual_delta=int(actual.sum()),max_abs_row_delta=int(actual.abs().max()),
                layers=entry['layers']))
        self.call+=1
