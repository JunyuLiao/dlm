"""Raw-trace invariants, not merely existence/count-based completion checks."""
from collections import defaultdict
import numpy as np
from .config import TILE
from .routing import FIELDS


def validate_trace(record,*,heads=None):
    assert tuple(record['fields'])==FIELDS
    with np.load(record['path']) as raw:
        data=raw['metrics'];n=len(record['names'])
        assert data.ndim==5 and data.shape[0]==n and data.shape[-1]==len(FIELDS)
        assert data.shape[3]==(record['kv_length']-record['prefix_length']+TILE-1)//TILE
        if heads is not None:assert data.shape[2]==heads
        assert np.isfinite(data).all() and (data>=0).all()
        assert (data[...,1]<=data[...,0]).all()
        assert np.array_equal(data[...,15]+data[...,17]+data[...,19],data[...,0])
        assert np.array_equal(data[...,16]+data[...,18]+data[...,20],data[...,1])
        assert np.array_equal(data[...,0],np.broadcast_to(data[0,...,0],data[...,0].shape))
        e=raw['eligible'];masks=raw['selected_q0']
        assert masks.shape[:3]==data.shape[:3] and masks.shape[-1]==e.shape[-1]
        assert not (masks & ~e[None]).any()
        assert np.array_equal((e[None]&~masks).sum(-1),data[...,0,1])
        for i,name in enumerate(record['names']):
            target=int(name.rsplit('_s',1)[1])/100
            regular=(data[i,...,9]==0)&(data[i,...,10]==0)
            assert np.array_equal(data[i,...,1][regular],np.floor(data[i,...,0]*target)[regular])
        return dict(heads=data.shape[2],query_tiles=data.shape[3],candidate_masks=n,
            eligible=float(data[0,...,0].sum()),mask_checks=n*data.shape[1]*data.shape[2]*data.shape[3])


def validate_consecutive_coverage(records,*,layers,heads=None):
    by_layer=defaultdict(list);mask_checks=0
    for record in records:
        by_layer[record['layer']].append(record)
        mask_checks+=validate_trace(record,heads=heads)['mask_checks']
    assert set(by_layer)==set(range(layers)), 'missing decoder layers'
    assert {r['attention_type'] for r in records}=={'local','global'}
    sequences=[]
    for layer,rows in by_layer.items():
        steps=[r['step'] for r in rows]
        assert steps==list(range(len(steps))) and len(steps)>=2,'missing or nonconsecutive dense-history steps'
        assert not rows[0]['history_available'] and all(r['history_available'] for r in rows[1:]),'history cache alignment failed'
        sequences.append(steps)
    assert all(s==sequences[0] for s in sequences),'layer coverage differs across steps'
    return dict(layers=layers,heads=heads,steps=sequences[0],physical_mask_checks=mask_checks)
