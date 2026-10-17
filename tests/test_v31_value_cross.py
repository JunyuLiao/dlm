from types import SimpleNamespace
from scripts.v31_value_cross_study import schedule
from scripts.v31_value_cross_worker import bind


def test_cross_covers_all_cells_backends_without_conditioning():
    pool=[dict(index=i,seed=42) for i in range(503)]
    orders=[]
    for engine in range(4):
        cells,arms=schedule(pool,engine)
        assert len(cells)==28 and len(arms)==29
        assert arms[0]['backend']=='control'
        for start in range(4,28,6):
            group=arms[start+1:start+7]
            assert group[0]['label']=='control_before'
            assert group[-1]['label']=='control_after'
            assert {(x['backend'],x['mode']) for x in group[1:-1]}=={
                (b,m) for b in ('reference','device_exact') for m in ('full_swap8','residual_direction_guard')}
            assert len({(c['index'],c['seed']) for c in cells[start:start+6]})==1
        assert {(c['index'],c['seed']) for c in cells[4:]}=={(i,s) for i in (0,49) for s in (1,42)}
        orders.append([e['label'] for e in arms[5:11]])
    assert len({tuple(o) for o in orders})>1


def test_dispatch_resets_and_preserves_native_output_and_control():
    events=[]
    class Runtime:
        def __init__(self,name):self.name=name;self.model='model';self.mode=None
        def reset(self):events.append((self.name,'reset'))
        def output(self,*args):events.append((self.name,self.mode));return self.name
        def receipt(self):return {'value_refinement':self.mode}
    fast,reference=Runtime('fast'),Runtime('reference')
    adapter=SimpleNamespace(mage_state={},mage_k=8192,begin_request=lambda:fast.reset())
    arms=[dict(backend='control',mode='current_v31_control',label='control'),
          dict(backend='reference',mode='full_swap8',label='reference_full_swap8'),
          dict(backend='device_exact',mode='residual_direction_guard',label='device_residual')]
    calls=[]
    def native(*args):calls.append(args);return 'native'
    bind(adapter,fast,reference,native,lambda:{'baseline':True},arms)
    for i,expected in enumerate(('native','reference','fast')):
        adapter.begin_request()
        assert adapter._mage(29,None,None,None,100,256)==expected
        assert adapter._value_receipt()['cross_request_index']==i
    assert len(calls)==3
    assert events.count(('fast','reset'))==3
    assert events.count(('reference','reset'))==3
    assert ('reference','full_swap8') in events
    assert ('fast','residual_direction_guard') in events
