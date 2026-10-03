"""CPU lifecycle/coverage/raw-sample contracts, using a deterministic fake clock."""
from types import SimpleNamespace as NS
import unittest

from scripts.v29_component_timing import activate_monitor, measure_components, zero_counter_deltas


class Event:
    def __init__(self, log):self.log=log
    def record(self):self.log.append('record')
    def synchronize(self):self.log.append('end_sync')
    def elapsed_time(self, other):return .25


class ComponentTimingTests(unittest.TestCase):
    def test_warm_then_counter_then_all_samples_with_order_and_host_boundaries(self):
        log=[];clock=iter(x*.001 for x in range(100)).__next__
        torch=NS(cuda=NS(synchronize=lambda:log.append('warm_sync'),Event=lambda **kw:Event(log)))
        funcs={'a':lambda:log.append('a'),'b':lambda:log.append('b')}
        def snapshot():log.append('snapshot');return {'capture':0,'jit':7}
        result=measure_components(torch,funcs,repeats=4,warm=2,snapshot=snapshot,clock=clock)
        self.assertEqual(log[:6],['a','a','b','b','warm_sync','snapshot'])
        self.assertEqual(log[-1],'snapshot')
        rows=result['sample_rows'];self.assertEqual(len(rows),8)
        self.assertEqual([r['arm'] for r in rows],['a','b','b','a','a','b','b','a'])
        self.assertEqual([r['sample_ordinal'] for r in rows],list(range(8)))
        self.assertEqual(rows[0]['host_start_relative_s'],0.)
        self.assertAlmostEqual(rows[-1]['host_completed_relative_s'],.023)
        for r in rows:
            self.assertEqual(r['cuda_event_ms'],.25)
            self.assertAlmostEqual(r['host_call_ms'],1.)
            self.assertAlmostEqual(r['synchronized_host_ms'],2.)
        self.assertEqual(result['compilation_deltas'],{'capture':0,'jit':0})
        self.assertFalse(result['chrome_timeline_saved'])

    def test_actual_timed_monitor_delta_rejects_result(self):
        torch=NS(cuda=NS(synchronize=lambda:None,Event=lambda **kw:Event([])))
        values=iter([{'capture':0,'jit':1},{'capture':0,'jit':2}])
        with self.assertRaisesRegex(RuntimeError,'compilation/capture'):
            measure_components(torch,{'a':lambda:None},repeats=4,warm=1,snapshot=lambda:next(values))

    def test_missing_counter_receipt_is_not_silent_zero(self):
        for before,after in (({},{}),({'jit':0},{}),({'jit':0},{'jit':False}),({'jit':-1},{'jit':-1})):
            with self.assertRaises(RuntimeError):zero_counter_deltas(before,after)

    def test_monitor_activation_is_verified_and_not_assumed(self):
        calls=[]
        monitor=NS(activate=lambda **kw:calls.append(kw),is_active=lambda:True,
                   _mode='warn',_cutedsl_hook_installed=True)
        self.assertTrue(activate_monitor(monitor,lambda:lambda:None)['active'])
        self.assertEqual(calls,[{'mode':'warn','verbose':False}])
        with self.assertRaises(RuntimeError):activate_monitor(monitor,lambda:None)
        monitor._mode='error'
        with self.assertRaises(RuntimeError):activate_monitor(monitor,lambda:lambda:None)
        monitor._mode='warn';monitor._cutedsl_hook_installed=False
        with self.assertRaises(RuntimeError):activate_monitor(monitor,lambda:lambda:None)

    def test_bad_repetition_contract_rejected_before_calls(self):
        for warm,repeats in ((0,4),(1,3),(False,4)):
            with self.assertRaises(ValueError):measure_components(None,{'a':lambda:None},repeats=repeats,warm=warm,snapshot=lambda:{'jit':0})


if __name__=='__main__':unittest.main()
