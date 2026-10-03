"""CPU-only opt-in/private/no-overwrite trace tests; no torch import."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

from scripts import v29_vllm_cost_profile as base
from scripts.v29_trace_profile import (TracePlan, validate_trace_spec, validate_bound_tools,
                                      export_no_overwrite, trace_session_patch)


class TraceTests(unittest.TestCase):
    def config(self, root, shapes=False):
        return dict(diagnostic_only=True, profile_ordinal=1, chrome_trace_export=dict(
            enabled=True, record_shapes=shapes, with_stack=False, own_root=str(root),
            private_directory=str(root/'new_trace')))

    def test_default_disabled_and_cli_mismatch(self):
        self.assertIsNone(validate_trace_spec({},None,False,'new_run'))
        with self.assertRaises(ValueError): validate_trace_spec({},'private',False,'new_run')
        with self.assertRaises(ValueError): validate_trace_spec({},None,True,'new_run')

    def test_frozen_private_settings_and_shapes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); spec=self.config(root,True)
            plan=validate_trace_spec(spec,root/'new_trace',True,root/'new_run')
            self.assertTrue(plan.record_shapes)
            self.assertEqual(plan.output.name,'chrome_trace.private.json')
            with self.assertRaises(ValueError): validate_trace_spec(spec,root/'new_trace',True,root/'new_run',ordinal=2)
            with self.assertRaises(ValueError): validate_trace_spec(spec,root/'new_trace',False,root/'new_run')
            with self.assertRaises(ValueError): validate_trace_spec(dict(spec,diagnostic_only=False),root/'new_trace',True,root/'new_run')
            spec['chrome_trace_export']['with_stack']=True
            with self.assertRaises(ValueError): validate_trace_spec(spec,root/'new_trace',True,root/'new_run')

    def test_existing_output_outside_root_and_run_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); spec=self.config(root)
            target=root/'new_trace';target.mkdir();(target/'user_file').write_text('keep')
            with self.assertRaises(ValueError): validate_trace_spec(spec,target,False,root/'new_run')
            self.assertEqual((target/'user_file').read_text(),'keep')
            spec['chrome_trace_export']['private_directory']=str(root.parent/'outside')
            with self.assertRaises(ValueError): validate_trace_spec(spec,root.parent/'outside',False,root/'new_run')
            spec=self.config(root);spec['chrome_trace_export']['private_directory']=str(root/'new_run'/'trace')
            with self.assertRaises(ValueError): validate_trace_spec(spec,root/'new_run'/'trace',False,root/'new_run')

    def test_both_tools_must_be_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools=[Path(tmp)/'trace.py',Path(tmp)/'old.py']
            validate_bound_tools(dict(files={str(p):'expected' for p in tools}),tools)
            with self.assertRaises(ValueError):validate_bound_tools(dict(files={str(tools[0]):'expected'}),tools)

    def test_exclusive_export_keeps_existing_user_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'trace.private.json';target.write_text('user')
            prof=NS(export_chrome_trace=lambda _:self.fail('must reject before exporter'))
            with self.assertRaises(ValueError):export_no_overwrite(prof,target)
            self.assertEqual(target.read_text(),'user')

    def test_export_failure_cleans_own_temp_and_preserves_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'user').write_text('keep')
            def fail(path):Path(path).write_text('partial private trace');raise RuntimeError('export failed')
            with self.assertRaises(RuntimeError):export_no_overwrite(NS(export_chrome_trace=fail),root/'trace.private.json')
            self.assertEqual([p.name for p in root.iterdir()],['user'])

    def test_late_existing_output_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'trace.private.json'
            def late(path):Path(path).write_text('{}');target.write_text('late user file')
            with self.assertRaises(FileExistsError):export_no_overwrite(NS(export_chrome_trace=late),target)
            self.assertEqual(target.read_text(),'late user file')
            self.assertEqual([p.name for p in root.iterdir()],['trace.private.json'])

    def test_export_once_after_boundary_and_profiler_class_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[];export=[]
            class Prof:
                def __enter__(self):return self
                def __exit__(self,*_):pass
                def events(self):return []
                def export_chrome_trace(self,path):export.append(path);Path(path).write_text('{"traceEvents":[]}')
            def factory(**kwargs):calls.append(kwargs);return Prof()
            torch=NS(profiler=NS(profile=factory,record_function=lambda _:nullcontext(),
                                ProfilerActivity=NS(CPU=1,CUDA=2)))
            original=base.ProfileSession
            with trace_session_patch(base,TracePlan(root,False)):
                session=base.ProfileSession(torch,1);session.request_start()
                self.assertEqual(calls,[])
                session.request_start();self.assertFalse(calls[0]['record_shapes'])
                self.assertFalse(calls[0]['with_stack'])
                self.assertFalse((root/'chrome_trace.private.json').exists())
                session.stop(boundary_completed=True);session.stop(failed=True)
                self.assertEqual(len(export),1)
                receipt=json.loads((root/'trace_receipt.private.json').read_text())
                self.assertTrue(receipt['private']);self.assertFalse(receipt['automatically_published'])
            self.assertIs(base.ProfileSession,original)
            self.assertIs(torch.profiler.profile,factory)

    def test_failed_request_does_not_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            class Prof:
                def __enter__(self):return self
                def __exit__(self,*_):pass
                def events(self):return []
                def export_chrome_trace(self,_):raise AssertionError('failed request export forbidden')
            torch=NS(profiler=NS(profile=lambda **_:Prof(),record_function=lambda _:nullcontext(),
                                ProfilerActivity=NS(CPU=1,CUDA=2)))
            with trace_session_patch(base,TracePlan(root,False)):
                session=base.ProfileSession(torch,1);session.request_start();session.request_start()
                session.stop(failed=True)
            self.assertEqual(list(root.iterdir()),[])


if __name__=='__main__':unittest.main()
