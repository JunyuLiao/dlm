"""Public toy data only. Run with stdlib unittest; no torch/GPU/SSH needed."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import types
import shutil
import unittest
from unittest.mock import patch

from scripts import v29_expanded_panel as run
from scripts import v29_expanded_summary as summary
from scripts.v27_vllm_bind import fingerprint
from scripts.v28_jit_receipts import COUNT_FIELDS


def dump(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def fake_receipt(arm, settings, n=10):
    if arm == 'dense':
        return None
    snapshot = dict(num_alloc_retries=0, num_ooms=0, allocated_bytes=64,
                    reserved_bytes=128, peak_allocated_bytes=64, peak_reserved_bytes=128)
    adapter = dict(begins=n, observes=n, global_calls=5*n, invalidates=1, order_errors=0,
                   split_fa4_calls=5, v28_config=settings, kv_probe_timed_checks=0,
                   allocator=run.qualify.allocator_receipt(snapshot, snapshot),
                   kv_layout_checks=[dict(layer=layer, max_abs_error=0, page_size=64) for layer in [5,11,17,23,29]])
    method = dict(effective_method=dict(run.METHOD), active_layers=[5,11,17,23,29],
                  gated_native_calls=0, layer_native_calls=0, unsupported_mask_refreshes=0,
                  fused_observations=1, dp_routes=1)
    return dict(adapter=adapter, method=method if arm == 'method' else None, timing=None)


class Fixture:
    def __init__(self, root, hosts=('toy_host',), qualification=False):
        self.root = root
        self.spec = run.build_spec('aime', 'public_toy', list(range(8101,8109)), list(hosts))
        dump(root/'public_spec.json', self.spec)
        rows = [dict(id=f'aime26/{i}', prompt='Public toy problem.', prompt_tokens=[1,2],
                     generation_budget=8192, thinking=True, expected='42') for i in range(1,31)]
        dump(root/'manifest.private.json', rows)
        dump(root/'gold.private.json', [dict(id=row['id'], expected='42') for row in rows])
        dump(root/'catalog.private.json', dict(aime26=dict(manifest=str(root/'manifest.private.json'),
                                                        gold=str(root/'gold.private.json'))))
        self.bindings = {}
        for host in hosts:
            source = root/host/'source'; source.mkdir(parents=True)
            (source/'DEPLOY_SHA').write_text('a'*40+'\n')
            for relative in run.SOURCES:
                path = source/relative; path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('Public toy source.\n')
            model = root/host/'model'; model.mkdir()
            dump(model/'config.json', {'public_toy':True}); dump(model/'generation_config.json', {'public_toy':True})
            config = dict(q_block=128,q_regroup=False,q_carry64=False,condition='public_toy',
                          source_hashes={str(source/'scripts/v27_vllm_panel_run.py'):run.panel.digest(source/'scripts/v27_vllm_panel_run.py')})
            config['fingerprint']=fingerprint(config)
            dump(root/host/'config.private.json',config)
            out = root/host/'frozen'
            run.freeze_private(root/'public_spec.json',root/'catalog.private.json',root/host/'config.private.json',
                               model,source,host,'PUBLIC_TOY_UUID',out)
            self.bindings[host] = out/'binding.private.json'
        self.entries=[]
        blocks=sorted({next(i for i,b in enumerate(self.spec['blocks']) if b['host']==host) for host in hosts}) if qualification else range(8)
        for block in blocks:
            host=self.spec['blocks'][block]['host']
            binding=run.panel.read(self.bindings[host]); config=run.panel.read(binding['config'])
            settings=run.execution_settings(self.spec)
            for arm in run.ARMS:
                directory=root/f'b{block}_{arm}'; directory.mkdir()
                run_id=directory.name
                records=[]; completions=[]
                for index in ([0] if qualification else range(30)):
                    row=dict(schema=run.old.SCHEMA,protocol_id=self.spec['protocol_id'],deploy_commit='a'*40,
                             host=host,gpu_uuid='PUBLIC_TOY_UUID',arm=arm,engine_seed=self.spec['blocks'][block]['engine_seed'],
                             seed_applied=False,measurement_mode='request_boundary_sync',run_id=run_id,
                             qualification_only=qualification,dataset='aime26',index=index,repeat=0 if qualification else block,
                             adapter_sha256=None if arm=='dense' else binding['source_files']['experiments/numerical_qk_reuse/vllm_adapter.py'],
                             method_fingerprint=config['fingerprint'] if arm=='method' else None,
                             torch='public_toy_torch',vllm='public_toy_vllm',wall_s=1.,prefill_s=.25,decode_span_s=.75,
                             denoise_forward_count=10,scheduler_denoise_forward_count=9,speculative_unused_denoising=1,
                             commit_forward_count=2,execution_count_source='vllm_existing_async_cpu_snapshot',
                             output_tokens=12,finish_reason='eos',graph_captures_timed=0,
                             compilation_deltas=dict(num_backend_compilations=0,num_inductor_compiles=0,
                                 **{'jit_monitor_'+key:0 for key in COUNT_FIELDS}),receipts=fake_receipt(arm,settings),
                             **self.spec['settings'],**self.spec['arm_settings'][arm])
                    records.append(row)
                    completions.append(dict(dataset='aime26',index=index,repeat=0 if qualification else block,arm=arm,run_id=run_id,
                                            id=f'aime26/{index+1}',completion='<channel|>Answer: 42',finish_reason='eos'))
                self.write_rows(directory/'records.jsonl',records)
                self.write_rows(directory/'completions.private.jsonl',completions)
                dump(directory/'terminal.json',dict(protocol_id=self.spec['protocol_id'],arm=arm,block=block,
                     run_id=run_id,complete=True,mode='qualification' if qualification else 'benchmark',
                     qualification_dataset='aime26' if qualification else None,v29_config=settings,
                     expected_timed=len(records),completed_timed=len(records),gpu_reserved_seconds=1.))
                self.entries.append(dict(binding=str(self.bindings[host]),run_dir=str(directory)))
        self.family=root/'family.private.json'; dump(self.family,dict(workers=self.entries))

    @staticmethod
    def write_rows(path,rows):
        Path(path).write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def mutate(self, field, value, block=0,arm='method'):
        path=self.root/f'b{block}_{arm}'/'records.jsonl'
        rows=list(run.old._jsonl([path])); rows[0][field]=value; self.write_rows(path,rows)


class SpecTests(unittest.TestCase):
    def test_complete_counts_and_budget(self):
        for suite,requests in [('longbench',1888),('aime',960),('humaneval',5248)]:
            spec=run.build_spec(suite,'public_toy',list(range(8)),['toy_host'])
            self.assertEqual(len(run.old.expected_inventory(spec)),requests)
            self.assertTrue(all(value['budget']==8192 for value in spec['task_contracts'].values()))
            self.assertEqual(spec['adapter_settings']['kv_copy_backend'],'torch')
            self.assertEqual(spec['adapter_settings']['merge_backend'],'torch')

    def test_reject_duplicate_seed(self):
        with self.assertRaises(ValueError): run.build_spec('aime','toy',[1]*8,['toy_host'])

    def test_reject_missing_controls_or_margin_or_bool_threads(self):
        original=run.build_spec('aime','toy',list(range(8)),['toy_host'])
        for mutate in (lambda s:s['control_indices']['aime26'].pop(),
                       lambda s:s['accuracy_analysis'].update(noninferiority_margin=.05),
                       lambda s:s['settings'].update(cpu_threads=True)):
            value=deepcopy(original); mutate(value)
            with self.assertRaises(ValueError): run.validate_spec(value)

    def test_token_and_gold_separation(self):
        rows=[dict(id=f'aime26/{i}',prompt_tokens=[1],generation_budget=8192,expected='public toy') for i in range(1,31)]
        clean=run.validate_manifest(rows,'aime26',[row['id'] for row in rows],8192)
        self.assertTrue(all('expected' not in row for row in clean))
        rows[0]['prompt_tokens']=[True]
        with self.assertRaises(ValueError): run.validate_manifest(rows,'aime26',[row['id'] for row in rows],8192)

    def test_short_task_budget_cannot_silently_change(self):
        rows=[dict(id=f'humaneval/{i}',prompt_tokens=[1],generation_budget=2048) for i in range(164)]
        with self.assertRaises(ValueError): run.validate_manifest(rows,'humaneval',[row['id'] for row in rows],8192)

    def test_qualification_selector_restores_on_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cells.json'; dump(path,[dict(dataset='aime26'),dict(dataset='humaneval')])
            before=run.panel.read
            with self.assertRaises(RuntimeError):
                with run.qualification_cells({'cells':str(path)},'aime26'):
                    self.assertEqual(run.panel.read(path),[dict(dataset='aime26')]); raise RuntimeError('toy')
            self.assertIs(run.panel.read,before)

    def test_backend_constructor_and_restore(self):
        class Adapter:
            def __init__(self,kv_copy_backend='torch',merge_backend='torch'):
                self.kv_copy_backend=kv_copy_backend; self.merge_backend=merge_backend
        module=types.SimpleNamespace(VllmMethodAdapter=Adapter)
        settings=run.execution_settings(run.build_spec('aime','toy',list(range(8)),['toy_host']))
        validator=run.panel.validate_receipts
        with patch.object(run.qualify,'adapter_variant',side_effect=lambda base,*args:base):
            with self.assertRaises(RuntimeError):
                with run.instrument_runner(module,settings,1):
                    self.assertEqual(module.VllmMethodAdapter().merge_backend,'torch'); raise RuntimeError('toy')
        self.assertIs(module.VllmMethodAdapter,Adapter); self.assertIs(run.panel.validate_receipts,validator)

    def test_backend_receipt_rejects_wrong_execution(self):
        settings=run.execution_settings(run.build_spec('aime','toy',list(range(8)),['toy_host']))
        with patch.object(run.qualify,'validate_variant_receipt'):
            for key in ('triton_kv_copy_calls','triton_kv_copy_elements','triton_lse_merge_calls'):
                with self.assertRaises(ValueError):run.validate_receipt('allkept',{'adapter':{key:1}},10,run.METHOD,settings)
            settings.update(kv_copy_backend='triton',merge_backend='triton')
            with self.assertRaises(ValueError):run.validate_receipt('allkept',{'adapter':{}},10,run.METHOD,settings)
            run.validate_receipt('allkept',{'adapter':dict(triton_kv_copy_calls=1,triton_kv_copy_elements=2,
                                                        triton_lse_merge_calls=1)},10,run.METHOD,settings)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.fixture=Fixture(Path(self.temp.name))

    def test_complete_inventory_and_no_generation_gold_dependency(self):
        (self.fixture.root/'gold.private.json').unlink()
        cells,completions,spec,binding=summary.load_family(self.fixture.family)
        self.assertEqual(len(cells),240); self.assertEqual(len(completions),960)
        self.assertTrue(all('expected' not in row for row in run.panel.read(binding['manifests']['aime26'])))

    def test_reject_missing_worker(self):
        dump(self.fixture.family,dict(workers=self.fixture.entries[:-1]))
        with self.assertRaises(ValueError):summary.load_family(self.fixture.family)

    def test_reject_duplicate_worker(self):
        dump(self.fixture.family,dict(workers=self.fixture.entries+[self.fixture.entries[0]]))
        with self.assertRaises(ValueError):summary.load_family(self.fixture.family)

    def test_reject_open_terminal(self):
        path=self.fixture.root/'b0_method'/'terminal.json'; value=run.panel.read(path); value['complete']=False;dump(path,value)
        with self.assertRaises(ValueError):summary.load_family(self.fixture.family)

    def test_reject_duplicate_cell_even_if_terminal_count_changed(self):
        path=self.fixture.root/'b0_method'/'records.jsonl'; rows=list(run.old._jsonl([path]));rows.append(rows[0]);self.fixture.write_rows(path,rows)
        terminal=path.parent/'terminal.json'; value=run.panel.read(terminal);value.update(expected_timed=31,completed_timed=31);dump(terminal,value)
        with self.assertRaises(ValueError):summary.load_family(self.fixture.family)

    def test_reject_source_drift(self):
        binding=run.panel.read(next(iter(self.fixture.bindings.values())))
        (Path(binding['deploy'])/'scripts/v27_vllm_metrics.py').write_text('changed public toy')
        with self.assertRaises(ValueError):summary.load_family(self.fixture.family)

    def test_reject_seed_pseudopairing(self):
        self.fixture.mutate('engine_seed',8101,block=1)
        with self.assertRaises(ValueError):summary.load_family(self.fixture.family)

    def test_reject_real_count_mismatch(self):
        self.fixture.mutate('denoise_forward_count',9)
        with self.assertRaises(ValueError):summary.load_family(self.fixture.family)

    def test_reject_host_or_settings_drift(self):
        self.fixture.mutate('cpu_threads',True)
        with self.assertRaises(ValueError):summary.load_family(self.fixture.family)

    def test_reject_timed_jit(self):
        path=self.fixture.root/'b0_method'/'records.jsonl'; rows=list(run.old._jsonl([path]));rows[0]['compilation_deltas']['jit_monitor_triton_events']=1;self.fixture.write_rows(path,rows)
        with self.assertRaises(ValueError):summary.load_family(self.fixture.family)

    def test_reject_qualification_leak(self):
        self.fixture.mutate('qualification_only',True)
        with self.assertRaises(ValueError):summary.load_family(self.fixture.family)

    def test_aime_delegation_and_cap_semantics(self):
        module=types.ModuleType('experiments.diffusion_gemma_aime26_modes.protocol')
        calls=[]
        module.final_response=lambda raw,thinking:calls.append((raw,thinking)) or 'public toy final'
        module.numeric_score=lambda text,expected:dict(correct=True,extracted='42')
        binding=run.panel.read(next(iter(self.fixture.bindings.values())))
        cells={('aime26',0,0):{'dense':{}},('aime26',1,0):{'dense':{}}}
        completions={('aime26',i,0,'dense'):dict(id=f'aime26/{i+1}',completion='Public toy',finish_reason=term)
                     for i,term in enumerate(('eos','length'))}
        with patch.dict('sys.modules',{'experiments.diffusion_gemma_aime26_modes.protocol':module}), patch.object(summary,'scorer_selftest',return_value={'aime26':True}):
            summary.score_tasks(cells,completions,self.fixture.spec,binding,{'aime26':self.fixture.root/'gold.private.json'})
        self.assertTrue(cells[('aime26',0,0)]['dense']['strict_correct'])
        self.assertFalse(cells[('aime26',1,0)]['dense']['strict_correct'])
        self.assertTrue(cells[('aime26',1,0)]['dense']['task_correct']);self.assertEqual(len(calls),2)

    def test_gold_drift_rejected_before_scorer(self):
        binding=run.panel.read(next(iter(self.fixture.bindings.values())))
        (self.fixture.root/'gold.private.json').write_text('[]')
        with self.assertRaises(ValueError):summary.score_tasks({}, {},self.fixture.spec,binding,{'aime26':self.fixture.root/'gold.private.json'})


class AdditionalTests(unittest.TestCase):
    def test_snapshot_config_alias_frozen_canonical_files_score_without_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture=Fixture(root)
            binding_path=next(iter(fixture.bindings.values()));binding=run.panel.read(binding_path)
            model=Path(binding['model']);blobs=root/'model_blobs';blobs.mkdir()
            canonical={name:str(blobs/name) for name in ('config.json','generation_config.json')}
            for name in canonical:shutil.copyfile(model/name,canonical[name])
            original_resolve=Path.resolve
            def simulated_snapshot_resolve(path,*args,**kwargs):
                if str(path) in {str(model/name) for name in canonical}:
                    return Path(canonical[path.name])
                return original_resolve(path,*args,**kwargs)
            out=root/'snapshot_freeze'
            with patch.object(Path,'resolve',simulated_snapshot_resolve):
                run.freeze_private(root/'public_spec.json',root/'catalog.private.json',binding['config'],model,binding['deploy'],'toy_host','PUBLIC_TOY_UUID',out)
            frozen_path=out/'binding.private.json';frozen=run.panel.read(frozen_path)
            self.assertEqual(frozen['model_config_files'],canonical)
            self.assertEqual(frozen['model'],str(model))
            self.assertTrue(all(path in frozen['files'] for path in canonical.values()))
            shutil.rmtree(model)
            mapping={path:path for path in frozen['files']}
            mapping[str(Path(frozen['deploy'])/'DEPLOY_SHA')]=str(Path(frozen['deploy'])/'DEPLOY_SHA')
            pin=run.panel.digest(frozen_path)
            artifacts=run.FrozenArtifacts(frozen_path,mapping,pin)
            run.read_frozen(frozen_path,artifacts)
            for entry in fixture.entries:
                entry.update(binding=str(frozen_path),binding_sha256=pin,scorer_artifacts=mapping,
                    worker_sha256={name:run.panel.digest(Path(entry['run_dir'])/name)
                    for name in ('terminal.json','records.jsonl','completions.private.jsonl')})
            dump(fixture.family,{'workers':fixture.entries})
            self.assertEqual(len(summary.load_family(fixture.family)[1]),960)
            for replacement in ({},dict(canonical,unknown=str(blobs/'config.json')),
                                dict(canonical,**{'config.json':str(model/'config.json')})):
                bad=deepcopy(frozen);bad['model_config_files']=replacement;dump(frozen_path,bad)
                with self.assertRaises((ValueError,FileNotFoundError)):
                    run.read_frozen(frozen_path,run.FrozenArtifacts(frozen_path,mapping,run.panel.digest(frozen_path)))
            dump(frozen_path,frozen)
            (blobs/'config.json').write_text('changed')
            with self.assertRaises(ValueError):run.FrozenArtifacts(frozen_path,mapping,pin)

    def test_frozen_main_implicit_q128_defaults_and_q64_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture=Fixture(root)
            binding=run.panel.read(next(iter(fixture.bindings.values())))
            original=run.panel.read(binding['config'])
            original.pop('q_block');original.pop('q_regroup');original.pop('q_carry64')
            original['fingerprint']=fingerprint(original);path=root/'implicit_config.json';dump(path,original)
            run.freeze_private(root/'public_spec.json',root/'catalog.private.json',path,binding['model'],binding['deploy'],'toy_host','PUBLIC_TOY_UUID',root/'implicit')
            for key,value in (('q_block',64),('min_route_keys',2048)):
                bad=deepcopy(original);bad[key]=value;bad['fingerprint']=fingerprint(bad);dump(path,bad)
                with self.assertRaises(ValueError):run.freeze_private(root/'public_spec.json',root/'catalog.private.json',path,binding['model'],binding['deploy'],'toy_host','PUBLIC_TOY_UUID',root/'wrong')

    def test_scorer_mirror_validates_complete_original_binding_without_old_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture=Fixture(root)
            binding_path=next(iter(fixture.bindings.values()));binding=run.panel.read(binding_path)
            snapshot=root/'binding.original.json';shutil.copyfile(binding_path,snapshot)
            mirror=root/'mirror';mirror.mkdir();mapping={}
            paths=list(binding['files'])+[str(Path(binding['deploy'])/'DEPLOY_SHA')]
            for index,original in enumerate(paths):
                target=mirror/str(index);shutil.copyfile(original,target);mapping[original]=str(target)
            pin=run.panel.digest(snapshot)
            for entry in fixture.entries:
                entry.update(binding=str(snapshot),binding_sha256=pin,scorer_artifacts=mapping)
                entry['worker_sha256']={name:run.panel.digest(Path(entry['run_dir'])/name)
                    for name in ('terminal.json','records.jsonl','completions.private.jsonl')}
            dump(fixture.family,{'workers':fixture.entries})
            shutil.rmtree(root/'toy_host')
            cells,completions,_,scoring=summary.load_family(fixture.family)
            self.assertEqual(len(completions),960);self.assertEqual(scoring['host'],'toy_host')
            self.assertEqual(run.panel.read(snapshot),binding)
            original_pin=fixture.entries[0]['worker_sha256']['records.jsonl']
            fixture.entries[0]['worker_sha256']['records.jsonl']='b'*64
            dump(fixture.family,{'workers':fixture.entries})
            with self.assertRaises(ValueError):summary.load_family(fixture.family)
            fixture.entries[0]['worker_sha256']['records.jsonl']=original_pin
            for change in ('missing','unknown','alias','binding_pin','byte_drift','deploy_sha'):
                altered=deepcopy(mapping);bad_pin=pin
                if change=='missing':altered.pop(paths[0])
                elif change=='unknown':altered['unknown']=altered[paths[0]]
                elif change=='alias':altered[paths[0]]=altered[paths[1]]
                elif change=='binding_pin':bad_pin='b'*64
                else:
                    target=Path(altered[paths[-1] if change=='deploy_sha' else paths[0]])
                    saved=target.read_bytes();target.write_text('changed')
                with self.assertRaises(ValueError):
                    artifacts=run.FrozenArtifacts(snapshot,altered,bad_pin);run.read_frozen(snapshot,artifacts)
                if change in ('byte_drift','deploy_sha'):target.write_bytes(saved)

    def test_rebind_from_source_mirror_preserves_method_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);mirror=root/'mirror';new=root/'destination';old=root/'unavailable_original'
            for base in (mirror,new):
                (base/'scripts').mkdir(parents=True);(base/'scripts/toy.py').write_text('public source')
            config=dict(public_setting=17,source_hashes={str(old/'scripts/toy.py'):run.panel.digest(mirror/'scripts/toy.py')})
            config['fingerprint']=fingerprint(config);saved=deepcopy(config)
            rebound=run.rebind_config_from_mirror(config,old,mirror,new)
            self.assertEqual(config,saved);self.assertEqual(run._method_fields(rebound),run._method_fields(config))
            self.assertEqual(rebound['source_hashes'],{str((new/'scripts/toy.py').resolve()):next(iter(config['source_hashes'].values()))})
            for base in (mirror,new):
                target=base/'scripts/toy.py';target.write_text('changed')
                with self.assertRaises(ValueError):run.rebind_config_from_mirror(config,old,mirror,new)
                target.write_text('public source')
            config['fingerprint']='b'*64
            with self.assertRaises(ValueError):run.rebind_config_from_mirror(config,old,mirror,new)

    def test_unparsed_valid_response_is_scored_failure_not_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture=Fixture(Path(folder),qualification=True)
            cells,completions,spec,binding=summary.load_family(fixture.family,True)
            for completion in completions.values():
                completion['completion']='Public toy unfinished thought without a final channel.'
            module=types.ModuleType('experiments.diffusion_gemma_aime26_modes.protocol')
            module.final_response=lambda raw,thinking:''
            module.numeric_score=lambda text,expected:dict(correct=False,extracted=None)
            with patch.dict('sys.modules',{'experiments.diffusion_gemma_aime26_modes.protocol':module}), patch.object(summary,'scorer_selftest',return_value={'aime26':True}):
                selftests=summary.score_tasks(cells,completions,spec,binding,{'aime26':fixture.root/'gold.private.json'})
            result=summary.summarize(cells,spec,40,True,selftests)
            self.assertTrue(result['qualification_ready'])
            self.assertTrue(all(row['strict_correct']==0 for row in result['arms']))
            self.assertTrue(all(not row['parsed'] and not row['final_channel_present'] for rows in cells.values() for row in rows.values()))
            run.validate_qualification(result,spec,binding,'public_toy_torch','public_toy_vllm')
            exported=json.dumps(result)
            for private in ('Public toy unfinished',str(fixture.root),'aime26/1','completion','prompt_tokens','gold_sha256'):
                self.assertNotIn(private,exported)

    def test_qualification_proof_binds_generation_host_gpu_and_software(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture=Fixture(Path(folder),qualification=True)
            cells,_,spec,binding=summary.load_family(fixture.family,True)
            for rows in cells.values():
                for row in rows.values():row.update(strict_correct=False,task_correct=False,parsed=False,final_channel_present=False)
            proof=summary.summarize(cells,spec,40,True,{'aime26':True})
            run.validate_qualification(proof,spec,binding,'public_toy_torch','public_toy_vllm')
            for field,value in (('host','other_host'),('gpu_uuid','other_gpu'),('torch','other_torch'),('vllm','other_vllm'),('deploy_commit','b'*40),('cpu_threads',2),('method_fingerprint','b'*64),('adapter_sha256','c'*64)):
                bad=deepcopy(proof);bad['qualified_environments'][0][field]=value
                with self.assertRaises(ValueError):run.validate_qualification(bad,spec,binding,'public_toy_torch','public_toy_vllm')

    def test_every_assigned_host_requires_qualification(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture=Fixture(Path(folder),('toy_a','toy_b'),True)
            cells,completions,spec,_=summary.load_family(fixture.family,True)
            self.assertEqual(len(completions),8)
            self.assertEqual({row['repeat'] for rows in cells.values() for row in rows.values()},{0})
            for rows in cells.values():
                for row in rows.values():row.update(strict_correct=False,task_correct=False,parsed=False,final_channel_present=False)
            proof=summary.summarize(cells,spec,40,True,{'aime26':True})
            self.assertEqual(len(proof['qualified_environments']),2)
            for path in fixture.bindings.values():
                run.validate_qualification(proof,spec,run.panel.read(path),'public_toy_torch','public_toy_vllm')
            proof['qualified_environments'].pop()
            with self.assertRaises(ValueError):run.validate_qualification(proof,spec,run.panel.read(next(iter(fixture.bindings.values()))),'public_toy_torch','public_toy_vllm')

    def test_missing_public_scorer_selftest_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture=Fixture(Path(folder),qualification=True)
            cells,_,spec,_=summary.load_family(fixture.family,True)
            for rows in cells.values():
                for row in rows.values():row.update(strict_correct=True,task_correct=True,parsed=True,final_channel_present=True)
            self.assertFalse(summary.summarize(cells,spec,40,True,{})['qualification_ready'])

    def test_humaneval_public_toy_selftest_delegates_correct_and_wrong(self):
        from scripts import v27_humaneval as he
        spec=run.build_spec('humaneval','public_toy',list(range(8)),['toy_host'])
        with patch.object(he,'sandbox_preflight') as preflight,patch.object(he,'run_test',side_effect=[(True,'passed'),(False,'assertion')]) as test:
            self.assertEqual(summary.scorer_selftest(spec,{}),{'humaneval':True})
            preflight.assert_called_once();self.assertEqual(test.call_count,2)
            self.assertIn('return x',test.call_args_list[0].args[0])
            self.assertIn('return 0',test.call_args_list[1].args[0])

    def test_cross_host_rebinding_preserves_all_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture=Fixture(Path(folder),('toy_a','toy_b'))
            cells,completions,spec,binding=summary.load_family(fixture.family)
            self.assertEqual(len(completions),960)
            self.assertEqual({row['host'] for rows in cells.values() for row in rows.values()},{'toy_a','toy_b'})

    def test_qualification_mode_is_disjoint(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture=Fixture(Path(folder),qualification=True)
            cells,completions,spec,binding=summary.load_family(fixture.family,True)
            self.assertEqual(len(completions),4)
            with self.assertRaises(ValueError):summary.load_family(fixture.family)

    def test_humaneval_uses_official_tests_and_eos_not_required(self):
        from scripts import v27_humaneval as he
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);gold=root/'gold.json';contract=root/'contract.json'
            problem=dict(task_id='HumanEval/0',prompt='def toy(x):\n    """Public toy."""\n',
                         test='def check(f):\n    assert f(2)==2\n',entry_point='toy')
            dump(gold,{'humaneval/0':problem})
            dump(contract,he.contract('d'*64,run.panel.digest(he.__file__)))
            binding=dict(gold_sha256={'humaneval':run.panel.digest(gold)},task_contracts={'humaneval':str(contract)})
            spec=run.build_spec('humaneval','public_toy',list(range(8)),['toy_host'])
            cells={('humaneval',0,0):{'dense':{}}}
            completions={('humaneval',0,0,'dense'):dict(id='humaneval/0',finish_reason='length',
                       completion='<channel|>```python\ndef toy(x):\n    return x\n```')}
            with patch.object(summary,'scorer_selftest',return_value={'humaneval':True}), patch.object(he,'sandbox_preflight') as preflight,patch.object(he,'run_test',return_value=(True,'passed')) as test:
                summary.score_tasks(cells,completions,spec,binding,{'humaneval':gold})
                preflight.assert_called_once();self.assertIn(problem['test'],test.call_args.args[0])
            self.assertTrue(cells[('humaneval',0,0)]['dense']['strict_correct'])



class ExternalRebindTests(unittest.TestCase):
    def setup_case(self, root):
        old = root/'unavailable_origin'
        mirror, new = root/'mirror', root/'destination'
        for base in (mirror, new):
            base.mkdir()
            (base/'toy.py').write_text('public internal source')
        external = root/'unavailable_external'/'public.so'
        external2 = root/'unavailable_external'/'public_identity.json'
        folder = new/'.external_sources'; folder.mkdir()
        target, target2 = folder/'public.so', folder/'public_identity.json'
        target.write_bytes(b'public synthetic artifact')
        target2.write_text('public synthetic identity')
        parent = dict(library=str(external), public_setting=19,
                      source_hashes={str(external):run.panel.digest(target),
                                     str(external2):run.panel.digest(target2)})
        parent['fingerprint'] = fingerprint(parent)
        config = dict(parent_config=parent, public_setting=17,
                      source_hashes={str(old/'toy.py'):run.panel.digest(mirror/'toy.py'),
                                     str(external):run.panel.digest(target)})
        config['fingerprint'] = fingerprint(config)
        return old, mirror, new, config, {str(external):str(target), str(external2):str(target2)}

    def test_explicit_complete_map_preserves_math_hashes_and_input(self):
        with tempfile.TemporaryDirectory() as folder:
            old,mirror,new,config,mapping=self.setup_case(Path(folder));saved=deepcopy(config)
            result=run.rebind_config_from_mirror(config,old,mirror,new,mapping)
            self.assertEqual(config,saved)
            self.assertEqual(run._method_fields(result),run._method_fields(saved))
            run._validate_fingerprints(result)
            self.assertEqual(result['parent_config']['library'],saved['parent_config']['library'])
            for original,target in mapping.items():
                self.assertEqual(result['parent_config']['source_hashes'][target],saved['parent_config']['source_hashes'][original])
            self.assertNotEqual(result['fingerprint'],saved['fingerprint'])

    def test_map_rejects_missing_and_extra_original_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            old,mirror,new,config,mapping=self.setup_case(Path(folder))
            missing=dict(mapping);missing.pop(next(iter(missing)))
            extra=dict(mapping);extra[str(Path(folder)/'unknown.so')]=next(iter(mapping.values()))
            for bad in (missing,extra):
                with self.assertRaises(ValueError):run.rebind_config_from_mirror(config,old,mirror,new,bad)

    def test_map_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            old,mirror,new,config,mapping=self.setup_case(Path(folder))
            Path(next(iter(mapping.values()))).write_bytes(b'different public artifact')
            with self.assertRaises(ValueError):run.rebind_config_from_mirror(config,old,mirror,new,mapping)

    def test_map_rejects_destination_outside_external_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            old,mirror,new,config,mapping=self.setup_case(Path(folder))
            original=next(iter(mapping));outside=new/'public.so'
            outside.write_bytes(Path(mapping[original]).read_bytes());mapping[original]=str(outside)
            with self.assertRaises(ValueError):run.rebind_config_from_mirror(config,old,mirror,new,mapping)

    def test_map_rejects_alias_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            old,mirror,new,config,mapping=self.setup_case(Path(folder))
            first,second=list(mapping)
            mapping[second]=mapping[first]
            with self.assertRaises(ValueError):run.rebind_config_from_mirror(config,old,mirror,new,mapping)

    def test_map_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            old,mirror,new,config,mapping=self.setup_case(Path(folder))
            original=next(iter(mapping));target=Path(mapping[original]);outside=Path(folder)/'outside.so'
            outside.write_bytes(target.read_bytes());target.unlink()
            try:target.symlink_to(outside)
            except OSError as exc:self.skipTest('host cannot create public toy symlink: '+type(exc).__name__)
            with self.assertRaises(ValueError):run.rebind_config_from_mirror(config,old,mirror,new,mapping)

    def test_internal_mirror_and_destination_still_require_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            old,mirror,new,config,mapping=self.setup_case(Path(folder))
            for root in (mirror,new):
                (root/'toy.py').write_text('changed public internal source')
                with self.assertRaises(ValueError):run.rebind_config_from_mirror(config,old,mirror,new,mapping)
                (root/'toy.py').write_text('public internal source')

    def test_default_external_behavior_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            old,mirror,new,config,mapping=self.setup_case(Path(folder))
            with self.assertRaises(OSError):run.rebind_config_from_mirror(config,old,mirror,new)
            for original,target in mapping.items():
                Path(original).parent.mkdir(exist_ok=True)
                shutil.copyfile(target,original)
            result=run.rebind_config_from_mirror(config,old,mirror,new)
            self.assertEqual(result['parent_config']['source_hashes'],config['parent_config']['source_hashes'])

    def test_cli_external_map_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,mirror,new,config,mapping=self.setup_case(root)
            dump(root/'config.json',config);dump(root/'mapping.json',mapping)
            run.main(['rebind-mirror','--config',str(root/'config.json'),'--old-root',str(old),
                      '--mirror-root',str(mirror),'--new-root',str(new),'--out',str(root/'out.json'),
                      '--external-map',str(root/'mapping.json')])
            result=run.panel.read(root/'out.json')
            self.assertEqual(run._method_fields(result),run._method_fields(config))


if __name__=='__main__':
    unittest.main()
