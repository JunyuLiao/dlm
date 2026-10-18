"""CPU-only checks of external profiling gates and API-boundary attribution."""
import copy
import json
import sqlite3
import pytest
from scripts import v31_refresh_external_profile as external


def boundaries(thread=7, offset=0):
    syncs, records = [], []
    cursor = offset + 100_000_000
    for i in range(11):
        prefill = 1_000_000_000 + i * 10_000_000
        decode = 2_000_000_000 + i * 20_000_000
        for end in (cursor, cursor + prefill, cursor + prefill + decode):
            syncs.append(dict(start=end - 10_000, end=end, thread=thread))
        if i:
            records.append(dict(timer_policy='request_and_prefill_boundaries', wall_s=(prefill+decode)/1e9,
                                prefill_s=prefill/1e9, decode_s=decode/1e9))
        cursor += prefill + decode + 100_000_000
    return syncs, records


def arms():
    return [dict(policy='control', label=str(i), warmup=i < 3) for i in range(11)]


def test_unique_api_triplets_require_full_source_timing_contract():
    syncs, records = boundaries()
    result = external.align_boundaries(syncs, records)
    assert result['status'] == 'candidate' and result['candidate_count'] == 1
    assert len(result['windows']) == 11 and result['windows'][3]['request_index'] == 3
    bad = copy.deepcopy(records);bad[5]['prefill_s'] += .02
    assert external.align_boundaries(syncs, bad)['status'] == 'unknown'
    bad[5] = records[5];bad[0]['timer_policy'] = 'synchronized_each_engine_step'
    assert external.align_boundaries(syncs, bad)['status'] == 'unknown'
    assert external.align_boundaries(syncs, records[:-1])['status'] == 'unknown'


def test_ambiguous_alignment_and_extra_internal_sync_are_not_silently_selected():
    syncs, records = boundaries()
    more, _ = boundaries(thread=8, offset=100_000_000_000)
    result = external.align_boundaries(syncs + more, records)
    assert result['status'] == 'unknown' and result['candidate_count'] == 2
    extra = dict(start=syncs[16]['end']+10, end=syncs[16]['end']+20, thread=7)
    assert external.align_boundaries(syncs + [extra], records)['status'] == 'unknown'


def test_activity_windows_exclude_startup_keep_unknown_and_union_overlap():
    syncs, records = boundaries();alignment=external.align_boundaries(syncs, records)
    w=alignment['windows'][3];start=w['start_ns'];end=w['prefill_end_ns']
    data=[dict(start=0,end=100,device=0,name='opaque',kind='kernel'),
          dict(start=start+10,end=start+110,device=0,name='FlashAttentionForward',kind='kernel'),
          dict(start=start+60,end=start+160,device=0,name='opaque',kind='kernel'),
          dict(start=end+10,end=end+210,device=0,name='fused_moe',kind='kernel')]
    report=external.activity_summary(data,alignment,arms())
    assert report['segmentation_status']=='qualified'
    assert report['whole_process_categories']['unknown']['activities']==2
    request=report['request_windows'][3]
    assert not request['warmup']
    assert request['phases']['prefill']['devices']['0']['gpu_busy_union_ms']==pytest.approx(.00015)
    assert request['phases']['decode']['categories']['name_moe_candidate']['activities']==1
    assert sum(v['activities'] for v in request['phases']['prefill']['categories'].values())==2
    crossing=[dict(start=end-1,end=end+1,device=0,name='opaque',kind='kernel')]
    bad=external.activity_summary(crossing,alignment,arms())
    assert bad['segmentation_status']=='unknown' and bad['request_windows']==[]
    assert bad['gpu_activities_crossing_boundaries']==1


def test_sqlite_reader_uses_runtime_threads_and_readonly_file(tmp_path):
    path=tmp_path/'trace.sqlite';db=sqlite3.connect(path)
    db.executescript('CREATE TABLE StringIds(id INTEGER,value TEXT); CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,nameId INTEGER); CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,demangledName INTEGER);')
    db.executemany('INSERT INTO StringIds VALUES(?,?)',[(1,'cudaDeviceSynchronize'),(2,'fused_moe_kernel')])
    syncs,records=boundaries();db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)',[(r['start'],r['end'],r['thread'],1) for r in syncs])
    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?)',(syncs[9]['end']+1,syncs[9]['end']+100,0,2));db.commit();db.close()
    before=path.read_bytes();report=external.sqlite_summary(path,records,arms())
    assert report['segmentation_status']=='qualified' and path.read_bytes()==before
    assert report['whole_process_categories']['name_moe_candidate']['activities']==1


def test_reserve_charges_prior_failures_orphans_and_campaign_cap():
    reservations=[dict(phase='headroom',reserved_s=1800.,elapsed_s=1693.),dict(phase='qualification',reserved_s=900.,elapsed_s=472.),dict(phase='factorial',reserved_s=3900.,elapsed_s=3900.)]
    assert external.reserve_available(reservations)==600.
    reservations.append(dict(phase='reserve',reserved_s=600.,elapsed_s=100.,status='failed'))
    assert external.reserve_available(reservations)==500.
    reservations[-1]['elapsed_s']=None
    assert external.reserve_available(reservations)==0.
    assert external.reserve_available([dict(phase='factorial',reserved_s=7100.,elapsed_s=7100.)])==100.


def test_parity_requires_outputs_bindings_schedule_and_no_audit_hooks():
    schedule=[dict(policy='control',label='implicit',warmup=True)]+[dict(policy='control',label=str(i),warmup=i<2)for i in range(10)]
    rows=[]
    for i in range(10):
        row={k:1 for k in ('index','panel_seed','rng_seed','budget','prompt_tokens','max_model_len','chunk','block_size','denoise_forwards','canvases','output_tokens','sampler_calls','engine_steps','prefill_steps')}
        row.update(dataset='lb',manifest_sha256='m',prompt_sha256='p',frozen_source_identity_sha256='s',cudagraph_mode='PIECEWISE',resolved_cudagraph_mode='PIECEWISE',fix_51994=True,fa4_local_fix=True,output_hash='h',finish_reason='stop',cuda_graph_captures=0,timer_policy='request_and_prefill_boundaries',value_clean_timing=True,value_audit=False,receipts=dict(adapter=dict(refresh_label=str(i),refresh_policy='control',refresh_request_index=i+1)))
        rows.append(row)
    reference=copy.deepcopy(rows)
    assert external.parity_check(rows,reference,schedule)['status']=='matched'
    rows[2]['output_hash']='different'
    assert external.parity_check(rows,reference,schedule)['measured_matches']==7
    rows[2]['output_hash']='h';rows[3]['receipts']['adapter']['headroom']={}
    assert external.parity_check(rows,reference,schedule)['status']=='mismatched'
    rows[3]['receipts']['adapter'].pop('headroom');rows[4]['cuda_graph_captures']=1
    assert external.parity_check(rows,reference,schedule)['status']=='mismatched'
    assert external.parity_check(rows[:-1],reference,schedule)['status']=='incomplete'


def parity_fixture():
    schedule=[dict(policy='control',label='implicit',warmup=True)]+[
        dict(policy='control',label=str(i),warmup=i<2) for i in range(10)]
    rows=[]
    for i in range(10):
        row={key:1 for key in ('index','panel_seed','rng_seed','budget','prompt_tokens','max_model_len',
            'chunk','block_size','denoise_forwards','canvases','output_tokens','sampler_calls','engine_steps','prefill_steps')}
        row.update(dataset='lb',manifest_sha256='m',prompt_sha256='p',frozen_source_identity_sha256='source',
            cudagraph_mode='PIECEWISE',resolved_cudagraph_mode='PIECEWISE',fix_51994=True,fa4_local_fix=True,
            adapter_sha256='adapter',method_fingerprint='method',mage_k=8192,mage_reselect_trigger=.15,
            output_hash='output',finish_reason='stop',cuda_graph_captures=0,
            timer_policy='request_and_prefill_boundaries',value_clean_timing=True,value_audit=False,
            receipts=dict(adapter=dict(refresh_label=str(i),refresh_policy='control',refresh_request_index=i+1,
                global_eligible_tiles=1000,global_kept_tiles=200,mage_selections=5,mage_reselections=5)))
        rows.append(row)
    return rows,copy.deepcopy(rows),schedule


@pytest.mark.parametrize('counter',['global_kept_tiles','mage_reselections'])
def test_matching_output_cannot_hide_routing_counter_change(counter):
    rows,reference,schedule=parity_fixture()
    assert external.parity_check(rows,reference,schedule)['status']=='matched'
    rows[2]['receipts']['adapter'][counter]+=1
    result=external.parity_check(rows,reference,schedule)
    assert result['status']=='mismatched' and result['measured_matches']==7


@pytest.mark.parametrize('field',['adapter_sha256','method_fingerprint','mage_k','mage_reselect_trigger'])
def test_matching_output_cannot_hide_method_identity_change(field):
    rows,reference,schedule=parity_fixture()
    rows[2][field]='changed'
    assert external.parity_check(rows,reference,schedule)['status']=='mismatched'


def private_binding_fixture():
    row={key:1 for key in external.official.COMPARE_FIELDS}
    row.update(dataset='lb',index=0,panel_seed=42,rng_seed=123,repeat=0,finish_reason='stop')
    return [dict(row) for _ in range(10)],[dict(row,id='opaque') for _ in range(10)]


def test_private_binding_requires_explicit_official_verification(monkeypatch):
    rows,private=private_binding_fixture()
    calls=[]
    def unverified(record,pools):
        calls.append(record)
        return {},{},False
    monkeypatch.setattr(external.official,'bind',unverified)
    with pytest.raises(RuntimeError):
        external.check_private_bind(rows,private,object())
    assert calls


@pytest.mark.parametrize('field',['prompt_sha256','dataset','index','panel_seed'])
def test_private_binding_checks_public_private_identity(monkeypatch,field):
    rows,private=private_binding_fixture()
    monkeypatch.setattr(external.official,'bind',lambda record,pools:({},{},True))
    external.check_private_bind(rows,private,object())
    private[3][field]='different'
    with pytest.raises(RuntimeError):
        external.check_private_bind(rows,private,object())


def test_worker_command_reuses_original_execution_with_isolated_caches(tmp_path,monkeypatch):
    original=tmp_path/'original';output=tmp_path/'external'
    monkeypatch.setattr(external,'ORIGINAL',original)
    monkeypatch.setattr(external.base,'sha',lambda path:'bound-hash')
    seen=[]
    cache_keys=('VLLM_CACHE_ROOT','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH',
        'FA4_LOCAL_FIX_DIR','TMPDIR','FLASHINFER_WORKSPACE_BASE')
    def environment(root,suite,arm):
        seen.append((root,suite,arm))
        return {key:str(root/'private/cache'/key) for key in cache_keys}|{
            'REFRESH_AUDIT':'1','HEADROOM_PROFILE':'1','HEADROOM_STALE':'bad','VALUE_AUDIT':'1'}
    monkeypatch.setattr(external.base,'env_for',environment)
    command,env=external.worker_command(output,{'suites':{},'source_commit':'original-source'},600.)
    assert seen==[(output,'longbench_v2','current_v31_control')]
    assert str(original/'private/execution/scripts/v31_refresh_diagnostic_worker.py') in command
    assert str(original/'private/pools/longbench_v2') in command
    assert env['PYTHONPATH']==f'{original}/private/execution/src:{original}/private/execution'
    assert env['REFRESH_BASE_WORKER']==str(original/'private/execution/scripts/v31_vllm_paired_bench.py')
    assert env['REFRESH_SCHEDULE']==str(output/'private/arms.json')
    assert env['PYTHONDONTWRITEBYTECODE']=='1'
    for key in cache_keys:
        assert env[key].startswith(str(output/'private/cache'))
    for key in ('REFRESH_AUDIT','HEADROOM_PROFILE','VALUE_AUDIT'):
        assert env[key]=='0'
    for key in ('VALUE_CLEAN_TIMING','VALUE_BOUNDARY_TIMING'):
        assert env[key]=='1'
    assert 'HEADROOM_STALE' not in env


@pytest.mark.parametrize('completed',[9,10])
def test_cpu_validation_requires_all_native_requests_even_if_profiler_exits_zero(tmp_path,monkeypatch,completed):
    original=tmp_path/'original';output=tmp_path/'external'
    rows,reference,schedule=parity_fixture()
    plan=dict(engine=2,phase='headroom',family='shared',suite='longbench_v2',pass_name='clean',
        request_count=10,measured_count=8,folder='clean_engine')
    def write(path,value,jsonl=False):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(x)+'\n' for x in value) if jsonl else json.dumps(value))
    write(original/'protocol.json',dict(plans=[plan],suites={'longbench_v2':{}}))
    write(original/'clean_engine/records.jsonl',reference,True)
    write(output/'protocol.json',{'cache_copy_manifest':{}})
    write(output/'records.jsonl',rows[:completed],True)
    write(output/'private/generation.private.jsonl',rows[:completed],True)
    write(output/'private/cells.json',[{}]*10)
    write(output/'private/arms.json',schedule)
    monkeypatch.setattr(external,'ORIGINAL',original)
    monkeypatch.setattr(external,'inventory',lambda path:{})
    monkeypatch.setattr(external.study,'validate_row',lambda *args:[])
    monkeypatch.setattr(external,'check_private_bind',lambda *args:None)
    monkeypatch.setattr(external.official,'Pools',lambda *args,**kwargs:object())
    result=external.validate_output(output)
    assert result['records']==completed and result['private_records']==completed
    assert result['status']==('complete' if completed==10 else 'failed_or_partial')
    assert result['parity_status']==('matched' if completed==10 else 'incomplete')
