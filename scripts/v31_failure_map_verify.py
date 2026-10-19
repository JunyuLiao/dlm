"""Independent artifact, scoring, coverage and physical-work review (CPU only)."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from scripts import v31_failure_map_study as s


def verify(study):
    p=s.verify(study);state=s.base.read(study/'state.json')
    dest=study/'verification/attempt001'
    if dest.exists():raise RuntimeError('new verification attempt required')
    phases={x:s.load_phase(study,x) for x in state['completed_phases']}
    pools=s.official.Pools(study/f'private/pools/{s.SUITE}',fields=('thinking',))
    final,_=s.official.load_final_response()
    gold=s.base.read(study/f'private/pools/{s.SUITE}'/(p['suites'][s.SUITE]['dataset']+'_gold.json'))
    checked=0;rescored=0;nonfinite_unmeasured=0;jit=[];work=defaultdict(lambda:defaultdict(int));phase_work=defaultdict(lambda:defaultdict(int));per_item=[]
    for phase,rows in phases.items():
        markers=s.base.read(study/phase/'phase_complete.json')['engines']
        for name in markers:
            leaf=(study/name).parent
            records=s.prior.complete_jsonl(leaf/'records.jsonl')
            raw=s.prior.complete_jsonl(leaf/'private/generation.private.jsonl')
            arms=s.base.read(leaf/'private/arms.json')[1:]
            scored=s.base.read(leaf/'scored.json')
            assert len(records)==len(raw)==len(arms)
            printed=0;warnings=0
            for line in (leaf/'private/worker.log').read_text(errors='replace').splitlines():
                if 'JIT compilation during inference:' in line and printed>=3:warnings+=1
                if line.startswith('{'):
                    try:message=json.loads(line)
                    except ValueError:continue
                    if 'arm' in message and 'output_tokens' in message:printed+=1
            assert printed==len(records)
            if warnings:jit.append(dict(leaf=str(leaf.relative_to(study)) if leaf.is_relative_to(study) else str(leaf.relative_to(study.parent)),
                warnings_after_explicit_warmup=warnings,pass_name=s.base.read(leaf/'config.json')['pass_name']))
            j=0
            for r,g,a in zip(records,raw,arms):
                assert r['frozen_source_identity_sha256']==s.base.sha(study/'receipts/source_fingerprints.json')
                assert r['fix_51994'] and r['fa4_local_fix'] and r['cuda_graph_captures']==0
                assert r['denoise_forwards']+r['canvases']==r['sampler_calls']
                assert r['prefill_steps']+r['sampler_calls']==r['engine_steps']
                meta,_,bound=s.official.bind(g,pools);assert bound
                assert all(r[f]==g[f] for f in s.official.COMPARE_FIELDS)
                checked+=1
                if a['warmup']:continue
                score=float(s.official.lb_extract_answer(final(g['completion'],bool(meta['thinking'])))==gold[g['id']])
                out=scored[j];j+=1
                assert score==out['score'] and r['output_hash']==out['output_hash']
                assert r['index']==out['index'] and r['panel_seed']==out['panel_seed']
                rescored+=1
                per_item.append({k:v for k,v in out.items() if k!='native_receipt'})
                if not r['value_audit']:continue
                ad=r['receipts']['adapter'];group=f"{phase}/{out['family']}/k{out['k']}/{out['label']}"
                fields=('global_eligible_tiles','global_kept_tiles','native_local_eligible_rectangles','native_local_skipped_tiles','overall_eligible_rectangles','overall_skipped_rectangles')
                assert all(f in ad for f in fields)
                assert ad['native_local_skipped_tiles']==0
                assert ad['overall_eligible_rectangles']==ad['global_eligible_tiles']+ad['native_local_eligible_rectangles']
                assert ad['overall_skipped_rectangles']==ad['global_eligible_tiles']-ad['global_kept_tiles']
                assert not ad.get('order_errors')
                if ad.get('value_nonfinite_attention_calls') is None:nonfinite_unmeasured+=1
                else:assert ad['value_nonfinite_attention_calls']==0
                if out['family']=='shared':
                    assert ad['failure_map_budget_errors']==0 and ad['failure_map_budget_units']>0
                for f in fields:work[group][f]+=ad[f]
                for label,item in ad['value_phase_tiles'].items():
                    for f in ('calls','eligible_tiles','kept_tiles','skipped_tiles'):
                        phase_work[group+'/'+label][f]+=item[f]
        expected=sum(s.base.read((study/name).parent/'config.json')['measured_count'] for name in markers)
        assert len(rows)==expected
    ledger=s.base.read(study.parent/'gpu_budget.json');charges=defaultdict(float)
    for r in ledger['reservations']:
        assert r['elapsed_s'] is not None and r['elapsed_s']>=0
        charges[r['phase']]+=r['elapsed_s']
    assert sum(charges.values())<=s.CAP
    assert all(t<=s.BUDGET[phase] for phase,t in charges.items())
    assert abs(state['gpu_reservation_s']-sum(charges.values()))<1e-6
    if 'screen' in phases:
        assert len(phases['screen'])==320
        for k in s.KS:
            for label in ('control_before','control_after'):
                selected=[r for r in phases['screen'] if r['k']==k and r['label']==label]
                assert len(selected)==32 and len({(r['index'],r['panel_seed']) for r in selected})==32
        for label in (*s.REFS,'fixed16'):
            selected=[r for r in phases['screen'] if r['label']==label]
            assert len(selected)==32 and len({(r['index'],r['panel_seed']) for r in selected})==32
    repair_failure_visibility=[]
    repair_extra_work=[]
    if 'repair' in phases:
        original={r['index']:r['failure_type'] for r in s.base.read(study/'confirmation/confirmed_failures.json')['repair_cases']}
        grouped=defaultdict(list)
        for r in phases['repair']:grouped[(r['index'],r['panel_seed'],r['engine_repeat'])].append(r)
        for (index,seed,repeat),rows in sorted(grouped.items()):
            controls=[r for r in rows if r['label'] in ('control_before','control_after')]
            refs=[r for r in phases['confirmation'] if (r['index'],r['panel_seed'],r['engine_repeat'])==(index,seed,repeat) and r['label'] in s.REFS]
            visible=s.case_type(controls,refs)==original[index]
            repair_failure_visibility.append(dict(index=index,seed=seed,repeat=repeat,original_failure_type=original[index],
                visible_against_clean_reference=visible,
                caveat='cross-pass descriptive check only; instrumented controls do not establish a matched dense causal comparison'))
        for r in phases['repair']:
            d=r['native_receipt']['refresh_diagnostic']
            if not d['events']:continue
            dense_rectangles=0;tail_rectangles=0;tail_products=0
            for event in d['events']:
                batch,n,h,dimension=event['dense_output']['shape']
                assert batch==1 and n==event['query_rows']
                nk=event['prefix']+n;tail=nk-64*(event['prefix']//64)
                dense_rectangles+=h*((n+127)//128)*((nk+63)//64)
                tail_rectangles+=h*((n+127)//128)*((tail+63)//64)
                tail_products+=h*n*tail*dimension
            extra=d['extra_diagnostic_work']
            assert extra['observer_qk_rectangles']==extra['observer_pv_rectangles']==dense_rectangles
            changed=sum(event['changed_prefix_blocks'] for event in d['events'])
            partner_label={'sparse_fresh':'sparse_hold','dense_fresh':'dense_hold'}.get(r['label'])
            partner=next((x for x in grouped[(r['index'],r['panel_seed'],r['engine_repeat'])] if x['label']==partner_label),None)
            noop_drift=(changed==0 and partner is not None and any(r[f]!=partner[f] for f in
                        ('output_hash','denoise_forwards','canvases','score')))
            repair_extra_work.append(dict(index=r['index'],seed=r['panel_seed'],repeat=r['engine_repeat'],label=r['label'],
                dense_qk_rectangles=dense_rectangles,dense_pv_rectangles=dense_rectangles,
                additional_fp32_tail_qk_rectangles=tail_rectangles,additional_fp32_tail_scalar_products=tail_products,
                changed_prefix_blocks=changed,zero_change_fresh_support_outcome_drift=noop_drift,
                first_post_event_call=d['first_post_event_call'],
                other_unpriced_diagnostic_work='normalization, selection, list construction, finite/norm reductions, tensor hashing and state capture',
                note='Tail product derived from the frozen _mage_select_fa4 implementation and recorded output shape; audit-only, no speed claim'))
    pass_finished=[]
    for phase,rows in phases.items():
        groups=defaultdict(list)
        for r in rows:groups[(r['family'],r['k'],r['label'])].append(r)
        for (family,k,label),group in groups.items():
            pass_finished.append(dict(phase=phase,family=family,k=k,label=label,records=len(group),
                correct_and_finished=sum(r['score']==1 and r['finish_reason'] not in s.official.CAPPED for r in group),
                correct_but_capped=sum(r['score']==1 and r['finish_reason'] in s.official.CAPPED for r in group)))
    paired_references=[]
    for phase in ('screen','confirmation'):
        if phase not in phases:continue
        for reference in s.REFS:
            # Reuse the frozen estimator with a temporary reference alias. The
            # original rows and published arm labels remain untouched.
            comparison=[r for r in phases[phase] if r['family']=='shared']
            comparison += [dict(r,label='dense_full_fix51994') for r in phases[phase] if r['label']==reference]
            paired_references.extend(dict(phase=phase,reference_label=reference,**item)
                                     for item in s.paired_intervals(comparison))
    s.base.write(dest/'per_item.json',per_item,True)
    s.base.write(dest/'physical_work.json',dict(aggregate=dict(work),phases=dict(phase_work),
        denominator='audit denoising GLOBAL full support plus LOCAL window-intersecting rectangles; not clean work'),True)
    result=dict(status='verified',source_commit=p['source_commit'],protocol_sha256=s.base.sha(study/'protocol.json'),
        records_checked=checked,official_rescoring_agreements=rescored,nonfinite_metric_unmeasured_records=nonfinite_unmeasured,
        gpu_seconds_by_phase=dict(charges),total_gpu_seconds=sum(charges.values()),
        measured_jit_warnings=jit,clean_timing_jit_qualified=not any(x['pass_name']=='clean' for x in jit),
        repair_failure_visibility=repair_failure_visibility,
        repair_extra_work=repair_extra_work,
        pass_and_finished=pass_finished,
        paired_reference_intervals=paired_references,
        old_campaigns_remain_paused=True,analysis_script_sha256=s.base.sha(Path(__file__)))
    s.base.write(dest/'verified_complete.json',result,True)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True)
    verify(ap.parse_args().study)
