"""Descriptive CPU readout and legacy checks; does not change frozen R gates."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

STUDY=Path(__file__).resolve().parents[1];REPO=STUDY.parents[1]
R=STUDY/'track_r/attempt001';OLD=REPO/'results/denoising_e2e_research_20261024/attempt003'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    start=time.time();out=Path(__file__).resolve().parent;inputs={}
    def read(path):
        inputs[str(path.relative_to(REPO))]=sha(path)
        return json.loads(path.read_text())
    assert read(R/'final_complete.json')['passed']
    result=read(R/'analysis001/summary.json');protocol=read(R/'protocol.json')
    rows={(x['arm'],x['id'],x['seed']):x for x in result['rows']}
    old_cells=read(OLD/'e2e_protocol.json')['panel']['short'];legacy=[]
    for arm in protocol['arms']:
        for i,cell in enumerate(old_cells):
            new=rows[(arm,cell['id'],cell['seed'])];prior=read(OLD/f'runs/panel/short/{arm}/clean{i:03d}.json')
            old_audit=read(OLD/f'runs/panel/short/{arm}/audit{i:03d}.json');new_audit=read(R/new['audit_path'])
            checks={k:new[k]==prior[k] for k in ['output_sha256','N','C','T']}
            old_selector=old_audit['selector'];new_selector=new_audit['selector']
            if old_selector is not None:
                for k in ['physical_calls','selector_refresh_calls','held_mask_calls','carried_mask_calls','target_eligible_tiles','target_kept_tiles','target_skipped_tiles']:
                    checks[k]=old_selector[k]==new_selector[k]
            legacy.append(dict(id=cell['id'],seed=cell['seed'],arm=arm,checks=checks,passed=all(checks.values())))
    assert len(legacy)==200
    prefix=read(STUDY/'content_prefix_study001/panel001/summary.json')
    prefix_rows={(x['arm'],x['id'],x['seed']):x for x in prefix['rows']}
    groups=defaultdict(list)
    for x in result['rows']:
        groups[(x['benchmark'],'all',x['arm'])].append(x)
        groups[(x['benchmark'],x['stratum'],x['arm'])].append(x)
    properties=[]
    names=['duplicate_line_fraction','repeated_8gram_fraction','revision_vocabulary_count','repeated_function_definitions','final_channel_character','first_closed_box_in_final_character','end_markers']
    for (task,stratum,arm),values in sorted(groups.items()):
        measures={}
        for name in names:
            available=[x['text_metrics'][name] for x in values if x['text_metrics'][name] is not None]
            measures[name]=dict(mean=statistics.mean(available) if available else None,cells=len(available),missing=len(values)-len(available))
        parsed=[prefix_rows[(arm,x['id'],x['seed'])] for x in values]
        first=[x['first_officially_extractable_checkpoint_character'] for x in parsed if x['first_officially_extractable_checkpoint_character'] is not None]
        frac=[x['checkpoint_character_fraction'] for x in parsed if x['checkpoint_character_fraction'] is not None]
        properties.append(dict(benchmark=task,stratum=stratum,arm=arm,cells=len(values),questions=len({x['id'] for x in values}),
            measures=measures,finish_reasons=dict(Counter(x['finish_reason'] for x in values)),
            prefix=dict(available=len(first),missing=len(values)-len(first),mean_first_character=statistics.mean(first) if first else None,
                mean_character_fraction=statistics.mean(frac) if frac else None),
            endpoint_note='Literal text properties and retrospective extractability; not semantic redundancy, native stopping, correct prefix, or a causal length explanation.'))
    transitions=[];binding=[]
    for transition in result['correctness_transitions']:
        candidate=rows[('response_guard_qdirection',transition['id'],transition['seed'])]
        reference=rows[(transition['reference'],transition['id'],transition['seed'])]
        detail=[]
        for x in [candidate,reference]:
            public=R/x['clean_path'];private=R/'private/generations'/Path(x['clean_path']).relative_to('runs')
            raw=json.loads(private.read_text())
            assert (raw['id'],raw['seed'],raw['output_sha256'])==(x['id'],x['seed'],x['output_sha256'])
            assert sha(private)==x['private_generation_sha256']
            binding.append(dict(id=x['id'],seed=x['seed'],arm=x['arm'],private_generation_sha256=sha(private),clean_sha256=sha(public)))
            # Reuse pre-generation rubric and official scorer outcomes; no new semantic labels.
            detail.append(dict(arm=x['arm'],correct=x['correct'],quality_detail=x['quality_detail'],no_final_response=x['no_final_response'],
                failure_class=None if x['correct'] else 'official_extraction_failure' if not x['quality_detail']['parsed'] else 'official_task_test_or_numeric_failure',
                N=x['N'],C=x['C'],T=x['T'],output_cap=x['output_cap'],text_metrics=x['text_metrics'],prefix=prefix_rows[(x['arm'],x['id'],x['seed'])]))
        transitions.append(dict(id=transition['id'],seed=transition['seed'],benchmark=transition['benchmark'],reference=transition['reference'],
            direction='reference_correct_to_response_wrong' if reference['correct'] else 'reference_wrong_to_response_correct',details=detail))
    summary=dict(passed=True,legacy_full_stratum_checks=legacy,legacy_checks_passed=sum(x['passed'] for x in legacy),legacy_checks_total=len(legacy),
        legacy_note='Supplemental200-cell legacy comparison, not a replacement for the pre-generation24-cell engineering gate. Any drift is retained, not rerun.',
        text_properties=properties,all_correctness_transitions=transitions,private_transition_input_bindings=binding,
        primary_decisions_unchanged=result['decisions'],new_model_forwards=0,allocated_CPU_seconds=time.time()-start,
        causal_content_mechanism_established=False,retrospective_prefix_is_native_stop=False,
        interpretation='The old AIME lead remains in the legacy stratum; it does not generalize to new-to-Response questions. Literal text statistics cannot validate distracting-information or redundant-reasoning explanations.')
    for name,value in [('summary.json',summary),('input_hashes.json',inputs)]:
        with (out/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True)
    with (out/'final_complete.json').open('x') as f:json.dump(dict(passed=True,summary_sha256=sha(out/'summary.json'),source_sha256=sha(Path(__file__)),R_protocol_sha256=sha(R/'protocol.json'),model_forwards=0),f,indent=2)
    print(json.dumps(dict(legacy_cells=200,legacy_checks_passed=summary['legacy_checks_passed'],correctness_transition_comparisons=len(transitions),new_model_forwards=0)))


if __name__=='__main__':main()
