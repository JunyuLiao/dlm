"""First-refresh bank agreement and score separation, without future tuning."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();out=root/'support_analysis001';out.mkdir(exist_ok=False)
    arms=['current_error_fixed4','response_guard_qdirection','margin_guard_weighted_bias','validity_reuse_mass_certificate','sham_fixed4']
    by_arm={}
    for arm in arms:
        by_arm[arm]={}
        for p in (root/'runs/panel').glob('*/'+arm+'/audit[0-9][0-9][0-9].json'):
            r=json.loads(p.read_text());banks=r['selector']['first_two_candidate_banks']
            if banks:by_arm[arm][r['id'],r['seed']]=dict(record=r,first_bank=banks[0])
    comparisons=[]
    for arm in arms[1:]:
        common=sorted(set(by_arm[arm])&set(by_arm[arms[0]]))
        for key in common:
            base=by_arm[arms[0]][key]['first_bank'];cur=by_arm[arm][key]['first_bank']
            if base['step']!=2 or cur['step']!=2 or base['prefix']!=cur['prefix']:
                raise RuntimeError('first selection landmark mismatch')
            identical_bank=True;distinct=0;matched=0;response_separated=0;eligible_alternatives=0
            unit_details=[]
            for a,b in zip(base['units'],cur['units']):
                def signature(unit):
                    return [(c['tiles'],c['feasible'],c['current_error'],c['deleted_mass']) for c in unit['candidates']]
                same=signature(a)==signature(b);identical_bank &= same
                chosen_a=a['chosen'];chosen_b=b['chosen']
                changed=(chosen_a['tiles'] if chosen_a else None)!=(chosen_b['tiles'] if chosen_b else None)
                distinct+=changed
                if a['ranking_identifying']:eligible_alternatives+=1
                pair=False;sep=False
                if chosen_a and chosen_b:
                    pair=abs(chosen_a['current_error']-chosen_b['current_error'])/max(chosen_a['current_error'],chosen_b['current_error'],1e-6)<=.05 and abs(chosen_a['deleted_mass']-chosen_b['deleted_mass'])<=.01
                    sep=abs(chosen_a['response']-chosen_b['response'])/max(abs(chosen_a['response']),abs(chosen_b['response']),1e-6)>=.25
                matched+=changed and pair;response_separated+=changed and pair and sep
                unit_details.append(dict(head=a['head'],query_block=a['query_block'],bank_equal=same,
                    feasible=a['feasible'],distinct_choice=changed,current_error_mass_matched=pair,response_25pct_separation=sep))
            comparisons.append(dict(id=key[0],seed=key[1],arm=arm,first_bank_inputs_agree=identical_bank,
                distinct_unit_choices=distinct,units_with_at_least4_feasible=eligible_alternatives,
                matched_distinct_unit_choices=matched,response_separated_matched_unit_choices=response_separated,
                units=unit_details,limitation='Common first selector boundary follows identical native bootstrap; later repeated interventions/trajectories are different, so this is candidate separation, not pulse propagation or independent predictive validation.'))
    result=dict(comparisons=comparisons,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_score_tuning=True,method_promoted=False)
    (out/'summary.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(comparisons=len(comparisons),first_banks_equal=sum(x['first_bank_inputs_agree'] for x in comparisons),
        distinct_unit_choices=sum(x['distinct_unit_choices'] for x in comparisons))))


if __name__=='__main__':main()
