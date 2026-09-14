"""Independent counter/coverage and saved-snapshot checks; CPU only."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from dllm.evaluation.ruler.io import read_jsonl
from .runner import ROOT, BASE


def audit(root=ROOT):
    original={r['sample_id']:r for r in read_jsonl(BASE/'dense/predictions.jsonl')}
    coverage={}
    for condition in ('lambda1_forward','lambda1_reverse','dense_eager'):
        out=root/condition
        assert (out/'summary.json').exists()
        predictions=read_jsonl(out/'predictions.jsonl')
        assert len(predictions)==50 and {r['sample_id'] for r in predictions}==set(original)
        for row in predictions:
            for field in ('prompt','inference_seed','outputs','tokens_to_generate'):
                assert row[field]==original[row['sample_id']][field]
            assert row['completion_tokens'] and all(isinstance(x,int) and x>=0 for x in row['completion_tokens'])
        if condition=='dense_eager':
            continue
        with (out/'attention_stats/per_head.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        by_example=defaultdict(set)
        for row in rows:
            by_example[row['example_id']].add((int(row['layer']),int(row['head'])))
            assert float(row['effective_blasst_lambda'])==1.
            assert int(row['eligible_tiles'])==int(row['skipped_tiles'])+int(row['retained_tiles'])
            assert 0<=int(row['skipped_valid_elements'])<=int(row['valid_elements'])
        expected={(layer,head) for layer in range(30) for head in range(16)}
        assert set(by_example)==set(original)
        assert all(pairs==expected for pairs in by_example.values())
        with (out/'attention_stats/per_step.csv').open() as handle:
            steps=list(csv.DictReader(handle))
        for field in ('eligible_tiles','skipped_tiles','valid_elements','skipped_valid_elements'):
            assert sum(int(r[field]) for r in rows)==sum(int(r[field]) for r in steps)
        coverage[condition]=dict(examples=len(by_example),layers=30,heads=16,head_rows=len(rows),step_rows=len(steps))
    checked=0
    for row in read_jsonl(root/'same_state/predictions.jsonl'):
        out=root/'same_state/shards'/row['sample_id']
        records=json.loads((out/'diagnostics.json').read_text())
        with np.load(out/'snapshots.npz') as arrays:
            for record in records:
                name=record['snapshot_key']
                maxima=arrays[name+'_maxima']; counts=arrays[name+'_counts']; mass=arrays[name+'_mass']
                valid=counts>0; eligible=valid.any(0)
                assert np.allclose(mass.sum(-1),valid.any(-1),atol=1.e-5)
                for order in ('forward','reverse'):
                    ordered=maxima if order=='forward' else maxima[:,::-1]
                    previous=np.concatenate([np.full((len(maxima),1),-np.inf),np.maximum.accumulate(ordered,axis=-1)[:,:-1]],axis=-1)
                    with np.errstate(invalid='ignore'):
                        keep=ordered>=previous
                    if order=='reverse':
                        keep=keep[:,::-1]
                    keep &= valid
                    expected=record['policies'][order+'_1']
                    assert int(eligible.sum())==expected['eligible_tiles']
                    assert int((eligible & ~keep.any(0)).sum())==expected['skipped_tiles']
                    assert int((counts*~keep).sum())==expected['skipped_elements']
                    assert np.isclose((mass*keep).sum(),expected['retained_mass_sum'],atol=1.e-4)
                for key in ('closure_1','matched_blasst_rank','matched_mass_oracle'):
                    reference='forward_1' if key=='closure_1' else 'sol_50'
                    assert record['policies'][key]['skipped_tiles']==record['policies'][reference]['skipped_tiles']
                assert record['policies']['closure_1']['retained_mass_sum']+1.e-5>=record['policies']['forward_1']['retained_mass_sum']
                checked+=1
    return dict(endpoint_coverage=coverage,independently_recomputed_snapshots=checked,
                conserved_head_step_counters=True,matched_budget_checks=True)
