"""Read-only historical-reference check of the observed75% generation collapse."""
import json
from pathlib import Path

from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_value_aware_followup.evidence import DECODING_FIELDS
from .protocol import ROOT, sha
from .workflow import semantic_policy


def historical_parity(root=ROOT):
    cases = []
    sources = {}
    for name in ('blasst_aggressive', 'value', 'mass', 'aligned'):
        path = root/f'verified_policies/aime26/{name}_s75.json'
        policy = json.loads(path.read_text())
        sources[str(path)] = sha(path.read_bytes())
        trace = next(t for t in policy['trace'] if t['iteration']==policy['selected_round'])
        for identity, source in sorted(trace['sources'].items()):
            path = Path(source['path'])
            if sha(path.read_bytes()) != source['sha256']:
                raise ValueError('Historical reference changed')
            ref = json.loads(path.read_text())
            if ref['fingerprint'] == policy['fingerprint']:
                raise ValueError('This check requires the prior unaccelerated reference')
            sources[str(path)] = source['sha256']
            path = shard_path(root, 'final', name+'_s75', identity)
            out = json.loads(path.read_text())
            sources[str(path)] = sha(path.read_bytes())
            if any(ref[k]!=out[k] for k in ('id','seed','prompt_hash','generation_budget','config')):
                raise ValueError('Reference settings mismatch')
            if any(ref['generation_metadata'].get(k)!=out['generation_metadata'].get(k) for k in DECODING_FIELDS):
                raise ValueError('Reference decoding settings mismatch')
            if semantic_policy(ref['config'],ref['thresholds']) != semantic_policy(out['config'],out['thresholds']):
                raise ValueError('Reference numerical threshold mismatch')
            def counts(data):
                return {(r['layer'],r['head'],r['step'],r['attention_type']):
                        (r['eligible'],r['skipped']) for r in data['records']}
            cases.append(dict(id=identity, condition=name+'_s75', generation_budget=ref['generation_budget'],
                exact_tokens=ref['completion_tokens']==out['completion_tokens'],
                exact_physical_counts=counts(ref)==counts(out), reference_score=ref['score'],
                accelerated_score=out['score'], generated_tokens=len(out['completion_tokens'])))
    if len(cases)!=24 or not all(r['exact_tokens'] and r['exact_physical_counts'] for r in cases):
        raise ValueError('Historical full-budget generation/count parity failed')
    result = dict(passed=True, cases=cases, sources=sources, inference_performed=False,
        interpretation='All24 cached native-reference75% AIME calibration generations and physical counters match. Their zero scores predate GPU acceleration; no algorithm or final result was changed.')
    _write(root/'historical75_parity.json', result)
    return result


if __name__ == '__main__':
    result = historical_parity()
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','cases')}))
