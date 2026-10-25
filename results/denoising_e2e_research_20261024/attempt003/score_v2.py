"""Score full private completions with pinned official task rules."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path('/home/exouser/ljy/dlm')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from scripts import v31_official as official
from scripts import v27_humaneval as he


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    dest = root / 'scoring001'
    dest.mkdir(exist_ok=False)
    protocol = json.loads((root / 'e2e_protocol.json').read_text())
    for p, h in protocol['scorer_sources'].items():
        if sha(p) != h:
            raise RuntimeError('scorer source drift')
    rows = {r['id']: r for r in json.loads((root / 'private/tokenized_manifest.json').read_text())}
    gold_path = REPO / 'results/humaneval_value_observe_no_overlap_k256_20261019/attempt001/private/pools/thinking_on/gold.json'
    gold = {r['task_id']: r for r in json.loads(gold_path.read_text()).values()}
    final, final_source = official.load_final_response()
    numeric, numeric_source = official.aime_numeric_score()
    he.sandbox_preflight()
    outcomes, bindings = [], []
    for suite in ['short', 'long']:
        for arm in protocol['arms']:
            bindings=json.loads((root/'result_directory_bindings.json').read_text())
            leaf = root / bindings['result_paths'].get(suite,{}).get(arm,'runs/panel/'+suite+'/'+arm)
            if not (leaf / 'final_complete.json').exists():
                continue
            for record_path in sorted(leaf.glob('clean[0-9][0-9][0-9].json')):
                record = json.loads(record_path.read_text())
                row = rows[record['id']]
                if record['protocol_sha256'] != sha(root / 'e2e_protocol.json'):
                    raise RuntimeError('record protocol mismatch')
                if record['prompt_hash'] != row['prompt_hash'] or record['prompt_tokenization_sha256'] != hashlib.sha256(json.dumps(row['prompt_tokens']).encode()).hexdigest():
                    raise RuntimeError('prompt identity drift')
                private_path = root / 'private/generations/panel' / suite / arm / record_path.name
                private = json.loads(private_path.read_text())
                if private['output_sha256'] != record['output_sha256']:
                    raise RuntimeError('generation/record mismatch')
                raw = private['completion']
                response = final(raw, row['thinking'])
                detail = {}
                if row['benchmark'] == 'aime26':
                    result = numeric(response, row['expected'])
                    passed = result['correct']; detail['parsed'] = result['extracted'] is not None
                elif row['benchmark'] == 'humaneval':
                    problem = gold[row['source_id']]
                    code, reason = he.extract_code(raw, problem['entry_point'])
                    if code is None:
                        passed, outcome = False, 'not_run'
                    else:
                        passed, outcome = he.run_test(he.program_for(problem, code))
                    if outcome == 'timeout':
                        passed, outcome = he.run_test(he.program_for(problem, code))
                    detail.update(extraction=reason, outcome=outcome)
                else:
                    pred = official.lb_extract_answer(response)
                    passed = pred == row['expected']; detail['parsed'] = pred is not None
                outcomes.append(dict(id=row['id'], seed=record['seed'], arm=arm, suite=suite,
                    benchmark=row['benchmark'], correct=bool(passed), finish_reason=record['finish_reason'],
                    no_final_response=not response, **detail))
                bindings.append(dict(id=row['id'], seed=record['seed'], arm=arm,
                    record_sha256=sha(record_path), private_generation_sha256=sha(private_path),
                    output_sha256=record['output_sha256'], generation_budget=record['generation_budget']))
    result = dict(protocol_sha256=sha(root / 'e2e_protocol.json'), outcomes=outcomes,
        bindings=bindings, human_eval_sandbox_preflight='passed',
        final_response_source=final_source, numeric_score_source=numeric_source,
        scorer_sources={str(p): sha(p) for p in [Path(he.__file__), Path(official.__file__), gold_path]},
        limits='Exposed25-question two-seed screen;10AIME/10HumanEval/5LongBench items; question-cluster uncertainty, no formal noninferiority.')
    (dest / 'official_scores.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    (dest / 'scoring_complete.json').write_text(json.dumps(dict(passed=True,
        records=len(outcomes), scores_sha256=sha(dest / 'official_scores.json')), indent=2) + '\n')
    print(json.dumps(dict(scored=len(outcomes), arms=sorted({x['arm'] for x in outcomes}))))


if __name__ == '__main__':
    main()
