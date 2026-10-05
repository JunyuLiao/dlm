"""Freeze gold-free v18 RULER/AIME manifests and paired execution schedules (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.v13_seed_runs import plan_schedule

SEEDS = (101, 202, 303)
RULER_ARMS = ('D_native', 'D_matched', 'U50', 'U60', 'U70', 'T50', 'T60', 'T70')
AIME_ARMS = ('D_native', 'D_matched', 'U50', 'U60', 'T50', 'T60')
REVISION = 'f7f5b7f5fa82ffc52addd066915886d497f5517b'
GOLD_KEYS = frozenset(('outputs', 'answer', 'expected', 'expected_answer', 'gold', 'solution', 'reference_solution'))


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode()).hexdigest()


def read_rows(path):
    raw = Path(path).read_text()
    rows = json.loads(raw) if raw.lstrip().startswith('[') else [json.loads(s) for s in raw.splitlines() if s.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError('empty or malformed manifest')
    return rows


def strip_gold(rows, *, unique=True):
    result = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('prompt'), str) or not row['prompt'] or not row.get('id'):
            raise ValueError('manifest row needs id and prompt')
        clean = {k: v for k, v in row.items() if k not in GOLD_KEYS}
        if clean.get('prompt_hash') != sha(clean['prompt']):
            raise ValueError('prompt hash mismatch')
        if not isinstance(clean.get('generation_budget', 8192), int) or not 1 <= clean.get('generation_budget', 8192) <= 8192:
            raise ValueError('generation budget mismatch')
        if 'prompt_tokens' in clean and clean.get('prompt_token_count', len(clean['prompt_tokens'])) != len(clean['prompt_tokens']):
            raise ValueError('prompt token count mismatch')
        if any(k in clean for k in GOLD_KEYS):
            raise AssertionError('gold leaked')
        result.append(clean)
    if unique and len({r['id'] for r in result}) != len(result):
        raise ValueError('duplicate manifest id')
    return result


def generation_rows(rows, *, thinking=True):
    # Generation seed lives only in the frozen schedule, not in a question manifest row.
    return [{**{k: v for k, v in row.items() if k not in ('seed', 'generator_seed')},
             'thinking': thinking} for row in rows]


def verify_prompt_tokens(adapter, rows):
    """CPU tokenizer check of the exact frozen list; returns counts only."""
    checked = 0
    for row in rows:
        if type(row.get('thinking')) is not bool or not isinstance(row.get('prompt_tokens'), list):
            raise ValueError('prompt token list or thinking flag absent')
        actual = adapter.encode_prompt(row['prompt'], {'thinking': row['thinking']})
        if actual != row['prompt_tokens'] or len(actual) != row.get('prompt_token_count', len(actual)):
            raise ValueError('frozen prompt token mismatch: ' + str(row['id']))
        checked += 1
    return checked


def timing_subset(rows):
    tasks = sorted({r['task'] for r in rows})
    if len(tasks) != 13 or Counter(r['task'] for r in rows) != {task: 10 for task in tasks}:
        raise ValueError('RULER panel must have 13 tasks with 10 distinct questions each')
    return [r['id'] for task in tasks for r in sorted((r for r in rows if r['task'] == task),
                                                       key=lambda r: sha('v18/timing/' + r['id']))[:2]]


def arm_specs(names):
    result = {}
    for name in names:
        kind = name[0] if name[0] in ('U', 'T') else name
        target = int(name[1:]) if kind in ('U', 'T') else None
        result[name] = dict(frontier_arm=name, method={'U': 'unweighted', 'T': 'T',
                                                       'D_matched': 'kernel_dense', 'D_native': 'native_dense'}.get(kind, kind),
                            target=target, fast_t=kind == 'T',
                            support_geometry='native_legal', rank=32, projection_seed=1729,
                            canvas=256, max_denoising_steps=48)
    return result


def build(name, rows, arms, warm_ids, *, schedule_seed, authorization, source_sha, construction):
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate question IDs')
    if not set(warm_ids) <= set(ids):
        raise ValueError('warm IDs outside panel')
    specs = arm_specs(arms)
    arm_hashes = {k: sha(json.dumps(v, sort_keys=True)) for k, v in specs.items()}
    generation = dict(max_new_tokens=('per_manifest_row' if name == 'ruler4k' else 8192),
                      thinking=(name != 'ruler4k'), native_adaptive=True,
                      temperature_schedule=[0.8, 0.4], confidence_threshold=0.005,
                      stability_threshold=1, entropy_bound=0.1)
    protocol_id = 'v18_' + name + '_' + sha(json.dumps([ids, specs, source_sha, generation], sort_keys=True))[:12]
    schedule = plan_schedule(protocol_id, REVISION, arm_hashes, ids, list(SEEDS), schedule_seed)
    schedule = [e for e in schedule if e['role'] == 'attempt0' or e['id'] in warm_ids]
    for i, e in enumerate(schedule):
        e['index'] = i
    return dict(schema='v18_frontier_protocol_v1', status='calibration_pending', protocol_id=protocol_id, dataset=name,
                construction=construction, ids=ids, seeds=list(SEEDS), arms=specs, arm_hashes=arm_hashes,
                model_revision=REVISION, source_manifest_sha256=source_sha,
                prompt_hashes={r['id']: r['prompt_hash'] for r in rows},
                warm_ids=list(warm_ids), schedule_seed=schedule_seed, schedule=schedule,
                authorization=authorization, generation=generation,
                planned_executions=len(schedule))


def freeze(ruler_final, ruler_cal, aime, private: Path, out: Path, authorization: str):
    raw_ruler = read_rows(ruler_final)
    raw_cal = read_rows(ruler_cal)
    raw_aime = read_rows(aime)
    ruler, cal, aime_rows = map(strip_gold, (raw_ruler, raw_cal, raw_aime))
    if len(ruler) != 130 or len(cal) != 26 or len(aime_rows) != 30:
        raise ValueError('expected 130 RULER panel, 26 RULER calibration, 30 AIME questions')
    if any(r.get('generation_budget') not in (30, 32, 50, 120, 128) for r in ruler + cal):
        raise ValueError('RULER task budget outside frozen set')
    if any(r.get('generation_budget', 8192) != 8192 for r in aime_rows):
        raise ValueError('AIME budget must remain 8192')
    if set(r['id'] for r in ruler) & set(r['id'] for r in cal):
        raise ValueError('calibration/evaluation ID overlap')
    if set(r['prompt_hash'] for r in ruler) & set(r['prompt_hash'] for r in cal):
        raise ValueError('calibration/evaluation prompt overlap')
    if Counter(r['task'] for r in cal) != {task: 2 for task in sorted({r['task'] for r in ruler})}:
        raise ValueError('calibration task balance mismatch')
    if {r['id'] for r in aime_rows} != {f'aime26/{i}' for i in range(1, 31)}:
        raise ValueError('AIME ID coverage mismatch')
    timing_ids = timing_subset(ruler)
    manifests = {'ruler': generation_rows(ruler, thinking=False),
                 'ruler_calibration': generation_rows(cal, thinking=False),
                 'aime': generation_rows(aime_rows, thinking=True)}
    protocols = {
        'ruler': build('ruler4k', ruler, RULER_ARMS, timing_ids, schedule_seed=2026092601,
                       authorization=authorization, source_sha=sha(Path(ruler_final).read_bytes()),
                       construction='development_replication_of_existing_130'),
        'aime': build('aime26', aime_rows, AIME_ARMS, [r['id'] for r in aime_rows], schedule_seed=2026092602,
                      authorization=authorization, source_sha=sha(Path(aime).read_bytes()),
                      construction='existing_aime26_all30_new_generation_seeds'),
    }
    for name, value in manifests.items():
        path = private / f'{name}_generation_manifest.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, sort_keys=True) + '\n'
        if path.exists() and path.read_text() != payload:
            raise ValueError('generation manifest would overwrite a different frozen identity')
        path.write_text(payload)
        if name in protocols:
            protocols[name]['generation_manifest_sha256'] = sha(path.read_bytes())
            protocols[name]['generation_manifest_path'] = str(path.resolve())
    for name, value in protocols.items():
        path = out / f'{name}_frozen_protocol.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, indent=2, sort_keys=True) + '\n'
        if path.exists() and path.read_text() != payload:
            raise ValueError('protocol would overwrite a different frozen identity')
        path.write_text(payload)
    return protocols


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('ruler-final', 'ruler-cal', 'aime', 'private', 'out', 'authorization'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    result = freeze(a.ruler_final, a.ruler_cal, a.aime, Path(a.private), Path(a.out), a.authorization)
    print(json.dumps({k: v['planned_executions'] for k, v in result.items()}, sort_keys=True))


if __name__ == '__main__':
    main()
