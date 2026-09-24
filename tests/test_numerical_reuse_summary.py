import json
from pathlib import Path

import pytest

from scripts.native_reuse_summarize import PAIR_CONTROL_FIELDS, collect, write


def fixture_run(root: Path, arm: str, *, question='q1', seed=42, prompt_hash='a'*64,
                wall=2., calls=(3, 2), routing=None, counters=None, capped=False,
                timing=True, control_override=None):
    phase = 'smoke'
    config = dict(fingerprint=f'fingerprint-{arm}', condition=arm, phase=phase,
                  manifest_sha256='manifest', model='/model', model_metadata_hashes={'config.json': 'hash'},
                  revision='pinned', dtype='bfloat16', max_new_tokens=8192,
                  thinking=True, temperature=0., eos_enabled=True,
                  native_adaptive=True, policy_sha256='policy', policy_name='T_s50',
                  diagnostic=False, timing_events=timing)
    for name in PAIR_CONTROL_FIELDS:
        assert name in config
    config.update(control_override or {})
    target = root / 'configs' / f'{phase}.{arm}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config))
    receipt = dict(schema='numerical_qk_attempt0_v1', attempt=0,
                   phase=phase, condition=arm, id=question, seed=seed,
                   fingerprint=config['fingerprint'], prompt_hash=prompt_hash,
                   prompt='PRIVATE PROMPT', raw_completion='PRIVATE ANSWER',
                   completion_tokens=[918271], prediction='PRIVATE PREDICTION',
                   request_wall_seconds=wall,
                   generation_gpu_timeline_seconds=wall-.2 if timing else None,
                   generation_gpu_timeline_note='First encoder end to final event; includes host gaps and later encoder work'
                       if timing else 'CUDA timing events disabled',
                   output_tokens=8192 if capped else 400,
                   termination_reason='length' if capped else 'eos',
                   per_canvas=[dict(canvas_index=i, decoder_calls=n) for i, n in enumerate(calls)],
                   total_decoder_calls=sum(calls), routing=routing, counters=counters)
    path = root / phase / arm / f'seed_{seed}' / f'{question}.attempt0.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt))
    return path


def fixture_quality(path: Path, arm: str, *, question='q1', seed=42,
                    correct=False, capped=False, unparsed=False):
    payload = dict(schema='numerical_qk_quality_v1', records=[
        dict(phase='smoke', condition=arm, id=question, seed=seed,
             fingerprint=f'fingerprint-{arm}', capped=capped,
             unparsed=unparsed,
             final_response='PRIVATE FINAL RESPONSE',
             score=dict(correct=correct, extracted=None if unparsed else '1'))])
    path.write_text(json.dumps(payload))
    return path


def test_collect_all_attempt0_and_normalize_work_without_answer_leak(tmp_path):
    dense, reuse, fresh = (tmp_path / name for name in ('dense', 'reuse', 'fresh'))
    fixture_run(dense, 'native_dense', wall=2., calls=(3, 2))
    fixture_run(reuse, 'M1', wall=1., calls=(4, 1), capped=True,
                routing=[dict(physical_skipped=2, physical_eligible=5),
                         dict(physical_skipped=1, physical_eligible=3)],
                counters=dict(score_refresh_calls=1, decision_refresh_calls=2,
                              current_qk_elements=1000, reused_qk_elements=3000))
    fixture_run(fresh, 'fresh_junyu_T', wall=1.5, calls=(5,),
                routing=[dict(skipped=3, eligible=8)])
    q1 = fixture_quality(tmp_path / 'dense_quality.json', 'native_dense', correct=True)
    q2 = fixture_quality(tmp_path / 'reuse_quality.json', 'M1', capped=True, unparsed=True)
    q3 = fixture_quality(tmp_path / 'fresh_quality.json', 'fresh_junyu_T', correct=True)
    rows, summary = collect([dense, reuse, fresh], [q1, q2, q3])
    assert len(rows) == 3
    by_arm = {row['arm']: row for row in rows}
    assert by_arm['M1']['per_canvas_calls'] == [4, 1]
    assert by_arm['M1']['total_calls'] == 5
    assert by_arm['M1']['capped'] and by_arm['M1']['unparsed'] and not by_arm['M1']['correct']
    assert by_arm['M1']['score_refresh_calls'] == 1
    assert by_arm['M1']['decision_refresh_calls'] == 2
    assert by_arm['M1']['current_qk_elements'] == 1000
    assert by_arm['M1']['reused_score_elements'] == 3000
    assert (by_arm['M1']['pv_physical_skipped'], by_arm['M1']['pv_physical_eligible']) == (3, 8)
    assert (by_arm['fresh_junyu_T']['pv_physical_skipped'],
            by_arm['fresh_junyu_T']['pv_physical_eligible']) == (3, 8)
    assert by_arm['fresh_junyu_T']['current_qk_elements'] is None
    assert by_arm['native_dense']['pv_physical_skipped'] is None
    assert all('host gaps' in row['device_generation_timeline_note'] for row in rows)
    assert len(summary['pairs']) == 2
    assert {x['arm']: x['whole_wall_speedup'] for x in summary['pairs']} == {
        'M1': 2., 'fresh_junyu_T': 2/1.5}
    assert {x['arm']: x['correct'] for x in summary['arms']}['M1'] == 0
    json_path, csv_path = tmp_path / 'summary.json', tmp_path / 'summary.csv'
    write(rows, summary, json_path, csv_path)
    text = json_path.read_text() + csv_path.read_text()
    for private in ('PRIVATE PROMPT', 'PRIVATE ANSWER', 'PRIVATE PREDICTION',
                    'PRIVATE FINAL RESPONSE', '918271'):
        assert private not in text
    assert 'time_between_tokens' not in text


def test_duplicate_attempt0_and_bad_canvas_are_rejected(tmp_path):
    first, second = tmp_path / 'a', tmp_path / 'b'
    fixture_run(first, 'M1')
    fixture_run(second, 'M1')
    with pytest.raises(ValueError, match='Duplicate attempt-0'):
        collect([first, second])
    receipt = next(first.rglob('*.attempt0.json'))
    value = json.loads(receipt.read_text())
    value['per_canvas'][1]['decoder_calls'] = 9
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Total decoder calls'):
        collect([first])


def test_pairing_requires_matching_prompt_and_common_controls(tmp_path):
    dense, changed_prompt, changed_config = (tmp_path / name for name in ('dense', 'prompt', 'config'))
    fixture_run(dense, 'native_dense')
    fixture_run(changed_prompt, 'M1', prompt_hash='b'*64)
    fixture_run(changed_config, 'M3', control_override=dict(timing_events=False), timing=False)
    rows, summary = collect([dense, changed_prompt, changed_config])
    assert len(rows) == 3
    assert summary['pairs'] == []


def test_quality_must_cover_every_attempt_in_scored_arm(tmp_path):
    root = tmp_path / 'run'
    fixture_run(root, 'M1', question='q1')
    fixture_run(root, 'M1', question='q2')
    q = fixture_quality(tmp_path / 'quality.json', 'M1', question='q1')
    with pytest.raises(ValueError, match='omits attempt-0'):
        collect([root], [q])
