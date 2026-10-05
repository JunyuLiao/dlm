"""CPU audit of actual numerical-cache dispatch records; never exports answers."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit(path):
    data = json.loads(path.read_text())
    arm = data['condition']
    if arm == 'fresh_junyu_T':
        rows = data['routing']
        assert rows and all(0 <= r['skipped'] <= r['eligible'] for r in rows)
        by_kind = {}
        for kind in ('global', 'local'):
            selected = [r for r in rows if r['attention_type'] == kind]
            by_kind[kind] = dict(per_head_records=len(selected),
                eligible=sum(r['eligible'] for r in selected),
                skipped=sum(r['skipped'] for r in selected))
        return dict(id=data['id'], arm=arm, seed=data['seed'], source_fingerprint=data['fingerprint'],
            raw_path=str(path.resolve()), raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            decoder_calls=data['total_decoder_calls'], per_canvas_calls=[c['decoder_calls'] for c in data['per_canvas']],
            counters={}, physical_decision_units=by_kind, dispatch_audit='PASS',
            scope='Original Junyu per-head physical PV records; current QK executed, element counts not recorded; not a memory-traffic profile')
    assert arm in ('M1', 'M3')
    rows = data['routing']
    counts = data['counters']
    interval = 1 if arm == 'M1' else 2
    assert len(rows) == 30 * data['total_decoder_calls']
    keys = [(r['canvas'], r['decoder_call'], r['layer']) for r in rows]
    assert len(set(keys)) == len(keys)
    assert set(r['layer'] for r in rows) == set(range(30))
    for r in rows:
        step = r['decoder_call']
        assert r['score_anchor'] == step // 8 * 8
        assert r['score_age'] == step % 8
        assert r['score_refresh'] == (step % 8 == 0)
        assert r['decision_refresh'] == (step % interval == 0)
        assert (r['current_qk_elements'] > 0) == r['score_refresh']
        assert 0 <= r['physical_skipped'] <= r['physical_eligible']
    assert counts['unsupported_mask_refreshes'] == 0
    assert counts['attention_calls'] == len(rows)
    assert counts['score_refresh_calls'] == sum(r['score_refresh'] for r in rows)
    assert counts['decision_refresh_calls'] == sum(r['decision_refresh'] for r in rows)
    assert counts['current_qk_elements'] == sum(r['current_qk_elements'] for r in rows)
    by_kind = {}
    for kind in ('global', 'local'):
        selected = [r for r in rows if r['kind'] == kind]
        by_kind[kind] = dict(calls=len(selected),
            eligible=sum(r['physical_eligible'] for r in selected),
            skipped=sum(r['physical_skipped'] for r in selected))
    return dict(id=data['id'], arm=arm, seed=data['seed'], source_fingerprint=data['fingerprint'],
        raw_path=str(path.resolve()), raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        decoder_calls=data['total_decoder_calls'], per_canvas_calls=[c['decoder_calls'] for c in data['per_canvas']],
        score_age_histogram=dict(Counter(r['score_age'] for r in rows)), counters=counts,
        physical_decision_units=by_kind, dispatch_audit='PASS',
        scope='Dispatch records plus independent GPU throwing-spy test; not DRAM traffic or instruction profiler')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=('M1', 'M3', 'fresh_junyu_T'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.root / 'smoke' / args.arm).glob('seed_*/*.attempt0.json'))
    if not paths:
        raise ValueError('No completed attempt-0 receipts')
    result = dict(schema='numerical_reuse_dispatch_audit_v1', records=[audit(p) for p in paths])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(arm=args.arm, completed=len(paths), status='PASS')))


if __name__ == '__main__':
    main()
