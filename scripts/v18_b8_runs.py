"""Run the frozen B8 16-execution protocol through the v15 resumable driver."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def check_protocol(path: Path) -> None:
    from scripts.v13_seed_runs import execution_key

    p = json.loads(path.read_text())
    s = p['schedule']
    if p['schema'] != 'v18_b8_no_risk_export_v1' or len(s) != 16 or len({execution_key(e) for e in s}) != 16:
        raise ValueError('expected frozen 16-execution B8 protocol')
    if p['seeds'] != [17] or set(p['arms']) != {'B8_P', 'B8_no_risk_export'}:
        raise ValueError('B8 arm or seed mismatch')


def verify_ledger(protocol_path: Path, ledger_path: Path) -> dict:
    from scripts.v13_seed_runs import execution_key

    check_protocol(protocol_path)
    p = json.loads(protocol_path.read_text())
    events = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    run_events = [e for e in events if e.get('event') == 'run']
    runs = {e['execution_key']: e for e in run_events}
    if len(run_events) != 16 or len(runs) != 16 or set(runs) != {execution_key(e) for e in p['schedule']}:
        raise ValueError('ledger does not contain exactly the frozen 16 executions')
    groups = {}
    for entry in p['schedule']:
        row = runs[execution_key(entry)]
        for field in ('id', 'seed', 'arm', 'role', 'repeat', 'cell_id'):
            if row.get(field) != entry[field]:
                raise ValueError(f'ledger {field} differs from frozen schedule')
        groups.setdefault((entry['id'], entry['seed'], entry['role']), {})[entry['arm']] = row
    differences = []
    for cell, arms in groups.items():
        old, new = arms['B8_P'], arms['B8_no_risk_export']
        if not old.get('ok') or not new.get('ok'):
            differences.append(dict(cell=cell, reason='execution_failure'))
            continue
        for field in ('completion_token_hash', 'per_canvas_calls', 'termination', 'phases'):
            if old.get(field) is None or new.get(field) is None:
                differences.append(dict(cell=cell, reason=f'{field}_missing'))
            elif old[field] != new[field]:
                differences.append(dict(cell=cell, reason=field))
        if cell[2] == 'warm':
            for arm, row in arms.items():
                if not row.get('acceptance', {}).get('accepted'):
                    differences.append(dict(cell=cell, reason=f'{arm}_warm_rejected'))
    return dict(equivalent=not differences, groups=len(groups), executions=len(runs), differences=differences)


def b8_phases(receipt: dict) -> dict | None:
    counters = receipt.get('counters') or {}
    fields = ('anchors', 'ordinary', 'fallback_native', 'calls')
    if any(not isinstance(counters.get(field), int) or counters[field] < 0 for field in fields):
        return None
    if counters['anchors'] + counters['ordinary'] + counters['fallback_native'] != counters['calls']:
        return None
    return {field: counters[field] for field in fields}


def run() -> None:
    from scripts import v15_seed_runs
    from scripts import v9_clean_request_timing as timing

    original_redacted = timing.redacted
    original_acceptance = v15_seed_runs.warm_acceptance

    def redacted(receipt):
        return dict(original_redacted(receipt), phases=b8_phases(receipt))

    def warm_acceptance(first, warm):
        result = original_acceptance(first, warm)
        for field in ('completion_token_hash', 'per_canvas_calls', 'termination', 'phases'):
            if first is None or first.get(field) is None or warm.get(field) is None:
                result['reasons'].append(f'{field}_missing')
            elif first[field] != warm[field]:
                result['reasons'].append(f'{field}_mismatch')
        result['accepted'] = not result['reasons']
        return result

    timing.redacted = redacted
    v15_seed_runs.warm_acceptance = warm_acceptance
    v15_seed_runs.main()


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'verify':
        result = verify_ledger(Path(sys.argv[2]), Path(sys.argv[3]))
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['equivalent'] else 1)
    if '--protocol' not in sys.argv:
        raise SystemExit('--protocol required')
    check_protocol(Path(sys.argv[sys.argv.index('--protocol') + 1]))
    run()
