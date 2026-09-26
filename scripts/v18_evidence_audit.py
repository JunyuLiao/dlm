"""Read-only CP0 audit of the frozen v15 LongBench panel. Emits no answers or gold."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def token_hash(tokens):
    return sha_bytes(json.dumps(tokens, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def completion_hash(tokens):
    return sha_bytes(json.dumps(tokens, separators=(',', ':')).encode())


def issue(out, code, subject=None):
    item = {'code': code}
    if subject is not None:
        item['subject'] = str(subject)
    out.append(item)


def unique_ids(rows, expected, issues):
    ids = [r.get('id') for r in rows]
    if len(ids) != len(set(ids)):
        issue(issues, 'duplicate_manifest_id')
    if set(ids) != set(expected) or len(ids) != len(expected):
        issue(issues, 'manifest_id_coverage')


def validate_warm(first, warm):
    reasons = []
    for row, label in ((first, 'attempt0'), (warm, 'warm')):
        if not row or not row.get('ok'):
            reasons.append(label + '_missing_or_failed')
            continue
        for field in ('completion_token_hash', 'per_canvas_calls', 'termination', 'phases'):
            value = row.get(field)
            if value is None or value == '' or value == [] or value == {}:
                reasons.append(label + '_' + field + '_absent')
        phases = row.get('phases')
        if isinstance(phases, dict) and any(phases.get(k) is None for k in ('anchor', 'reselect', 'held', 'routed_calls')):
            reasons.append(label + '_phases_incomplete')
    if warm.get('triton_misses') is None or warm.get('triton_disk_entries_added') is None:
        reasons.append('warm_compile_fields_absent')
    if warm.get('triton_misses') or warm.get('triton_disk_entries_added') or warm.get('new_shared_objects'):
        reasons.append('warm_new_compile_or_library')
    if first and first.get('ok') and warm.get('ok'):
        for field in ('completion_token_hash', 'per_canvas_calls', 'termination'):
            if first.get(field) != warm.get(field):
                reasons.append(field + '_mismatch')
    return sorted(set(reasons))


def require_equal_lengths(*arrays):
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1:
        raise ValueError('scorer_input_length_mismatch')


def audit(root: Path, repo: Path):
    public = repo / 'results/numerical_qk_longcontext_scored_20260926'
    protocol = json.loads((public / 'frozen_protocol.json').read_text())
    selection = json.loads((public / 'panel_selection.json').read_text())
    index = json.loads((public / 'private_receipt_index.json').read_text())
    issues = []
    manifest_path = root / 'private/generation_manifest.json'
    gold_path = root / 'private/gold_scorer_only.json'
    ledger_path = root / 'panel/ledger.jsonl'
    manifest = json.loads(manifest_path.read_text())
    # Gold is deliberately never parsed. Its byte identity is sufficient here.
    mh, gh, lh = sha_file(manifest_path), sha_file(gold_path), sha_file(ledger_path)
    for expected in (protocol['manifest_sha256'], protocol['task_identity']['generation_manifest_sha256'],
                     selection['private_generation_manifest_sha256'], index['generation_manifest']['sha256']):
        if mh != expected:
            issue(issues, 'manifest_byte_sha_mismatch')
    for expected in (selection['private_gold_sha256'], index['gold_scorer_only']['sha256']):
        if gh != expected:
            issue(issues, 'gold_byte_sha_mismatch')
    if lh != index['ledgers']['panel']['sha256']:
        issue(issues, 'ledger_byte_sha_mismatch')
    ids = protocol['ids']
    unique_ids(manifest, ids, issues)
    if set(protocol['prompt_hashes']) != set(ids):
        issue(issues, 'protocol_prompt_coverage')
    panel = {r['id']: r for r in selection['panel']}
    if set(panel) != set(ids) or len(selection['panel']) != len(ids):
        issue(issues, 'selection_id_coverage')
    forbidden = {'answer', 'expected', 'expected_answer', 'reference_solution', 'gold', 'solution'}
    for row in manifest:
        rid = row.get('id')
        if forbidden.intersection(row):
            issue(issues, 'gold_field_in_generation_manifest', rid)
        if rid not in protocol['prompt_hashes']:
            continue
        ph = sha_bytes(row['prompt'].encode())
        if ph != row.get('prompt_hash') or ph != protocol['prompt_hashes'][rid] or ph != panel[rid]['prompt_sha256']:
            issue(issues, 'prompt_sha_mismatch', rid)
        if not isinstance(row.get('prompt_tokens'), list) or len(row['prompt_tokens']) != row.get('prompt_tokens_n') or row['prompt_tokens_n'] != panel[rid]['prompt_tokens_n']:
            issue(issues, 'prompt_token_count_mismatch', rid)
    contract = protocol['task_identity']['contract']
    if contract != selection['contract']:
        issue(issues, 'contract_identity_mismatch')
    sources = {'task_module_sha256': repo / 'scripts/v15_longbench_task.py',
               'final_channel_source_sha256': repo / 'experiments/diffusion_gemma_aime26_modes/protocol.py',
               'dataset_sha256': Path('/home/exouser/.cache/huggingface/hub/datasets--THUDM--LongBench-v2/snapshots') / contract['dataset_revision'] / 'data.json'}
    for name, path in sources.items():
        if not path.is_file() or sha_file(path) != contract[name]:
            issue(issues, 'source_sha_mismatch', name)
    nemo = Path('/home/exouser/ljy/dlm/reference/NeMo-Skills')
    for name, expected in contract['nemo_files'].items():
        path = nemo / name
        if not path.is_file() or sha_file(path) != expected:
            issue(issues, 'nemo_source_sha_mismatch', name)
    if protocol['scoring']['version'] != contract['scorer_version']:
        issue(issues, 'scorer_version_mismatch')
    schedule = protocol['schedule']
    scheduled = {f"{e['cell_id']}:{e['role']}:{e['repeat']}": e for e in schedule}
    if len(scheduled) != len(schedule):
        issue(issues, 'duplicate_schedule_key')
    records = {}
    events = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    for e in events:
        if e.get('event') != 'run':
            continue
        key = e.get('execution_key')
        if key not in scheduled:
            issue(issues, 'foreign_record', key)
            continue
        if key in records:
            issue(issues, 'duplicate_record', key)
            continue
        spec = scheduled[key]
        if any(e.get(k) != spec[k] for k in ('index', 'block', 'arm', 'id', 'seed', 'role', 'repeat', 'cell_id')):
            issue(issues, 'record_schedule_mismatch', key)
        if e.get('generation_seed') != spec['seed']:
            issue(issues, 'generation_seed_mismatch', key)
        if e.get('arm_config_hash') != protocol['arm_hashes'][spec['arm']]:
            issue(issues, 'arm_hash_mismatch', key)
        records[key] = e
    if set(records) != set(scheduled):
        issue(issues, 'schedule_coverage_mismatch')
    by_id = {r['id']: r for r in manifest}
    panel_receipts = [r for r in index['receipts'] if r.get('ledger') == 'panel']
    receipt_index = {r['execution_key']: r for r in panel_receipts}
    if len(receipt_index) != len(panel_receipts):
        issue(issues, 'duplicate_receipt_index')
    success0 = {k for k, e in records.items() if e.get('role') == 'attempt0' and e.get('ok')}
    if set(receipt_index) != success0:
        issue(issues, 'receipt_coverage_mismatch')
    checked = 0
    for key in success0:
        e = records[key]
        meta = receipt_index.get(key)
        if not meta:
            continue
        path = Path(e.get('private_receipt', ''))
        if path != Path(meta['receipt']) or not path.is_file():
            issue(issues, 'receipt_path_mismatch', key)
            continue
        if sha_file(path) != meta['sha256']:
            issue(issues, 'receipt_byte_sha_mismatch', key)
        rec = json.loads(path.read_text())
        if any(rec.get(f) != e.get(f) for f in ('id', 'seed', 'condition', 'fingerprint', 'prompt_token_hash')):
            issue(issues, 'receipt_ledger_identity_mismatch', key)
        if rec.get('seed') != meta['seed'] or rec.get('id') != meta['id'] or e['arm'] != meta['arm']:
            issue(issues, 'receipt_index_identity_mismatch', key)
        row = by_id.get(e['id'])
        if row is None or rec.get('prompt_hash') != row.get('prompt_hash') or rec.get('prompt_token_hash') != token_hash(row['prompt_tokens']):
            issue(issues, 'receipt_prompt_mismatch', key)
        if completion_hash(rec['completion_tokens']) != e.get('completion_token_hash'):
            issue(issues, 'completion_token_hash_mismatch', key)
        if rec.get('termination_reason') != e.get('termination') or rec.get('total_decoder_calls') != e.get('decoder_calls'):
            issue(issues, 'receipt_work_termination_mismatch', key)
        if [c['decoder_calls'] for c in rec['per_canvas']] != e.get('per_canvas_calls'):
            issue(issues, 'receipt_canvas_calls_mismatch', key)
        checked += 1
    warm_rejected = 0
    for key, e in records.items():
        if e.get('role') != 'warm':
            continue
        first = records.get(f"{e['cell_id']}:attempt0:0")
        reasons = validate_warm(first, e)
        if reasons:
            warm_rejected += 1
            for reason in reasons:
                issue(issues, 'warm_' + reason, key)
        accepted = bool(e.get('acceptance', {}).get('accepted'))
        if accepted != (not reasons):
            issue(issues, 'stored_warm_acceptance_disagrees', key)
    starts = [e for e in events if e.get('event') == 'start']
    if not starts:
        issue(issues, 'missing_start_identity')
    for e in starts:
        if e.get('protocol_id') != protocol['protocol_id'] or e.get('model_revision') != protocol['model_revision'] or e.get('arm_hashes') != protocol['arm_hashes']:
            issue(issues, 'start_identity_mismatch')
    # The v15 scorer used zip() without a length guard; report this explicitly.
    scorer_text = (repo / 'scripts/v15_longbench_task.py').read_text()
    if 'zip(texts, preds, golds, terminations)' in scorer_text and 'len(raw_completions) == len(golds)' not in scorer_text:
        issue(issues, 'scorer_input_length_guard_absent')
    require_equal_lengths([0] * len(success0), [0] * checked)
    return {'schema': 'v18_cp0_evidence_audit_v1', 'status': 'pass' if not issues else 'blocked',
            'protocol_id': protocol['protocol_id'], 'manifest_sha256': mh, 'gold_sha256': gh,
            'ledger_sha256': lh, 'manifest_items': len(manifest), 'scheduled_executions': len(schedule),
            'ledger_executions': len(records), 'successful_attempt0_receipts_checked': checked,
            'warm_rows': sum(e.get('role') == 'warm' for e in records.values()),
            'strict_warm_rejected': warm_rejected, 'prefilter_over_400k_chars': selection['prefiltered_over_400k_chars'],
            'pool_status': 'heuristic_prefiltered_not_token_proven',
            'issues': issues, 'issue_counts': dict(Counter(i['code'] for i in issues))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.root, a.repo)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('issues',)}, sort_keys=True))
    raise SystemExit(0 if result['status'] == 'pass' else 2)


if __name__ == '__main__':
    main()
