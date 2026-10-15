"""Export a small, sanitized research snapshot without changing source runs.

Only the explicit aggregate allowlist below can be exported. This never copies
prompts, private JSONL, tensors, caches, executable snapshots, or live queues
back into a campaign. Output must be a new directory below results/.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'model_inventory.json': 'v31_full503_refresh_20261014/model_source_fingerprints.json',
    'value_methods/original_audit.json': 'v31_value_selectors_20261008/diagnostics/complete_audit_summary_no_exact_attempt002.json',
    'value_methods/official_scores.json': 'v31_value_selectors_20261008/reports/audit_scoring_no_exact_attempt002/summary.json',
    'value_methods/scorer_source_fingerprints.json': 'v31_value_selectors_20261008/scorer_source_fingerprints.json',
    'value_methods/query_gate_complete.json': 'v31_value_v2_refinement_20261010/final_complete.json',
    'value_methods/mass_gated_complete.json': 'v31_value_next_iteration_20261011/final_complete.json',
    'v2_calibration/equal_work_complete.json': 'v31_value_v2_equal_work_20261011/final_complete.json',
    'v2_calibration/equal_work_source_fingerprints.json': 'v31_value_v2_equal_work_20261011/receipts/source_fingerprints.json',
    'v2_calibration/threshold_curve_total.csv': 'v31_value_v2_equal_work_20261011/sweep/curve.csv',
    'v2_calibration/official_work_accounting.csv': 'v31_value_v2_equal_work_20261011/official_audit/work_accounting.csv',
    'v2_calibration/official_phase_accounting.csv': 'v31_value_v2_equal_work_20261011/official_audit/phase_accounting.csv',
    'v2_calibration/selected_thresholds.json': 'v31_value_v2_equal_work_20261011/matched_calibration/selected_thresholds.json',
    'v2_calibration/progress_axis_correction.json': 'v31_v2_local_threshold_sweep_20261014_attempt002/receipts/progress_axis_validation.json',
    'v2_calibration/local_sweep_curve.csv': 'v31_v2_local_threshold_sweep_20261014_attempt002/curve.csv',
    'v2_calibration/local_sweep_pause.json': 'v31_v2_local_threshold_sweep_20261014_attempt002/PAUSED_USER_REQUEST.json',
    'v2_calibration/local_sweep_verification.json': 'v31_v2_local_threshold_sweep_20261014_attempt002/receipts/controls_only_verification.json',
    'v2_calibration/local_sweep_protocol.json': 'v31_v2_local_threshold_sweep_20261014_attempt002/protocol.json',
    'cached_refresh/initial_panel_native.csv': 'v31_cached_refresh_20261012/tables/native_panel.csv',
    'cached_refresh/initial_panel_complete.json': 'v31_cached_refresh_20261012/final_complete.json',
    'cached_refresh/stable_panel_complete.json': 'v31_cached_refresh_stable_20261013/final_complete.json',
    'cached_refresh/stable_native_workload_results.json': 'v31_cached_refresh_stable_20261013/reports/native_workload_results.json',
    'cached_refresh/stable_parameters.json': 'v31_cached_refresh_stable_20261013/specs/parameters.json',
    'cached_refresh/factorial_complete.json': 'v31_refresh_factor_isolation_20261014/final_complete.json',
    'cached_refresh/factorial_summary.json': 'v31_refresh_factor_isolation_20261014/operator/attempt001/summary.json',
    'cached_refresh/full503_protocol.json': 'v31_full503_refresh_20261014/specs/protocol.json',
    'cached_refresh/full503_checkpoint.json': 'v31_full503_refresh_20261014/queue_state.json',
    'cached_refresh/full503_first_round_fairness.json': 'v31_full503_refresh_20261014/qualification/first_round_fairness.json',
}
DENIED = {
    'prompt', 'prompt_tokens', 'completion', 'gold', 'gold_answer', 'answer',
    'token_ids', 'input_ids', 'output_ids', 'input_token_ids', 'output_token_ids',
    'accepted_ids', 'raw_generation', 'raw_completion', 'password', 'api_key',
    'access_token', 'refresh_token', 'credential', 'credentials', 'secret',
}
HOST_PATH = re.compile(r'/(?:home|media|mnt|tmp|root|Users|workspace)/[^\s\]\)\}"\'<>`]*')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_text(value):
    # Repository references are useful; machine-specific prefixes are not.
    value = value.replace(str(ROOT) + '/', '')
    return HOST_PATH.sub(lambda match: '<private_path:' + digest(match.group().encode())[:12] + '>', value)


def sanitize(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key.lower() in DENIED:
                raise ValueError(f'refusing raw or sensitive payload field: {key}')
            result[safe_text(key)] = sanitize(item)
        return result
    if isinstance(value, list):
        return [sanitize(x) for x in value]
    return safe_text(value) if isinstance(value, str) else value


def export_csv(data, *, total_curve=False, work_accounting=False):
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    if not rows:
        raise ValueError('empty aggregate CSV')
    # Historical canvas-id summaries are superseded, not republished as valid.
    fields = [f for f in rows[0] if not (total_curve and
              ('early' in f.lower() or 'cumulative' in f.lower() or 'first_c' in f.lower()))]
    if work_accounting:
        fields = [f for f in fields if 'early' not in f.lower() and 'cumulative' not in f.lower()]
    if any(f.lower() in DENIED for f in fields):
        raise ValueError('raw field in aggregate CSV')
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows({k: safe_text(row[k]) for k in fields} for row in rows)
    return out.getvalue().encode()


def export(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'results') or output.exists():
        raise ValueError('choose a new output directory below results/')
    # Read and validate everything before creating the publication directory.
    prepared = []
    for name, relative in SOURCES.items():
        path = ROOT / 'results' / relative
        original = path.read_bytes()
        if path.suffix == '.json':
            payload = json.dumps(sanitize(json.loads(original)), indent=2, sort_keys=True, allow_nan=False).encode() + b'\n'
        else:
            payload = export_csv(original, total_curve=name.endswith('threshold_curve_total.csv'),
                                 work_accounting=name.endswith('official_work_accounting.csv'))
        prepared.append((name, relative, original, payload))
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z',
                                     '--', 'src', 'scripts', 'experiments', 'tests'], cwd=ROOT).decode().split('\0')
    code = {name: digest((ROOT / name).read_bytes()) for name in sorted(set(files))
            if name and (ROOT / name).is_file() and Path(name).suffix in {'.py', '.cu', '.h', '.cpp', '.sh'}}
    output.mkdir(parents=True)
    entries = []
    for name, relative, original, payload in prepared:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append(dict(published=name, source='results/' + relative,
                            original_sha256=digest(original), published_sha256=digest(payload),
                            derived=True))
    manifest = dict(schema='v31_research_week_publication_v1',
        captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_head_at_capture=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        scope='publication snapshot; does not resume or complete any campaign',
        files=entries, current_code_sha256=code,
        historical_canvas_axis='superseded by v2_calibration/progress_axis_correction.json; early columns omitted from historical curve/work export',
        full503_status='generation checkpoint only; accuracy unscored at capture', clean_timing='paused')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    (output / 'README.md').write_text(
        '# Research publication snapshot — 2026-10-15\n\n'
        'Read the three direction documents in `docs/research_week_20261015/`.\n'
        'These are sanitized derived aggregates. Original run bytes stay intact;\n'
        '`manifest.json` records their hashes and the exported hashes separately.\n'
        'Private host paths are redacted. Inputs, answers, token IDs, raw outputs,\n'
        'tensor snapshots, runtime caches and live worker files are excluded.\n\n'
        'The full503 snapshot is a generation checkpoint, not a completed score.\n'
        'The V2 local threshold campaign remains paused. Historical canvas-axis\n'
        'claims are superseded by the correction receipt; only total-work fields\n'
        'are exported from the old development curve and work table.\n'
        'Audit timings are instrumented measurements, not clean latency.\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = export(args.output)
    print(json.dumps(dict(files=len(manifest['files']), code_files=len(manifest['current_code_sha256']))))


if __name__ == '__main__':
    main()
