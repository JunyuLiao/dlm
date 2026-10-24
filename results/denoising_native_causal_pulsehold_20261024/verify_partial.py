"""Verify an explicitly partial handoff; never manufacture a stage completion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    repo = root.parents[2]
    out = root / 'closeout001'
    protocol = read(root / 'protocol.json')
    summary = read(root / 'analysis_partial002/summary.json')
    analysis = read(root / 'analysis_partial002/readout_complete.json')
    exit_receipt = read(root / 'exit_receipt.json')
    launch = read(root / 'launch_receipt.json')
    checks = dict(
        sources=all(sha(f) == h for f, h in protocol['sources'].items()),
        scorer_sources=all(sha(f) == h for f, h in protocol['scorer_sources'].items()),
        protocol=sha(root / 'protocol.json') == read(root / 'source_fingerprints.json')['protocol_sha256'],
        private_manifest=sha(root / 'private/tokenized_manifest.json') == protocol['private_manifest_sha256'],
        public_manifest=sha(root / 'manifest_identity.json') == protocol['public_manifest_sha256'],
        runtime=sha(root / 'runtime_fingerprints.json') == protocol['runtime_fingerprints_sha256'],
        model_receipt=sha(root / 'model_fingerprints.json') == protocol['model_fingerprints_sha256'],
        model_16_files=read(root / 'model_fingerprints.json')['passed'] and len(read(root / 'model_fingerprints.json')['files']) == 16,
        analysis_summary=sha(root / 'analysis_partial002/summary.json') == analysis['summary_sha256'],
        analysis_source=sha(root.parent / 'analyze_partial.py') == analysis['analysis_source_sha256'],
        analysis_inputs=all(sha(root / f) == h for f, h in read(root / 'analysis_partial002/input_hashes.json').items()),
        original_nine_long_ids_blocked=len(protocol['blocked_ids']) == 9,
        stage_explicitly_partial=exit_receipt['returncode'] == 1 and not analysis['stage_complete'] and not (root / 'execution001/final_complete.json').exists(),
        no_task_or_promotion=summary['official_task_outcomes'] is None and not summary['method_promoted'],
        no_factorial_claim=summary['selected_factorial_branches'] == 0 and not summary['hypothesis_tested'],
        qualification_stop=read(out / 'qualification_failure.json')['message'] == 'full-request restore parity',
        allocation_bound=exit_receipt['allocated_worker_seconds'] + exit_receipt['previous_allocated_seconds'] <= 7200,
        forward_bound=exit_receipt['forwards'] + exit_receipt['previous_forwards'] <= 10000,
        gpu_idle=read(out / 'gpu_status.json')['idle'],
        CPU_derivative=read(out / 'response_finite_difference002.json')['passed'],
    )
    own_frozen = [Path(f) for f in protocol['sources'] if str(root) in f]
    checkpoint_checks = {}
    for f in own_frozen + [root / 'protocol.json']:
        name = str(f.relative_to(repo))
        p = subprocess.run(['git', 'show', launch['source_checkpoint'] + ':' + name], cwd=repo, capture_output=True)
        checkpoint_checks[name] = p.returncode == 0 and hashlib.sha256(p.stdout).hexdigest() == sha(f)
    checks['source_checkpoint'] = all(checkpoint_checks.values())
    for f in ['cpu004.xml', 'cpu005.xml']:
        suite = ET.parse(root / 'tests' / f).getroot().find('testsuite').attrib
        checks[f] = suite['tests'] == '24' and suite['failures'] == suite['errors'] == suite['skipped'] == '0'
    q = read(root / 'analysis_partial002/qualification_summary.json')
    checks['two_AIME_only_certificates'] = len(q['completed_anchors']) == 2 and all(x['passed'] and not x['full_request_qualified'] and x['post_perturbation_controls_exact'] for x in q['completed_anchors'])
    checks['actual_native_winners'] = q['measurement_totals_over_repeated_controls']['native_winner_mismatches'] == 0
    # Historical sources and immutable readout still agree after this attempt.
    prior = repo / 'results/denoising_native_causal_20261024/attempt006'
    prior_protocol = read(prior / 'protocol.json')
    checks['prior_attempt006_sources'] = all(sha(f) == h for f, h in prior_protocol['sources'].items())
    checks['prior_attempt006_analysis_inputs'] = all(sha(prior / f) == h for f, h in read(prior / 'analysis001/input_hashes.json').items())
    initial = read(root.parent / 'attempt001/starting_identity.json')
    checks['AGENTS_preserved'] = sha(repo / 'AGENTS.md') == initial['preexisting_document_hashes']['AGENTS.md']
    checks['ledger_unchanged'] = sha(repo / 'docs/RESULTS_LEDGER.md') == initial['preexisting_document_hashes']['docs/RESULTS_LEDGER.md']
    checks['HANDOFF_original_body_preserved'] = (repo / 'HANDOFF.md').read_text().endswith((root / 'private/pre_closeout_HANDOFF.md').read_text())
    checks['DECISIONS_original_body_preserved'] = (repo / 'docs/DECISIONS.md').read_text().startswith((root / 'private/pre_closeout_DECISIONS.md').read_text().rstrip())
    # Exclude private/cache/runtime trees before traversing the publication set.
    public = []
    for folder, dirs, files in os.walk(root.parent):
        dirs[:] = [d for d in dirs if d not in {'private', 'cache', 'runtime', '__pycache__'}]
        for name in files:
            if name not in {'cpu001.xml', 'cpu002.xml'}:
                public.append(Path(folder) / name)
    forbidden = {'prompt', 'gold', 'answer', 'completion', 'token_ids', 'prompt_token_ids',
                 'generated_text', 'credentials', 'api_key', 'password'}
    hits = []

    def inspect(value, name):
        if isinstance(value, dict):
            for key, v in value.items():
                if key in forbidden:
                    hits.append([name, key])
                inspect(v, name)
        elif isinstance(value, list):
            for v in value:
                inspect(v, name)

    for p in public:
        if p.suffix == '.json':
            inspect(read(p), str(p.relative_to(repo)))
    checks['public_JSON_forbidden_fields_absent'] = not hits
    checks['private_manifest_ignored'] = subprocess.run(['git', 'check-ignore', str(root / 'private/tokenized_manifest.json')], cwd=repo, capture_output=True).returncode == 0
    if not all(checks.values()):
        raise RuntimeError(str({k: v for k, v in checks.items() if not v}))
    generated_fix = root / 'runtime/fa4_local_fix/692156e78b190853/flash_fwd_sm90.py'
    write(out / 'derived_runtime_source.json', dict(file=str(generated_fix.relative_to(root)), sha256=sha(generated_fix), generated_from_pinned_fix_function=True))
    write(out / 'privacy_audit.json', dict(passed=True, forbidden_JSON_fields=sorted(forbidden), hits=hits, private_manifest_ignored=True, scope='public JSON field audit plus explicit exclusion of private tensors/logs; source code contains input-field names, never their contents'))
    write(out / 'verification.json', dict(closeout_checks_passed=True, checks=checks,
        stage_complete=False, source_checkpoint=launch['source_checkpoint'],
        source_checkpoint_file_checks=checkpoint_checks, protocol_sha256=sha(root / 'protocol.json'),
        verified_unix=time.time(), qualification_scope='Two AIME first-canvas anchors only',
        full_request_qualification_passed=False, method_promoted=False, official_task_scoring=False))
    write(out / 'publication_files.json', dict(files=[dict(path=str(p.relative_to(repo)), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(public)],
        scope='Public files present before this verifier wrote its own closeout artifacts; private/cache/runtime and failed cpu001/002 XML excluded'))
    write(out / 'partial_handoff_complete.json', dict(closeout_checks_passed=True, experimental_stage_complete=False,
        status='partial_development_qualification_stop', protocol_sha256=sha(root / 'protocol.json'),
        verification_sha256=sha(out / 'verification.json'), analysis_sha256=sha(root / 'analysis_partial002/readout_complete.json'),
        method_promoted=False, confirmation_complete=False, task_scoring_complete=False))
    print(json.dumps(dict(closeout_checks_passed=True, checks=len(checks), stage_complete=False,
        first_canvas_certificates=2, full_request_certificates=0, model_forwards=exit_receipt['forwards'])))


if __name__ == '__main__':
    main()
