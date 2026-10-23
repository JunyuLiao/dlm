"""Verify immutable diagnostic leaves and inventory sanitized publication files."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

REPO = Path('/home/exouser/ljy/dlm')
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ATTEMPT = ROOT.parent

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')

def main():
    previous = read(ROOT/'report003/integrity_receipt.json')
    phase0 = read(ATTEMPT/'execution001/phase0_audit.json')
    replay = ROOT/'native_replay001'
    p = read(replay/'protocol.json')
    receipt = read(replay/'smoke/receipt.json')
    tests = ET.parse(OUT/'cpu_tests.xml').getroot().find('testsuite').attrib
    gpu = read(OUT/'gpu_status.json')
    checks = dict(
        previous26_checks_passed=previous['passed'] and all(previous['checks'].values()),
        previous_artifacts_unchanged=all(sha(ROOT/name)==h for name,h in previous['artifact_sha256'].items()),
        replay_frozen_sources_match=all(sha(Path(name))==h for name,h in p['sources'].items()),
        replay_passed=receipt['passed'] and receipt['logical_native_replay_equal'],
        replay_no_extra_forwards=receipt['zero_extra_forwards'],
        replay_rng_and_sampler_parity=receipt['final_native_rng_equal'] and receipt['per_call_native_state_hashes_equal'],
        replay_marker_protocol_matches=read(replay/'smoke/final_complete.json')['protocol_sha256']==sha(replay/'protocol.json'),
        cpu24_passed=tests['tests']=='24' and tests['failures']==tests['errors']=='0',
        gpu_idle=gpu['idle'] and all(x['returncode']==0 for x in gpu['checks']),
        original_protocol_unchanged=sha(ATTEMPT/'protocol.json')==phase0['existing_attempt_protocol_sha256'],
        original_receipt_unchanged=sha(ATTEMPT/'receipt.json')==phase0['existing_attempt_receipt_sha256'],
        original_identity_unchanged=sha(ATTEMPT/'manifest_identity.json')==p['original_identity_sha256'],
        ledger_unchanged=sha(REPO/'docs/RESULTS_LEDGER.md')==phase0['initial_sha256']['docs/RESULTS_LEDGER.md'],
        agents_unchanged=sha(REPO/'AGENTS.md')==phase0['initial_sha256']['AGENTS.md'])
    native = [read(x) for x in sorted((replay/'smoke').glob('request*/record.json'))]
    measured = [r for r in native if r['label']!='warm']
    checks['replay_all5_native_counters_match'] = len(measured)==5 and all(
        r['output_tokens']==275 and r['sampler_calls']==12 and
        r['model_execute_calls']==r['engine_steps']==13 and
        r['new_cuda_captures']==r['jit_event_count']==0 for r in measured)
    fingerprints = [read(x) for x in sorted((replay/'smoke').glob('request*/replay_state.json'))]
    checks['replay_all30_layers_all12_boundaries'] = len(fingerprints)==3 and all(
        len(rows)==12 and all(len(row['layers'])==30 for row in rows) for rows in fingerprints)
    snapshots = ATTEMPT/'execution001/private/initial_changes'
    for name in ('HANDOFF.md','docs/DECISIONS.md'):
        checks['prior_'+name.replace('/','_')+'_preserved'] = (snapshots/name).read_text() in (REPO/name).read_text()
    reservations = previous['reservations'] + [dict(attempt='native_replay001',
        **read(replay/'exit_receipt.json'))]
    total = previous['reserved_wall_seconds']+read(replay/'exit_receipt.json')['seconds']
    if not all(checks.values()):
        raise RuntimeError('closeout checks failed: '+str(checks))
    publication = []
    excluded = {'private','cache','runtime','__pycache__'}
    allowed = {'.json','.md','.py','.xml','.png','.pdf','.txt'}
    for file in sorted(ATTEMPT.rglob('*')):
        if not file.is_file() or file.is_symlink() or any(x in file.parts for x in excluded):
            continue
        if 'research_sources' in file.parts and file.name!='retrieval_receipt.json':
            continue
        if file.suffix not in allowed or file.parent==OUT:
            continue
        publication.append(file)
    scaffold = [REPO/name for name in (
        'experiments/diffusion_gemma_sparse_attention_research/trajectory_dynamics.py',
        'experiments/diffusion_gemma_sparse_attention_research/dynamics_screen.py',
        'experiments/diffusion_gemma_sparse_attention_research/native_observer.py',
        'scripts/run_denoising_dynamics_census.py',
        'tests/test_trajectory_dynamics.py','tests/test_dynamics_screen_protocol.py')]
    public = publication+scaffold
    forbidden = {'prompt','expected','gold','answer','completion','token_ids','prompt_token_ids',
        'generated_text','credentials','api_key','password'}
    hits = []
    counts = dict(json_files=0,scalar_prompt_token_count_fields=0)
    def inspect(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden:
                    hits.append(dict(path=str(path.relative_to(REPO)), key=key))
                if key=='prompt_tokens':
                    if isinstance(item,int):
                        counts['scalar_prompt_token_count_fields']+=1
                    else:
                        hits.append(dict(path=str(path.relative_to(REPO)),key=key))
                inspect(item,path)
        elif isinstance(value,list):
            for item in value:
                inspect(item,path)
    for file in publication:
        if file.suffix=='.json':
            counts['json_files']+=1
            inspect(read(file),file)
    if hits:
        raise RuntimeError('publication privacy field audit failed: '+str(hits))
    write(OUT/'privacy_audit.json',dict(passed=True,forbidden_field_hits=hits,counts=counts,
        scope='Structured public JSON only; source and summary text manually inspected. Private inputs, logs, runtime caches and downloaded papers excluded.'))
    result = dict(schema='denoising_diagnostic_closeout_v1',checks=checks,passed=True,
        verified_unix=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        full30_complete=False,eligible_dense_complete=True,native_selector_promotion=False,
        full_state_fork_restore_validated=False,replay_physical_page_layout_equal=receipt['physical_page_layout_equal'],
        reservations=reservations,reserved_wall_seconds=total,
        resource_definition=previous['resource_definition'],cpu_tests=tests,
        source_hashes={str(x.relative_to(REPO)):sha(x) for x in scaffold})
    write(OUT/'verification.json',result)
    write(OUT/'publication_files.json',dict(files=[str(x.relative_to(REPO)) for x in public],
        inherited_untracked_scaffolding=[str(x.relative_to(REPO)) for x in scaffold],
        note='Scaffolding was present before this task; covariance axes and full Q/K/V differential diagnostics were corrected/extended. Study results and code are newly published. Pre-existing tracked edits are staged separately.'))
    inventory = [dict(path=str(x.relative_to(REPO)),bytes=x.stat().st_size,sha256=sha(x)) for x in public]
    write(OUT/'artifact_inventory.json',dict(files=inventory,
        excluded='Private data, runtime/cache/logs and downloaded source documents remain local; this inventory excludes its own closeout files.'))
    print(json.dumps(dict(passed=True,checks=len(checks),cpu_tests=tests['tests'],
        publication_files=len(public),reserved_wall_seconds=total)))

if __name__=='__main__':
    main()
