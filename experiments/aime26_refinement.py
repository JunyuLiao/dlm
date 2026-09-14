"""Calibration-only follow-up; preserves the original experiment and its shards."""
import json
import os
import subprocess
import time
from pathlib import Path

from experiments.diffusion_gemma_aime26.run import (
    ROOT, MODEL, REVISION, cached, create_adapter, aggregate, _write, sha,
)
from experiments.diffusion_gemma_aime26.report import report

OUT = ROOT / 'refined'


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.symlink_to(source.resolve())
    elif destination.resolve() != source.resolve():
        raise RuntimeError(f'Conflicting existing destination: {destination}')


def execute():
    OUT.mkdir(parents=True, exist_ok=True)
    # Queue rather than compete with the still-running initial evaluation.
    while True:
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                              '--format=csv,noheader'], capture_output=True, text=True)
        if gpu.returncode == 0 and not gpu.stdout.strip():
            break
        print('Waiting for GPU; next check in 15 minutes', flush=True)
        time.sleep(900)
    base = json.loads((ROOT / 'policy.json').read_text())
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    calibration = [r for r in manifest if r['calibration']]
    fingerprint = base['fingerprint']
    provenance = dict(base_fingerprint=fingerprint, code_sha256=sha(Path(__file__).read_bytes()),
                      calibration_ids=[r['source_id'] for r in calibration],
                      selection='minimum maximum local/global sparsity error; never accuracy',
                      reason='Refine 75% near miss and test moderate 90% scales before accepting saturation')
    provenance_path = OUT / 'refinement_provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        previous = json.loads(provenance_path.read_text())
        old_hash = '8dc6f095bb61b5685bd76446b49afc6cec717360e6753c63983617656093379c'
        if previous['code_sha256'] != old_hash or any(previous[k] != provenance[k] for k in provenance if k != 'code_sha256'):
            raise RuntimeError('Refinement protocol changed')
        _write(OUT / 'serialization_fix_provenance.json', dict(previous=previous, current=provenance,
            change='Copy selected record before attaching trace to avoid circular JSON; reuse unchanged original final conditions. No inference or candidate selection changes.'))
    _write(provenance_path, provenance)
    adapter = create_adapter('diffusion_gemma', MODEL, device='cuda',
                             precision='bfloat16', revision=REVISION).load()
    if not (OUT / 'policy.json').exists():
        policy = json.loads(json.dumps(base))
        # Fixed calibration-only candidates. Log scales deploy as lambda=exp(x)/L.
        candidates = {
            '0.75': [(8.54, 9.047452107806643), (8.55, 9.03), (8.53, 9.03),
                     (8.54, 9.06), (8.52, 9.02), (8.56, 9.02), (8.5, 9.04)],
            '0.9': [(10.831796002537907, 11.), (11., 11.5), (11.5, 11.5),
                    (12., 12.), (13., 13.), (14., 14.), (10.5, 10.5), (12., 10.5)],
        }
        for target, points in candidates.items():
            best = dict(base['targets'][target])
            best['source_directory'] = str(ROOT / 'verification' / f's{int(float(target)*100)}' / f"round{best['round']}")
            trace = []
            for index, (global_scale, local_scale) in enumerate(points):
                scales = {'global': global_scale, 'local': local_scale}
                directory = OUT / 'calibration' / target / str(index)
                outputs = [cached(adapter, row, directory / f"{row['source_id']}.json",
                                  fingerprint, scales)[0] for row in calibration]
                stats = aggregate(c for r in outputs for c in r['calls'])
                measured = {k: stats[k]['full_tile_sparsity'] for k in ('global', 'local')}
                error = max(abs(v-float(target)) for v in measured.values())
                result = dict(log_scales=scales, measured=measured, error=error,
                              within_two_points=error <= .02, source_directory=str(directory))
                trace.append(result)
                if error < best['error']:
                    best = result
                _write(OUT / f'calibration_trace_{target}.json', trace)
                print('Calibration', target, result, flush=True)
                if error <= .02:
                    break
            best = dict(best)
            best['refinement_trace'] = trace
            policy['targets'][target] = best
        policy['refinement_provenance'] = provenance
        _write(OUT / 'policy.json', policy)
    policy = json.loads((OUT / 'policy.json').read_text())
    for filename in ('manifest.json', 'setup.json', 'experiment.json', 'smoke.json', 'model_runtime.json'):
        link(ROOT / filename, OUT / filename)
    for condition in ('dense', 'blasst_s25', 'blasst_s50'):
        link(ROOT / condition, OUT / condition)
    failures = []
    for target in ('0.75', '0.9'):
        selected = policy['targets'][target]
        condition = f'blasst_s{int(float(target)*100)}'
        if selected['log_scales'] == base['targets'][target]['log_scales']:
            link(ROOT / condition, OUT / condition)
            continue
        for row in manifest:
            destination = OUT / condition / f"{row['source_id']}.json"
            try:
                if row['calibration']:
                    link(Path(selected['source_directory']) / destination.name, destination)
                else:
                    cached(adapter, row, destination, fingerprint, selected['log_scales'])
            except Exception as error:
                failures.append(dict(condition=condition, id=row['id'], error=repr(error)))
                _write(OUT / 'failures.json', failures)
    _write(OUT / 'status.json', dict(finished=True, failures=failures))
    report(OUT)


if __name__ == '__main__':
    execute()
