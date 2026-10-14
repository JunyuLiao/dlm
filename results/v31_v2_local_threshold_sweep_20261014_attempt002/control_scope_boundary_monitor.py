"""Honor the user's controls-only stop boundary without interrupting a GPU shard."""
import datetime
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent
PID = int((ROOT / 'coordinator.pid').read_text())
MONITOR_PID = int((ROOT / 'monitor.pid').read_text())
MARKER = ROOT / 'reports/longbench_v2/allkept_fa4/final_complete.json'


def process(pid):
    p = Path('/proc') / str(pid)
    if not p.exists():
        return None
    return {
        'state': (p / 'stat').read_text().rsplit(')', 1)[1].split()[0],
        'cmdline': (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(),
    }


def write(name, value):
    target = ROOT / name
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(target)


expected = process(PID)
if expected is None or 'scripts/v31_v2_local_threshold_sweep.py --stage run' not in expected['cmdline']:
    raise RuntimeError('Coordinator identity mismatch; no signals sent')
print(json.dumps({'event': 'boundary_armed', 'coordinator_pid': PID,
                  'stop_after': 'longbench_v2/allkept_fa4 complete scoring'}), flush=True)

while not MARKER.exists():
    current = process(PID)
    if current is None or current['state'] == 'Z':
        write('control_scope_boundary_failure.json', {'status': 'needs_controls_resume',
              'reason': 'coordinator stopped before all-kept control completion'})
        raise RuntimeError('Coordinator stopped before the authorized controls finished')
    time.sleep(0.25)

# The original queue waits for an idle GPU for at least 20 seconds before its
# next launch. Freeze its parent immediately on the completed scoring marker.
os.kill(PID, signal.SIGSTOP)
children_path = Path('/proc') / str(PID) / 'task' / str(PID) / 'children'
children = children_path.read_text().strip() if children_path.exists() else ''
if children:
    write('control_scope_boundary_failure.json', {'status': 'parent_stopped_needs_review',
          'reason': 'unexpected child at controls completion', 'children': children})
    raise RuntimeError('Parent remains stopped; unexpected child was not interrupted')

controls = ['current_v31_control', 'dense_full_fix51994', 'dense_piecewise', 'allkept_fa4']
for arm in controls:
    marker = ROOT / 'reports/longbench_v2' / arm / 'final_complete.json'
    m = json.loads(marker.read_text())
    if m['status'] != 'complete' or m['records'] != 503 or m['scored'] != 503:
        raise RuntimeError('Incomplete control; parent remains stopped')

threshold_attempts = list((ROOT / 'runs/longbench_v2').glob('v2_tau_*/shard*/attempt*/config.json'))
if threshold_attempts:
    write('control_scope_boundary_failure.json', {'status': 'parent_stopped_needs_review',
          'reason': 'unexpected LongBench threshold attempt',
          'attempts': [str(p.relative_to(ROOT)) for p in threshold_attempts]})
    raise RuntimeError('Unexpected threshold attempt; parent remains stopped')

# No GPU or scorer children remain. Terminate only this queue parent and its
# reporting monitor; all immutable generation/scoring artifacts are retained.
os.kill(PID, signal.SIGTERM)
os.kill(PID, signal.SIGCONT)
monitor = process(MONITOR_PID)
if monitor and 'scripts/v31_monitor_local_threshold_sweep.py' in monitor['cmdline']:
    os.kill(MONITOR_PID, signal.SIGTERM)
pause = {
    'status': 'paused_user_request',
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'controls_complete': True,
    'longbench_threshold_attempts': 0,
    'finer_threshold_attempts': 0,
    'pending': ['five initial LongBench V2 thresholds',
                'two finer thresholds across all three benchmarks'],
    'completed_runs_preserved': True,
    'resume_requires_user_instruction': True,
}
write('PAUSED_USER_REQUEST.json', pause)
write('controls_complete.json', {'status': 'complete', 'arms': controls,
      'records_per_arm': 503, 'scored_per_arm': 503,
      'remaining_campaign': 'paused_user_request'})
write('state.json', {'status': 'paused_user_request', 'controls_complete': True,
      'updated_unix': time.time(), 'pending': pause['pending']})
print(json.dumps({'event': 'controls_complete_campaign_paused', **pause}), flush=True)
