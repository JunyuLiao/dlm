"""Set the explicit pause after the last MATH worker has passed launch gating.

The current job finishes and is validated/scored by its unchanged coordinator.
The next (LongBench) job observes the marker before any launch or GPU query.
"""
import datetime
import hashlib
import json
from pathlib import Path
import time

control=Path(__file__).resolve().parent
request=json.loads((control/'request.json').read_text())
root=control.parents[2]
study=root/request['study']
assert hashlib.sha256((study/'protocol.json').read_bytes()).hexdigest()==request['protocol_sha256']
while True:
    state=json.loads((study/'state.json').read_text())
    trigger=request['trigger']
    if all(state.get(k)==trigger[k] for k in ['stage','suite','arm','pass_name']) and '/shard004/' in state.get('active_attempt',''):
        leaf=study/state['active_attempt']
        if (leaf/'config.json').exists():
            receipt=dict(status='pause_requested_after_current_final_math_worker',
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                reason=request['instruction'],current_attempt=state['active_attempt'],
                protocol_sha256=request['protocol_sha256'],no_worker_signals=True)
            marker=study/'PAUSED_USER_REQUEST.json'
            if not marker.exists():marker.write_text(json.dumps(receipt,indent=2)+'\n')
            (control/'boundary_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            break
    if state.get('suite')=='longbench_v2' and state.get('stage')=='runs':
        raise RuntimeError('LongBench reached before boundary; refuse silent continuation')
    if state.get('status')=='failed':
        raise RuntimeError('campaign failed before requested MATH boundary')
    time.sleep(10)
