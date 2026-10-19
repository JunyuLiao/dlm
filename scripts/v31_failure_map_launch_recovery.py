"""Preserve a zero-record launch-path failure and continue using absolute paths."""
import os
from pathlib import Path
import shutil
from scripts import v31_failure_map_study as s


def prepare(old, new):
    old=old.resolve();new=new.resolve()
    if new.exists() or old.parent!=new.parent:raise RuntimeError('new sibling attempt required')
    state=s.base.read(old/'state.json')
    if state['status']!='failed' or state['phase']!='screen':raise RuntimeError('unexpected parent status')
    receipts=list(old.glob('screen/*/attempt*/receipt.json'))
    if len(receipts)!=1:raise RuntimeError('only the first zero-record launch is eligible')
    receipt=s.base.read(receipts[0])
    if receipt['records'] or receipt['private_records'] or receipt['returncode']!=2 or receipt['timed_out']:
        raise RuntimeError('not a zero-record argument-path failure')
    if "can't open file" not in (receipts[0].parent/'private/worker.log').read_text():raise RuntimeError('unexpected failure')
    s.verify(old);s.prior.require_idle();s.load_phase(old,'qualification')
    new.mkdir();(new/'private').mkdir()
    for name in ('execution','pools','cache'):
        (new/'private'/name).symlink_to((old/'private'/name).resolve(),target_is_directory=True)
    for rel in ('protocol.json','spec.md','model_source_fingerprints.json','private/cohort.json','private/warmup.json',
                'receipts/source_fingerprints.json','qualification/aggregate.json'):
        (new/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/rel,new/rel)
    done=s.base.read(old/'qualification/phase_complete.json')
    done['engines']={os.path.relpath((old/name).resolve(),new):digest for name,digest in done['engines'].items()}
    done['inherited_parent_phase_sha256']=s.base.sha(old/'qualification/phase_complete.json')
    done.pop('continuation_sha256',None)
    s.base.write(new/'qualification/phase_complete.json',done,True)
    ledger=s.base.read(new.parent/'gpu_budget.json')
    s.base.write(new/'continuation.json',dict(parent=str(old.relative_to(s.ROOT)),
        parent_state_sha256=s.base.sha(old/'state.json'),reason='relative CLI study path caused interpreter path duplication before model import',
        correction='absolute study paths only; unchanged execution source, native protocol and inputs',
        preparation_source_sha256=s.base.sha(Path(__file__)),charge_start_index=len(ledger['reservations']),
        total_cap_s=14400,prior_failed_launch_charge_s=receipt['gpu_reservation_s']),True)
    s.base.write(new/'state.json',dict(status='prepared',phase='screen',completed_phases=['qualification'],
        protocol_sha256=s.base.sha(new/'protocol.json'),older_campaigns_remain_paused=True))
    print(new)


if __name__=='__main__':
    prepare(s.DEFAULT.parent/'attempt002',s.DEFAULT.parent/'attempt003')
