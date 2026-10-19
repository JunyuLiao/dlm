"""Explicitly use the frozen recovery reserve for cold-start qualification."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
from scripts import v31_failure_map_study as s


def recover(old, new):
    if new.exists():raise RuntimeError("new continuation directory required")
    state=s.base.read(old/'state.json')
    if state['status']!='budget_limited_partial' or state['phase']!='qualification':
        raise RuntimeError("only uncompleted, budget-limited qualification may use this addendum")
    s.verify(old);s.prior.require_idle()
    lock=(s.ROOT/'results/.v31_refresh_h100.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    new.mkdir(parents=True)
    (new/'private').mkdir()
    for name in ('execution','pools','cache'):
        (new/'private'/name).symlink_to((old/'private'/name).resolve(),target_is_directory=True)
    for rel in ('protocol.json','spec.md','model_source_fingerprints.json','private/cohort.json','private/warmup.json','receipts/source_fingerprints.json'):
        (new/rel).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(old/rel,new/rel)
    addendum=s.ROOT/'docs/V31_FAILURE_MAP_RECOVERY_20261019.md'
    shutil.copyfile(addendum,new/'recovery_addendum.md')
    s.base.write(new/'continuation.json',dict(parent=str(old.relative_to(s.ROOT)),
        parent_state_sha256=s.base.sha(old/'state.json'),native_source=s.base.read(old/'protocol.json')['source_commit'],
        recovery_script_sha256=s.base.sha(Path(__file__)),addendum_sha256=s.base.sha(addendum),
        recovery_cap_s=900,total_cap_s=14400,reason='first cold engine cost made qualification allocation insufficient'),True)
    markers={};values=[];completed=set()
    for marker in old.glob('qualification/*/attempt*/attempt_complete.json'):
        item=s.base.read(marker)
        for rel,digest in item['files_sha256'].items():
            if s.base.sha(marker.parent/rel)!=digest:raise RuntimeError('parent artifact drift')
        config=s.base.read(marker.parent/'config.json')
        completed.add((config['family'],config['k']))
        import os
        markers[os.path.relpath(marker,new)]=s.base.sha(marker)
        values.extend(s.base.read(marker.parent/'scored.json'))
    jobs=[('shared',k,k==8192) for k in s.KS]+[(f,8192,False) for f in s.REFS]
    terminal='failed'
    try:
        for family,k,fixed in jobs:
            if (family,k) in completed:continue
            rows=s.run_job(new,'recovery',family,k,s.base.read(new/'private/warmup.json'),[42],0,fixed,'audit')
            if rows is None:
                terminal='budget_limited_partial';break
            values.extend(rows)
            marker=new/f'recovery/{family}_k{k}_rep00_audit/attempt001/attempt_complete.json'
            markers[str(marker.relative_to(new))]=s.base.sha(marker)
            completed.add((family,k))
        if len(completed)==6:
            s.base.write(new/'qualification/aggregate.json',s.summarize(values),True)
            s.base.write(new/'qualification/phase_complete.json',dict(status='complete',records=len(values),engines=markers,
                protocol_sha256=s.base.sha(new/'protocol.json'),aggregate_sha256=s.base.sha(new/'qualification/aggregate.json'),
                continuation_sha256=s.base.sha(new/'continuation.json')),True)
            terminal='phase_stop'
    finally:
        ledger=s.base.read(new.parent/'gpu_budget.json')
        used=sum(r['reserved_s'] if r.get('elapsed_s') is None else r['elapsed_s'] for r in ledger['reservations'])
        s.base.write(new/'state.json',dict(status=terminal,phase='qualification',gpu_reservation_s=used,cap_s=s.CAP,
            completed_phases=['qualification'] if terminal=='phase_stop' else [],protocol_sha256=s.base.sha(new/'protocol.json'),older_campaigns_remain_paused=True))
        print(json.dumps(dict(event='qualification_recovery_terminal',status=terminal,completed_families=len(completed),gpu_reservation_s=used)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--old',type=Path,default=s.DEFAULT)
    ap.add_argument('--new',type=Path,default=s.DEFAULT.parent/'attempt002')
    a=ap.parse_args();recover(a.old,a.new)
