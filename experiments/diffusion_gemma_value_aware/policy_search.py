"""Calibration bookkeeping; routing semantics remain in the frozen operator.

Treat a cap-one boundary independently by attention type. Never sacrifice a
feasible global target because the local target misses at its boundary.
"""
import math

KINDS=('local','global')


def is_one(entry, upper=80.):
    return bool(entry.get('cap_one') and (entry.get('log_threshold')==0.
        or entry.get('lambda_at_one') or entry.get('log_scale',-math.inf)>=upper))


def fixed_boundaries(observations, target, upper):
    result={}
    for kind in KINDS:
        boundary=[r for r in observations if is_one(r['policy'][kind],upper)]
        result[kind]=bool(boundary and boundary[-1]['achieved'][kind]<target-.02)
    return result


def next_policy(observations, target, original=False, upper=80.):
    current=observations[-1];updated={}
    for kind in KINDS:
        entry=dict(current['policy'][kind]);key='log_scale' if 'log_scale' in entry else 'log_threshold'
        achieved=current['achieved'][kind]
        if original and is_one(entry,upper) and achieved<target-.02:
            entry.update(unattainable=True,lambda_at_one=True)
            updated[kind]=entry;continue
        if abs(achieved-target)<=.02:
            updated[kind]=entry;continue
        seen=[(r['policy'][kind][key],r['achieved'][kind]) for r in observations]
        below=[x for x,s in seen if s<target];above=[x for x,s in seen if s>=target]
        if original and achieved<target-.02 and not above:
            value=upper
        elif below and above and max(below)<min(above):
            value=(max(below)+min(above))/2
        else:
            value=entry[key]+max(-2.,min(2.,8*(target-achieved)))
        entry[key]=max(-30.,min(upper if original else 80.,value))
        if original:entry['lambda_at_one']=entry[key]>=upper
        updated[kind]=entry
    return updated


def select_point(observations, target, original=False, upper=80.):
    fixed=fixed_boundaries(observations,target,upper) if original else dict.fromkeys(KINDS,False)
    # Only select a jointly measured setting that actually uses lambda1 for
    # every independently unattainable type. Other types remain free to fit.
    candidates=[r for r in observations if all(not fixed[k] or is_one(r['policy'][k],upper) for k in KINDS)]
    if not candidates:raise ValueError('missing joint verification of independent cap-one fallback')
    def error(r):return max((abs(r['achieved'][k]-target) for k in KINDS if not fixed[k]),default=0.)
    return min(candidates,key=error),fixed
