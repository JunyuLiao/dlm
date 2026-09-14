"""Preserve calibration-only screening stops in final provenance and reporting."""
import hashlib
import json


def evidence(root):
    index = root / 'screening_pruning.json'
    if not index.exists():
        return [], {}
    sources = {}

    def read(relative):
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('pruning evidence must stay in its experiment bundle')
        data = path.read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        return data

    initial = json.loads(read('screening_pruning.json'))
    decisions = [initial]
    for entry in initial.get('subsequent_decisions', []):
        if entry.get('heldout_used') is not False:
            raise ValueError('pruning cannot be described as calibration-only')
        decisions.append(json.loads(read(entry['source'])))
        if entry.get('findings'):
            read(entry['findings'])
    for decision in decisions:
        if decision.get('heldout_used') is not False:
            raise ValueError('pruning cannot be described as calibration-only')
        if not decision.get('pruned_candidate_name') or not decision.get('scope'):
            raise ValueError('pruning decision needs an explicit candidate and scope')
        if decision.get('reason_source'):
            read(decision['reason_source'])
        for benchmark in ('aime26', 'longbench'):
            point = decision.get(f'{benchmark}_s50')
            if point and point.get('policy_source'):
                read(point['policy_source'])
                if sources[str(root / point['policy_source'])] != point['policy_sha256']:
                    raise ValueError('pruning evidence policy changed after the decision')
    return decisions, sources


def report_rows(decisions):
    return [dict(candidate=d['pruned_candidate_name'],
                 decision_time=d.get('decision_time_utc'),
                 stopped_scope=d['scope'],
                 aime_cal_score=d.get('aime26_s50', {}).get('score'),
                 lb_cal_score=d.get('longbench_s50', {}).get('score'))
            for d in decisions]
