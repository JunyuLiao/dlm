"""Independent raw-shard arithmetic checks and mechanism report."""
import json
import hashlib
from collections import defaultdict
from pathlib import Path
from .lambda1_probe import BASE,OUT
from .runner import _write

def main():
    summary=json.loads((BASE/'summary.json').read_text())
    table=[];checks=[]
    for benchmark in ('ruler16k','longbench','aime24','livecodebench_v6'):
        saved=next(r for r in summary['results'] if r['benchmark']==benchmark and r['condition']=='blasst_length_aware_s90')
        totals=defaultdict(lambda:defaultdict(float));count=0;identical75=True
        for path in sorted((BASE/'conditions/blasst_length_aware_s90/shards').glob('*.json')):
            r=json.loads(path.read_text())
            if r['benchmark']!=benchmark: continue
            count+=1
            old=json.loads((BASE/'conditions/blasst_length_aware_s75/shards'/path.name).read_text())
            identical75 &= old['completion_tokens']==r['completion_tokens'] and old['calls']==r['calls']
            # Separate metadata varies with lambda policy even where decisions agree.
            for c in r['calls']:
                assert c['effective_blasst_lambda']==1.
                t=totals[c['attention_type']]
                for k in ('eligible_tiles','skipped_tiles'): t[k]+=c[k]
                t['mass_sum']+=c['retained_dense_attention_mass']*c['retained_attention_mass_rows']
                t['mass_rows']+=c['retained_attention_mass_rows']
        for k in ('eligible_tiles','skipped_tiles','mass_sum','mass_rows'):
            totals['overall'][k]=totals['global'][k]+totals['local'][k]
        for kind,t in totals.items():
            assert t['eligible_tiles']==saved['actual'][kind]['eligible_tiles']
            assert t['skipped_tiles']==saved['actual'][kind]['skipped_tiles']
            assert abs(t['mass_sum']/t['mass_rows']-saved['actual'][kind]['retained_dense_attention_mass'])<1e-10
        assert count==saved['n']
        checks.append(dict(benchmark=benchmark,samples=count,all_effective_lambdas_one=True,raw_counts_and_mass_match_summary=True,identical_s75_s90_calls_and_tokens=identical75))
        table.append(dict(benchmark=benchmark,score=saved['accuracy'],dense_score=saved['dense_accuracy'],dense_mass=saved['retained_attention_mass'],
            sparse_mass=saved['sparse_trajectory_retained_mass'],counts=dict(totals)))
    replay=json.loads((OUT/'replay_audit.json').read_text());assert replay['passed']
    _write(OUT/'independent_audit.json',dict(passed=True,checks=checks,replay=replay,
        original_audit=json.loads((BASE/'audit.json').read_text()),full_run=table))

if __name__=='__main__': main()
