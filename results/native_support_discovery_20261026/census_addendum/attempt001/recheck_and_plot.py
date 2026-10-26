"""Independent native-commit audit and static figures; no new model generation."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;REPO=R.parents[3];C=R.parents[1]/'census/attempt001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def main():
    manifest=json.loads((C/'input_manifest.json').read_text());derived=json.loads((C/'analysis001/canvases.json').read_text());lookup={(r['study'],r['consumer'],r['id'],r['seed'],r['canvas']):r for r in derived};inputs={};count=0;tails=Counter();records=[]
    for m in manifest['records']:
        p=REPO/m['path'];assert sha(p)==m['sha256'];inputs[m['path']]=m['sha256'];raw=json.loads(p.read_text());canvas=0;length=0;ordinary=0
        consumer='FULL' if 'full' in m['arm'] else 'A'
        for row in raw['native_trace']:
            if row['commit']:
                d=lookup[(m['study'],consumer,m['id'],m['seed'],canvas)];assert d['m']==length;assert row['native_step']==0;canvas+=1;count+=1;length=0
            else:
                length+=1;ordinary+=1;assert row['native_step']==length
        assert ordinary==raw['N'] and canvas==raw['C'] and length==1;tails[m['study']]+=length
        records.append(dict(study=m['study'],id=m['id'],seed=m['seed'],consumer=consumer,committed_canvases=canvas,N=ordinary,uncommitted_tail=length))
    assert count==len(derived)==4953
    table=[]
    for consumer in ['FULL','A']:
        for rule in ['L1','L2']:
            xs=[r for r in derived if r['study']=='R' and r['correct'] and r['consumer']==consumer];counts=Counter((o['suffix'],'0' if o['U']==0 else '1' if o['U']==1 else '2+') for r in xs for o in r['opportunities'][rule]);table.append(dict(consumer=consumer,rule=rule,counts=[dict(suffix=s,U=u,count=n) for (s,u),n in sorted(counts.items())]))
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),sharey=True)
    for ax,consumer in zip(axes,['FULL','A']):
        for offset,suffix,color in [(-.18,'terminal','#2a788e'),(.18,'transient','#e79b45')]:
            vals=[sum(c['count'] for t in table if t['consumer']==consumer and t['rule']==rule for c in t['counts'] if c['suffix']==suffix and c['U']=='2+') for rule in ['L1','L2']]
            ax.bar([offset,1+offset],vals,width=.35,label=suffix,color=color)
        ax.set_xticks([0,1],['One-call trigger','Two-call trigger']);ax.set_title('Correct R controls: '+consumer);ax.set_ylabel('Opportunities with optimistic room ≥2');ax.legend()
    fig.suptitle('Large gaps are temporary stability, not a verified terminal suffix');fig.tight_layout();fig.savefig(R/'opportunity_room.png',dpi=180);fig.savefig(R/'opportunity_room.svg');plt.close(fig)
    pred=[]
    for task in ['aime26','humaneval']:
        for consumer in ['FULL','A']:
            rows=[r for r in derived if r['study']=='R' and r['correct'] and r['benchmark']==task and r['consumer']==consumer];counts=Counter()
            for row in rows:counts.update(row['predicate_counts'])
            pred.append(dict(task=task,consumer=consumer,predicate_counts=dict(counts),committed_calls=sum(r['m'] for r in rows),terminal_confidence_only=sum(r['verified_confidence_only_calls'] for r in rows)))
    save(R/'verification.json',dict(passed=True,source_sha256=sha(Path(__file__)),census_protocol_sha256=sha(C/'protocol.json'),inputs=inputs,requests=368,committed_canvases=count,native_uncommitted_tails=dict(tails),substituted_CPU_emission_intervals=False,model_calls=0));save(R/'opportunities.json',table);save(R/'predicates.json',pred);save(R/'requests.json',records);save(R/'final_complete.json',dict(passed=True,verification_sha256=sha(R/'verification.json'),census_protocol_sha256=sha(C/'protocol.json')))
if __name__=='__main__':main()
