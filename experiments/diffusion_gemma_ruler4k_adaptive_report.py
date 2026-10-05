"""Add dense-paired metrics to the completed adaptive RULER4K sweep."""
import csv, gzip, hashlib, json
from pathlib import Path

ROOT=Path('results/diffusion_gemma_ruler4k_adaptive_final_v21')
SOURCE=Path('results/diffusion_gemma_ruler4k_value_direction_s70_v19')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_shards(folder):
    out={}
    for p in Path(folder).glob('*.json'):
        x=json.loads(p.read_text());out[x['id']]=x
    return out
def records(x):
    src=x.get('records_source')
    if not src:return []
    return json.loads(gzip.decompress(Path(src['path']).read_bytes()))
def agreement(a,b):
    n=max(len(a),len(b)); return sum(i<len(a) and i<len(b) and a[i]==b[i] for i in range(n))/max(1,n)
def run():
    dense=load_shards(SOURCE/'dense/dense/shards')
    rows=[]
    for cond in ('adaptive_analytic_s50','adaptive_analytic_s75'):
        sparse=load_shards(ROOT/f'final/{cond}/shards')
        vals=[x for x in sparse.values() if x['id'] in dense]
        elig=skip=ge=gs=le=ls=mass=mass_rows=0.; agree=[]; delta=[]; proj=escal=0.
        for x in vals:
            d=dense[x['id']]; rr=records(x)
            elig+=sum(r['eligible'] for r in rr); skip+=sum(r['skipped'] for r in rr)
            ge+=sum(r['eligible'] for r in rr if r['attention_type']=='global');gs+=sum(r['skipped'] for r in rr if r['attention_type']=='global')
            le+=sum(r['eligible'] for r in rr if r['attention_type']=='local');ls+=sum(r['skipped'] for r in rr if r['attention_type']=='local')
            mass+=sum(r.get('mass_sum',0.) for r in rr);mass_rows+=sum(r.get('rows',0.) for r in rr)
            agree.append(agreement(x.get('completion_tokens',[]),d.get('completion_tokens',[])))
            delta.append(float(x.get('score',0.) or 0.)-float(d.get('score',0.) or 0.))
            for z in x.get('work_accounting',{}).get('adaptive_diagnostics',[]):
                proj+=z.get('projected_coordinate_evals',0.);escal+=z.get('escalated_tiles',0.)
        rows.append(dict(condition=cond,examples=len(vals),dense_accuracy=sum(float(dense[x['id']].get('score',0.) or 0.) for x in vals)/max(1,len(vals)),
            sparse_accuracy=sum(float(x.get('score',0.) or 0.) for x in vals)/max(1,len(vals)),
            accuracy_delta=sum(delta)/max(1,len(delta)),actual_sparsity=skip/max(1,elig),
            global_sparsity=gs/max(1,ge),local_sparsity=ls/max(1,le),
            retained_attention_mass=mass/max(1,mass_rows),token_agreement=sum(agree)/max(1,len(agree)),
            mean_projected_coordinate_evals=proj/max(1,len(vals)),mean_escalated_tiles=escal/max(1,len(vals))))
    (ROOT/'comparison.json').write_text(json.dumps(rows,indent=2))
    with (ROOT/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    p=ROOT/'report.md'; text=p.read_text(); text += '\n## Dense-paired metrics\n\n'
    text += '| condition | dense acc. | adaptive acc. | delta | actual sparsity | global | local | retained mass | token agreement | mean escalations |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for x in rows:
        text += f"| {x['condition']} | {100*x['dense_accuracy']:.2f}% | {100*x['sparse_accuracy']:.2f}% | {100*x['accuracy_delta']:+.2f}pp | {100*x['actual_sparsity']:.2f}% | {100*x['global_sparsity']:.2f}% | {100*x['local_sparsity']:.2f}% | {100*x['retained_attention_mass']:.2f}% | {100*x['token_agreement']:.2f}% | {x['mean_escalated_tiles']:.1f} |\n"
    text += '\nDense-paired token agreement counts equal positions and treats missing/extra positions as disagreements. Retained attention mass is the per-valid-row mass of the retained physical support.\n'
    p.write_text(text)
    audit=json.loads((ROOT/'audit.json').read_text())
    for q in (ROOT/'comparison.json',ROOT/'comparison.csv',ROOT/'report.md'): audit['artifacts'][q.name]=sha(q)
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True))
    (ROOT/'regeneration_verification.json').write_text(json.dumps(dict(
        passed=bool(audit.get('complete') and audit.get('completed')==audit.get('expected')),
        completed=audit.get('completed'), expected=audit.get('expected'),
        inference_performed=False, audit_sha256=sha(ROOT/'audit.json')), indent=2, sort_keys=True))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':run()
