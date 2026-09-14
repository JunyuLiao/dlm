"""Count-weighted full-AIME26 and held-out reports from completed shards."""
import csv
import json
import math
from .run import TARGETS,_write,aggregate

def report(root):
    manifest=json.loads((root/'manifest.json').read_text());policy=json.loads((root/'policy.json').read_text())
    table=[];violations=[];counts={}
    for name,target in [('dense',None)]+[(f'blasst_s{int(t*100)}',t) for t in TARGETS]:
        records=[]
        for row in manifest:
            path=root/name/f'{row["source_id"]}.json'
            if not path.exists(): continue
            r=json.loads(path.read_text())
            if any(r[k]!=row[k] for k in ('id','prompt_hash','seed','generation_budget')): violations.append(str(path))
            if r['fingerprint']!=policy['fingerprint']: violations.append(f'{path}: fingerprint')
            if target is not None and r['log_scales']!=policy['targets'][str(target)]['log_scales']: violations.append(f'{path}: threshold')
            layer_ids={c['layer'] for c in r['per_layer']}
            if layer_ids!=set(range(30)): violations.append(f'{path}: layer coverage')
            if r['finite_calls']<=0: violations.append(f'{path}: missing finite-output checks')
            for c in r['calls']:
                if not 0<=c['skipped_tiles']<=c['eligible_tiles']: violations.append(f'{path}: invalid tile counts')
                if target is not None:
                    expected=math.exp(r['log_scales'][c['attention_type']])/c['valid_kv_length']
                    if not math.isclose(c['effective_blasst_lambda'],expected,rel_tol=1e-12): violations.append(f'{path}: incorrect effective lambda')
            records.append((row,r))
        counts[name]=len(records)
        for split in ('full30','heldout24','calibration6'):
            subset=[r for row,r in records if split=='full30' or (row['calibration']==(split=='calibration6'))]
            if not subset: continue
            stats=aggregate(c for r in subset for c in r['calls'])
            for kind in ('overall','global','local'):
                a=stats[kind]
                table.append(dict(condition=name,target=target,split=split,attention_type=kind,n=len(subset),
                    correct=sum(r['score']['correct'] for r in subset),accuracy=sum(r['score']['correct'] for r in subset)/len(subset),
                    eligible_tiles=a['eligible_tiles'],skipped_tiles=a['skipped_tiles'],sparsity=a['full_tile_sparsity'],
                    retained_attention_mass=a['retained_dense_attention_mass'],
                    skipped_row_votes=sum(c.get('skippable_row_votes',0) for r in subset for c in r['calls'] if kind=='overall' or c['attention_type']==kind),
                    valid_row_votes=sum(c.get('valid_row_votes',0) for r in subset for c in r['calls'] if kind=='overall' or c['attention_type']==kind),
                    lambda_min=min((c['effective_blasst_lambda'] for r in subset for c in r['calls'] if 'effective_blasst_lambda' in c and (kind=='overall' or c['attention_type']==kind)),default=None),
                    lambda_max=max((c['effective_blasst_lambda'] for r in subset for c in r['calls'] if 'effective_blasst_lambda' in c and (kind=='overall' or c['attention_type']==kind)),default=None)))
    for r in table:
        r['row_vote_sparsity']=r['skipped_row_votes']/r['valid_row_votes'] if r['valid_row_votes'] else None
    audit=dict(complete=all(v==30 for v in counts.values()) and not violations,counts=counts,violations=violations)
    _write(root/'audit.json',audit);_write(root/'summary.json',dict(audit=audit,results=table,policy=policy))
    if table:
        with (root/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# AIME26 aggressive BLASST','',f'Completion audit: {audit}','',
        'Full benchmark: 30 questions; six questions used for sparsity calibration. The separate 24-question result excludes calibration examples. Thresholds never use answer correctness. BF16, native seeded decoding (0.4–0.8 schedule), thinking off, seed42, 2048-token limit, 128×64 physical tiles, prefix+canvas eligible.', '',
        'The [paper calibration](https://arxiv.org/pdf/2512.12087) fits λL=α exp(γs), separately for local/global attention. This experiment extends the existing previous-maximum comparison to finite λ>1. The paper pseudocode uses an updated maximum; its literal λ>1 extension would skip all tiles. Each query’s first valid tile is retained here. Dense calibration fits are followed by scale bisection and sparse calibration verification/refinement; all traces are preserved. Inference uses λ=exp(selected_log_scale)/actual_valid_KV_length.', '',
        '| Split | Condition | Accuracy | Whole sparsity | Global | Local |','|---|---|---:|---:|---:|---:|']
    for r in table:
        if r['attention_type']!='overall': continue
        others={x['attention_type']:x for x in table if x['condition']==r['condition'] and x['split']==r['split']}
        lines.append(f"| {r['split']} | {r['condition']} | {r['correct']}/{r['n']} ({100*r['accuracy']:.2f}%) | {100*r['sparsity']:.2f}% | {100*others['global']['sparsity']:.2f}% | {100*others['local']['sparsity']:.2f}% |")
    lines+=['','Calibration target attainment:','']
    for target,p in policy['targets'].items():
        lines.append(f"- Target {float(target):.0%}: local {p['measured']['local']:.2%}, global {p['measured']['global']:.2%}; within 2pp: {p['within_two_points']}. Scales: {p['log_scales']}.")
    lines+=['','Full-benchmark thresholds and query-vote inflation:','',
        '| Condition | Layer type | λ range | Row-vote sparsity | Physical tile sparsity |',
        '|---|---|---:|---:|---:|']
    for r in table:
        if r['split']!='full30' or r['condition']=='dense' or r['attention_type']=='overall': continue
        votes='—' if r['row_vote_sparsity'] is None else f"{100*r['row_vote_sparsity']:.2f}%"
        lines.append(f"| {r['condition']} | {r['attention_type']} | {r['lambda_min']:.5g}–{r['lambda_max']:.5g} | {votes} | {100*r['sparsity']:.2f}% |")
    lines+=['','Sparsity is summed skipped tiles / summed eligible tiles. Row-vote counts in raw BLASST calls quantify KV inflation; retained mass is measured against dense softmax on the corresponding sparse trajectory. Finite tile populations can impose a ceiling even with λ>1. Full30 accuracy includes calibration prompts and must be interpreted alongside heldout24. No speedup claims.','']
    (root/'report.md').write_text('\n'.join(lines))
    print(json.dumps(audit),flush=True)
