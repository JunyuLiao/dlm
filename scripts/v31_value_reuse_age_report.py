"""CPU-only, hash-bound age analysis of complete operator diagnostics."""
from collections import defaultdict
from pathlib import Path
import json
from scripts import v31_value_decision_replay as io


def report(root):
    paths={}; all_rows=[]
    for family in ('whole_layer','reuse_guard'):
        study=root/family
        marker=io.read(study/'operator_complete.json')
        assert marker['summary_sha256']==io.sha(study/'summary.json')
        assert marker['protocol_sha256']==io.sha(study/'operator_protocol.json')
        for group in io.read(study/'operator_protocol.json')['groups']:
            candidates=sorted((study/'operator'/group).glob('attempt*/complete.json'))
            assert candidates
            done=candidates[-1];d=io.read(done);p=done.parent/'records.json'
            assert d['records_sha256']==io.sha(p)
            assert d['protocol_sha256']==marker['protocol_sha256']
            paths[str(p.relative_to(root))]=io.sha(p)
            all_rows.extend(dict(r,family=family) for r in io.read(p))
    control={(r['family'],r['request'],r['canvas'],r['call'],r['layer']):r
             for r in all_rows if r['arm']=='mass_anchor'}
    buckets=defaultdict(list)
    for r in all_rows:
        if r['age']<=0 or r['prefix_tiles']<=r['budget'] or r['arm']=='mass_anchor':continue
        # Frame-weighted descriptive summaries; not independent samples or CIs.
        age=str(r['age']) if r['age']<=4 else ('5-8' if r['age']<=8 else '9+')
        for level,value in [('reuse_age',age),('layer',str(r['layer']))]:
            buckets[r['dataset'],r['family'],r['arm'],level,value].append(r)
    results=[]
    for (dataset,family,arm,level,value),rows in sorted(buckets.items()):
        refs=[control[family,r['request'],r['canvas'],r['call'],r['layer']] for r in rows]
        results.append(dict(dataset=dataset,family=family,arm=arm,dimension=level,value=value,
            frames=len(rows),trajectories=len({r['request'] for r in rows}),
            mean_error_ratio=sum(r['mean_post_o_error'] for r in rows)/sum(c['mean_post_o_error'] for c in refs),
            maxrow_harm_frames=sum(r['max_post_o_error']>1.02*c['max_post_o_error']+1e-6 for r,c in zip(rows,refs)),
            mean_error_harm_frames=sum(r['mean_post_o_error']>c['mean_post_o_error']+1e-6 for r,c in zip(rows,refs))))
    out=root/'reuse_age_analysis';out.mkdir(exist_ok=True)
    io.write(out/'summary.json',dict(metrics=results,source_files_sha256=paths,
        uncertainty='descriptive correlated frames; two LongBench trajectories and one sparse AIME trajectory, no independent-frame CI',
        held_out=False,timing='paused'),True)
    lines=['# Reuse age and whole-layer error','',
        'Ratios compare sums of mean row post-O errors on identical frames. '
        'Frames are correlated and are not independent accuracy samples. '
        'LongBench has two source trajectories; sparse AIME has one late trajectory '
        'and a counterfactual K=27 budget. A harm is a >2% maximum-row error increase.', '',
        '| Workload | Method | Reuse age | Frames | Error ratio | Max-row harms |',
        '|---|---|---:|---:|---:|---:|']
    for r in results:
        if r['dimension']=='reuse_age':
            lines.append(f"| {r['dataset']} | {r['arm']} | {r['value']} | {r['frames']} | {r['mean_error_ratio']:.4f} | {r['maxrow_harm_frames']} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    io.write(out/'report_complete.json',dict(status='complete',summary_sha256=io.sha(out/'summary.json'),
        source_files=len(paths),task_accuracy_established=False),True)


if __name__=='__main__':report(io.DEFAULT)
