"""Complete finite-roster baseline audit, not a representative benchmark panel."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,math
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'profiling/attempt002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def mean_ci(byquestion):
    a=np.asarray([np.mean(x) for x in byquestion.values()]);rng=np.random.default_rng(20261026);b=a[rng.integers(0,len(a),(20000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),interval95=np.quantile(b,[.025,.975]).tolist(),question_clusters=len(a))
def main():
    assert load(P/'execution_complete.json')['passed'];inputs={n:sha(P/n) for n in ['protocol.json','execution_complete.json','roster_identity.json','manifest_identity.json','manifest_binding.json']};meta={x['id']:x for x in load(P/'roster_identity.json')['identities']};records={}
    for p in (P/'runs/profile').glob('*/audit*.json'):
        v=load(p);key=(v['id'],v['seed'],v['arm']);assert key not in records;records[key]=v;inputs[str(p.relative_to(P))]=sha(p)
        assert v['model_forwards']==math.ceil(meta[v['id']]['prompt_token_count']/16384)+v['N']+v['C'];assert v['uncommitted_tail']==1;assert v['startup_model_calls_in_request']==v['extra_inside_execute_Python_forwards']==0
        assert v['work']['overall_skipped_rectangles']==v['work']['overall_rectangle_sparsity']==0
    assert len(records)==216;groups=defaultdict(list)
    for key,v in records.items():
        groups[(v['benchmark'],v['arm'],'all')].append(v)
        if v['benchmark']=='longbench_v2':groups[(v['benchmark'],v['arm'],meta[v['id']]['native_band'])].append(v)
    tables=[]
    for (task,arm,band),rs in groups.items():
        q=defaultdict(list)
        for v in rs:q[v['id']].append(int(v['correct']))
        tables.append(dict(benchmark=task,arm=arm,native_band=band,requests=len(rs),questions=len(q),correct=sum(v['correct'] for v in rs),quality=mean_ci(q),N=sum(v['N'] for v in rs),C=sum(v['C'] for v in rs),T=sum(v['T'] for v in rs),pooled_N_per_C=sum(v['N'] for v in rs)/sum(v['C'] for v in rs),output_capped=sum(v['output_capped'] for v in rs),missing_final=sum(v['no_final_response'] for v in rs),canvas_cap_events=sum(not r['commit'] and r['cap_reached'] for v in rs for r in v['native_trace']),native_context_range=[min(meta[v['id']]['prompt_token_count'] for v in rs),max(meta[v['id']]['prompt_token_count'] for v in rs)]))
    controls=[];paired=[]
    for task in ['aime26','humaneval','longbench_v2']:
        qdiff=defaultdict(list);qn=defaultdict(list);qnc=defaultdict(list);qt=defaultdict(list);trans=defaultdict(int)
        for (identity,seed,arm),a in records.items():
            if arm!='matched_allkept_l5_l17_l29' or a['benchmark']!=task:continue
            f=records[(identity,seed,'corrected_dense_full')];p=records[(identity,seed,'dense_piecewise')]
            eq={k:f[k]==p[k] for k in ['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','model_forwards','prompt_tokenization_sha256']};controls.append(dict(id=identity,seed=seed,checks=eq,all_passed=all(eq.values())))
            qdiff[identity].append(int(a['correct'])-int(f['correct']));qn[identity].append(a['N']/f['N']);qnc[identity].append(a['N_per_C']/f['N_per_C']);qt[identity].append(a['T']/f['T']);trans[f'{int(f["correct"])}->{int(a["correct"])}']+=1
        paired.append(dict(benchmark=task,A_minus_FULL_quality=mean_ci(qdiff),A_over_FULL_N=mean_ci(qn),A_over_FULL_paired_N_per_C=mean_ci(qnc),A_over_FULL_T=mean_ci(qt),quality_transitions=dict(trans)))
    save(R/'summary.json',dict(passed=True,profile_requests=216,questions=36,questions_per_task=12,two_seeds=[42,43],tables=tables,paired_A_FULL=paired,FULL_PIECEWISE_complete_match_cells=sum(v['all_passed'] for v in controls),FULL_PIECEWISE_comparisons=72,clean_timing_panel=False,scope='finite selection roster; not representative benchmark or sparse-method evaluation'));save(R/'FULL_PIECEWISE_checks.json',controls);save(R/'input_hashes.json',inputs);save(R/'final_complete.json',dict(passed=True,protocol_sha256=sha(P/'protocol.json'),summary_sha256=sha(R/'summary.json'),source_sha256=sha(Path(__file__))))
if __name__=='__main__':main()
