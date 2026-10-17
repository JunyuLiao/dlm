"""CPU-only, independently bound official scoring of immutable paused panels.

No completion text, answer, question ID or token IDs are emitted. Repetitions
are averaged within questions before paired inference. Partial coverage is
explicit; this analysis never marks the parent full503 panel complete.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import numpy as np
from scripts import v31_official as official

CURRENT = Path('results/v31_value_e2e_longbench503_20261017/native_attempt001')
OLD = Path('results/v31_value_decision_reuse_20261015/native_replicated_20261016_attempt001')
METHODS = ['full_swap8', 'residual_direction_guard']
LABELS = ['control_before', 'control_after', *METHODS, 'dense_full_fix51994', 'dense_piecewise', 'allkept_fa4']
FIELDS = dict(N='denoise_forwards', C='canvases', T='output_tokens', S='decode_s', W='wall_s')


def read(p):
    return json.loads(Path(p).read_text())


def lines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x]


def sha(p):
    return official.sha_file(p)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load(root, pattern, current=False):
    pool = root / 'private/pools/longbench_v2'
    pools = official.Pools(pool, fields=('thinking',))
    gold = read(pool/'longbench_v2_0shot_think_gold.json')
    final, parser = official.load_final_response()
    markers = sorted(root.glob(pattern))
    records, proof, hashes = {}, [], {}
    source = sha(root/'receipts/source_fingerprints.json')
    for marker in markers:
        done = read(marker); leaf = marker.parent
        require(done['status'] == 'complete' and done['returncode'] == 0, 'incomplete engine')
        require(not done.get('errors', done.get('structural_errors')), 'structural errors')
        require(done['protocol_sha256'] == sha(root/'protocol.json'), 'protocol mismatch')
        for name, digest in done['files_sha256'].items():
            require(sha(leaf/name) == digest, 'immutable artifact mismatch: '+name)
        hashes[str(marker.relative_to(root))] = sha(marker)
        log = leaf/'private/worker.log'
        warning = dict(allocator_oom=0, measured_jit=0, warmup_jit=0)
        if log.exists() and current:
            warm = read(leaf/'config.json')['warmup_requests']; completed = 0
            for line in log.read_text().splitlines():
                warning['allocator_oom'] += 'memory allocation failed with OOM' in line
                if 'JIT compilation during inference' in line:
                    warning['warmup_jit' if completed < warm else 'measured_jit'] += 1
                try:
                    logged = json.loads(line)
                except ValueError:
                    continue
                if isinstance(logged, dict) and 'denoise_forwards' in logged and 'wall_s' in logged:
                    completed += 1
            hashes[str(log.relative_to(root))] = sha(log)
        proof.append(dict(engine=str(leaf.relative_to(root)), gpu_reservation_s=done['gpu_reservation_s'], warnings=warning))
        for folder in sorted((leaf/'exports').iterdir()):
            label = folder.name
            public = lines(folder/'records.jsonl')
            private = lines(folder/f'private/run_{label}.private.jsonl')
            require(len(public) == len(private), 'public/private mismatch')
            for r, t in zip(public, private):
                i = r['index']
                require(i not in records.setdefault(label, {}), 'duplicate input')
                require(official.cell_key(r) == official.cell_key(t), 'private cell mismatch')
                meta, binding, verified = official.bind(t, pools)
                require(verified, 'unbound private record')
                for field in official.COMPARE_FIELDS:
                    require(r[field] == t[field], 'public/private settings mismatch')
                require(r['frozen_source_identity_sha256'] == source, 'source mismatch')
                require(r['cuda_graph_captures'] == 0, 'measured graph capture')
                require(r['sampler_calls'] == r['denoise_forwards']+r['canvases'], 'sampler count')
                require(r['canvases'] == math.ceil(r['output_tokens']/256), 'canvas count')
                if current:
                    require(r['panel_seed'] == 42 and r['repeat'] == 0, 'current schedule')
                    require(r['cuda_matmul_allow_tf32'] is False, 'TF32 mode')
                    adapter = (r.get('receipts') or {}).get('adapter') or {}
                    if r['value_audit']:
                        require(adapter['native_local_skipped_tiles'] == 0, 'LOCAL sparse')
                        require(adapter['global_calls'] == 5*r['denoise_forwards'], 'GLOBAL count')
                        require(adapter['native_local_calls_audited'] == 25*r['denoise_forwards'], 'LOCAL count')
                        if label in METHODS:
                            require(adapter['value_refinement_work']['budget_errors'] == 0, 'budget mismatch')
                            require(adapter['value_refinement_layers'] == [29], 'value scope')
                    else:
                        require(r['value_clean_timing'] and r['timer_policy'] == 'request_and_prefill_boundaries', 'clean timer')
                        require(abs(r['wall_s']-r['decode_s']-r['prefill_s']) < 2.1e-5, 'timer partition')
                score = int(official.lb_extract_answer(final(t['completion'], bool(meta['thinking']))) == gold[t['id']])
                records[label][i] = dict(r, score=score, engine=str(leaf.relative_to(root)))
    return records, dict(engines=proof, bindings=hashes, parser=parser,
                         protocol_sha256=sha(root/'protocol.json'), source_identity_sha256=source)


def average(rows):
    return dict({m: statistics.mean(r[f] for r in rows) for m, f in FIELDS.items()},
                accuracy=statistics.mean(r['score'] for r in rows),
                cap=statistics.mean(official.finish_class(r['finish_reason']) == 'capped' for r in rows))


def summaries(cells):
    if not cells:
        return dict(questions=0)
    means = {s: {m: statistics.mean(c[s][m] for c in cells) for m in cells[0][s]} for s in ('control', 'candidate')}
    a, b = means['control'], means['candidate']
    delta = {m: b[m]-a[m] for m in a}
    nc0, nc1 = a['N']/a['C'], b['N']/b['C']
    canvas = (b['C']-a['C'])*(nc0+nc1)/2
    within = (nc1-nc0)*(a['C']+b['C'])/2
    require(abs(canvas+within-delta['N']) < 1e-8, 'decomposition')
    return dict(questions=len(cells), **means, delta=delta,
                calls_per_canvas=[nc0, nc1], amortized_ms_per_N=[a['S']/a['N']*1000,b['S']/b['N']*1000],
                N_decomposition=dict(canvas=canvas, within_canvas=within),
                N_changes={k: sum(test(c['candidate']['N']-c['control']['N']) for c in cells)
                           for k, test in [('fewer',lambda x:x<0),('equal',lambda x:x==0),('more',lambda x:x>0)]},
                median_N_delta=statistics.median(c['candidate']['N']-c['control']['N'] for c in cells))


def compare(records, controls, candidates, indices, rng):
    cells = [dict(index=i, control=average([records[l][i] for l in controls]),
                  candidate=average([records[l][i] for l in candidates])) for i in indices]
    result = summaries(cells)
    draws = rng.integers(0,len(cells),(20000,len(cells)))
    ci = {}
    for field in ('accuracy','N','C','T','S','W'):
        diff = np.array([c['candidate'][field]-c['control'][field] for c in cells])
        ci[field] = np.quantile(diff[draws].mean(1), [.025,.975]).tolist()
    result['paired_question_bootstrap_95CI_delta'] = ci
    groups = {}
    for c in cells:
        scores = [records[l][c['index']]['score'] for l in controls+candidates]
        g = ('both_correct' if all(scores) else 'both_wrong' if not any(scores)
             else 'higher_mean_score' if c['candidate']['accuracy'] > c['control']['accuracy']
             else 'lower_mean_score' if c['candidate']['accuracy'] < c['control']['accuracy'] else 'mixed_equal')
        groups.setdefault(g,[]).append(c)
    result['outcome_conditioned_descriptive'] = {g:summaries(v) for g,v in groups.items()}
    ordered = sorted(cells,key=lambda c:c['candidate']['N']-c['control']['N'])
    net = sum(c['control']['N']-c['candidate']['N'] for c in cells)
    result['tail_sensitivity'] = dict(top5_savings=ordered[:5], top5_penalties=ordered[-5:],
        top5_share_net_saving=sum(c['control']['N']-c['candidate']['N'] for c in ordered[:5])/net if net>0 else None,
        excluding_top5_savings=summaries(ordered[5:]), excluding_top5_penalties=summaries(ordered[:-5]))
    result['cells'] = cells
    result['engine_shard_descriptive'] = {}
    engines = sorted({records[controls[0]][i]['engine'] for i in indices})
    for engine in engines:
        subset=[c for c in cells if records[controls[0]][c['index']]['engine'] == engine]
        result['engine_shard_descriptive'][engine]=summaries(subset)
    return result


def drift(records, before, after, indices):
    return dict(questions=len(indices), output_hash_disagreements=sum(records[before][i]['output_hash']!=records[after][i]['output_hash'] for i in indices),
                score_disagreements=sum(records[before][i]['score']!=records[after][i]['score'] for i in indices),
                mean_N_delta=statistics.mean(records[after][i]['denoise_forwards']-records[before][i]['denoise_forwards'] for i in indices),
                max_abs_N_delta=max(abs(records[after][i]['denoise_forwards']-records[before][i]['denoise_forwards']) for i in indices))


def main(output):
    require(not output.exists(), 'preserve prior analysis')
    require(read(CURRENT/'PAUSED_USER_REQUEST.json')['status']=='paused_user_request', 'full503 must remain paused')
    data, proof = {}, {}
    for name in ('clean','audit'):
        data[name], proof[name] = load(CURRENT,f'runs/{name}/*/shard*/attempt*/attempt_complete.json',True)
    data['old'], proof['old'] = load(OLD,'shared_runs/longbench_v2/shard*/repeat*/attempt*/attempt_complete.json')
    old_scores = read(OLD/'scoring/longbench_v2/attempt001/official.official.json')
    for label, rr in data['old'].items():
        for r in rr.values():
            require(old_scores[label][official.cell_key(r)] == r['score'], 'old official score disagreement')
    matched = sorted(set.intersection(*(set(rr) for name in ('clean','audit') for rr in data[name].values())))
    old_indices = sorted(set.intersection(*(set(rr) for rr in data['old'].values())))
    overlap = sorted(set(matched)&set(old_indices))
    require(len(matched)==64 and len(old_indices)==64 and overlap==[0,1,16,17,32,33,48,49], 'unexpected coverage')
    rng=np.random.default_rng(20261017)
    result=dict(status='complete_analysis_of_partial_panel', matched_indices=matched, old_indices=old_indices, overlap=overlap,
                interpretation='Paired question inference conditional on observed engines; only two current shards. Outcome conditioning, trimming and engine strata are descriptive. Old timing was audit-only and is not clean latency.')
    for name, rr in data.items():
        indices=old_indices if name=='old' else matched
        controls=[f'r{r}_control_{p}' for r in (1,2) for p in ('before','after')] if name=='old' else ['control_before','control_after']
        result[name]=dict(coverage={l:len(v) for l,v in rr.items()}, summaries={l:average([rows[i] for i in indices]) for l,rows in rr.items()}, comparisons={})
        result[name]['summaries']['v31_bookend_mean']=average([rr[l][i] for l in controls for i in indices])
        for method in METHODS:
            candidates=[f'r{r}_{method}' for r in (1,2)] if name=='old' else [method]
            result[name]['comparisons'][method]=compare(rr,controls,candidates,indices,rng)
            result[name]['comparisons'][method]['eight_overlap']=compare(rr,controls,candidates,overlap,rng)
        result[name]['bookend_drift']={str(r):drift(rr,f'r{r}_control_before',f'r{r}_control_after',indices) for r in (1,2)} if name=='old' else drift(rr,*controls,indices)
        if name=='old':
            result[name]['repeat_comparisons']={str(r):{m:compare(rr,[f'r{r}_control_before',f'r{r}_control_after'],[f'r{r}_{m}'],indices,rng) for m in METHODS} for r in (1,2)}
    result['clean_audit_agreement']={l:dict(output_hash_disagreements=sum(data['clean'][l][i]['output_hash']!=data['audit'][l][i]['output_hash'] for i in matched),
        score_disagreements=sum(data['clean'][l][i]['score']!=data['audit'][l][i]['score'] for i in matched),
        mean_N_audit_minus_clean=statistics.mean(data['audit'][l][i]['denoise_forwards']-data['clean'][l][i]['denoise_forwards'] for i in matched)) for l in LABELS}
    result['value_cost_audit']={}
    for label in METHODS:
        rr=[data['audit'][label][i] for i in matched]
        anchors=sum(r['receipts']['adapter']['value_refinement_work']['anchor_events'] for r in rr)
        held=sum(r['receipts']['adapter']['value_refinement_work']['held_calls'] for r in rr)
        ms=sum(r['receipts']['adapter']['value_anchor_module_cuda_elapsed_ms'] for r in rr)
        result['value_cost_audit'][label]=dict(questions=len(rr), anchors=anchors, anchors_per_problem=anchors/len(rr),
            value_ms_per_anchor=ms/anchors,value_s_per_problem=ms/len(rr)/1000,
            denoise_calls_per_anchor=sum(r['denoise_forwards'] for r in rr)/anchors,held_calls_per_anchor=held/anchors,
            interpretation='Instrumented module elapsed time includes launch gaps; not clean removable overhead')
    result['audit_physical_work']={}
    for label in LABELS:
        adapters=[data['audit'][label][i]['receipts']['adapter'] for i in matched]
        fields=['global_eligible_tiles','global_kept_tiles','native_local_eligible_rectangles','native_local_skipped_tiles','overall_eligible_rectangles','overall_skipped_rectangles']
        sums={f:sum(a[f] for a in adapters) for f in fields}
        sums['global_sparsity']=1-sums['global_kept_tiles']/sums['global_eligible_tiles']
        sums['overall_sparsity']=sums['overall_skipped_rectangles']/sums['overall_eligible_rectangles']
        result['audit_physical_work'][label]=sums
    config=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        analysis_sha256=sha(__file__), official_sha256=sha(official.__file__), bootstrap_seed=20261017,bootstrap_samples=20000,
        sources=proof, parent_status='paused; incomplete full503')
    output.mkdir(parents=True)
    for name,obj in [('config.json',config),('summary.json',result)]:
        (output/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    (output/'final_complete.json').write_text(json.dumps(dict(status='complete_cpu_analysis',files_sha256={n:sha(output/n) for n in ('config.json','summary.json')}),indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],matched=len(matched),old=len(old_indices),engines={n:len(p['engines']) for n,p in proof.items()})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
