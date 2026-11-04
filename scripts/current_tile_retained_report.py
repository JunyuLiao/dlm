"""Scannable six-panel closeout; historical baselines remain unchanged."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from scripts.llada22_v2_humaneval_diagnose import token_metrics
from scripts.multi_model_score import final_text

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'results/multi_model_value_20261030/attempt001'
OLD=REPO/'results/multi_model_value_report_20261102/attempt002'
ARMS={'dense_native':'Native dense','allkept_fa4':'All-kept FA4','mass_topk_reuse_r4':'V31',
      'value_v2_gaussian32_reuse_r4':'Original V2','value_v3a_singleton_reuse_r4':'V3a'}
ORDER=[('llada22','aime26'),('llada22','humaneval'),('llada22','ruler128k'),
       ('trado','aime26'),('trado','humaneval'),('trado','ruler32k')]
NAMES={'llada22':'LLaDA2.2-mini','trado':'TraDo-8B-Instruct'}
SUITES={'aime26':'AIME26','humaneval':'HumanEval','ruler128k':'RULER128K','ruler32k':'RULER32K'}


def read(p):return json.loads(p.read_text())
def rows(p):return list(map(json.loads,p.read_text().splitlines()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def main(root):
    marker=read(root/'final_complete.json');assert marker['panels']==6 and marker['requests']==768
    analysis=root/marker['analysis'];summary=read(analysis/'summary.json');scored=rows(analysis/'scored_records.jsonl')
    assert marker['receipt_sha256']==sha(analysis/'receipt.json')
    out=root/'publication001';out.mkdir(exist_ok=False)
    old=rows(OLD/'scored_records.jsonl');allmetrics=[];code_stats={}
    for model,suite in ORDER:
        key=f'{model}/{suite}';new=summary[key]
        assert .4<=new['consumer_sparsity']<=.6
        cohort=[(label,[r for r in old if (r['model'],r['suite'],r['arm'])==(model,suite,arm)]) for arm,label in ARMS.items()]
        cohort.append(('V2 current-tile init',[r for r in scored if (r['model'],r['suite'])==(model,suite)]))
        for label,rs in cohort:
            if not rs:continue
            counts=Counter()
            for r in rs:counts.update(r['counts'])
            n,c=counts['denoising_forwards'],counts['canvases'];e,k=counts['eligible_tiles'],counts['kept_tiles'];total=len(rs)
            metrics={'model':model,'suite':suite,'method':label,'requests':total,'correct_or_full_credit':sum(r['quality']==1. for r in rs),
                'accuracy':statistics.mean(r['quality'] for r in rs),'N':n,'C':c,'T':sum(r['output_tokens'] for r in rs),
                'N_per_request':n/total,'C_per_request':c/total,'N_per_C':n/c,'mean_output_tokens':statistics.mean(r['output_tokens'] for r in rs),
                'median_output_tokens':statistics.median(r['output_tokens'] for r in rs),'max_output_tokens':max(r['output_tokens'] for r in rs),
                'eligible_tiles':e,'kept_tiles':k,'sparsity':1-k/e,'eos_finished':sum(r['finished'] for r in rs),
                'cap_hits':sum(not r['finished'] and r['output_tokens']>=r['generation_cap'] for r in rs),
                'selection_layer_calls':counts['selection_layer_calls'],'held_layer_calls':counts['held_layer_calls'],
                'structural_edit_events':counts['structural_edit_events'],
                'fully_unmasked_refinement_calls':counts['fully_unmasked_refinement_calls']}
            if suite=='humaneval':
                data=read(root/'private/data'/model/'humaneval.json');gold={r['id']:r['gold'] for r in data}
                if label=='V2 current-tile init':
                    leaf=root/'panels'/model/suite/f"attempt{new['attempt']:03}"
                    paths=leaf.glob('runs/shard*/attempt001/private/generation.private.jsonl')
                else:
                    arm=next(a for a,l in ARMS.items() if l==label)
                    paths=(BASE/'runs'/model/suite/arm).glob('shard*/attempt001/private/generation.private.jsonl')
                private={r['id']:r for p in paths for r in rows(p)};codes=Counter(r.get('score_detail','') for r in rs)
                assert len(private)==164
                for r in rs:
                    m=token_metrics(final_text(private[r['id']]['text']),gold[r['id']]['entry_point'])
                    for k in ('entry_point_present','entry_point_docstring_only','entry_point_stub_only','entry_point_has_substantive_body'):codes[k]+=int(m[k])
                metrics['code_completeness']=dict(codes);code_stats[f'{model}/{label}']=dict(codes)
            allmetrics.append(metrics)
    assert len(allmetrics)==31
    write(out/'comparison_metrics.json',allmetrics);write(out/'code_completeness.json',code_stats)
    metric={(m['model'],m['suite'],m['method']):m for m in allmetrics}
    lines=['# Current-tile V2: final retained LLaDA and TraDo results','',
      'Completed six panels /768 benchmark requests. DiffusionGemma was stopped by the user; its queued HumanEval attempt failed during execution-link setup before GPU generation. Its three panels are excluded. The original nine-panel protocol is scope-superseded, not declared complete. H100 is idle; other methods and historical evidence remain unchanged.','',
      '| Model | Benchmark | Native dense | All-kept FA4 | V31 | Original V2 | V3a | V2 current-tile init |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for model,suite in ORDER:
        cells=[]
        for label in (*ARMS.values(),'V2 current-tile init'):
            m=metric.get((model,suite,label))
            cells.append('—' if m is None else f"{100*m['accuracy']:.2f}%"+(f" ({m['correct_or_full_credit']}/{m['requests']})" if not suite.startswith('ruler') else ''))
        lines.append('| '+' | '.join([NAMES[model],SUITES[suite],*cells])+' |')
    lines+=['','AIME is exact-match accuracy over30 problems x3seeds; HumanEval is official functional pass@1 over164 problems x1seed; RULER is fractional retrieval credit over130problems (10/task across13tasks), not binary accuracy. TraDo32K is new and has no historical baseline. All other methods are reused without regeneration.','',
      '| Model | Benchmark | Mean output tokens | N total | C total | N/request | C/request | N/C | Sparsity | EOS / cap hits |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model,suite in ORDER:
        m=metric[model,suite,'V2 current-tile init']
        lines.append(f"| {NAMES[model]} | {SUITES[suite]} | {m['mean_output_tokens']:.1f} | {m['N']} | {m['C']} | {m['N_per_request']:.2f} | {m['C_per_request']:.2f} | {m['N_per_C']:.3f} | {100*m['sparsity']:.2f}% | {m['eos_finished']} / {m['cap_hits']} |")
    lines+=['','N is denoising forwards, C native canvases, T emitted output tokens including emitted EOS. N/C is the pooled ratio. Cap hits are requests ending without EOS at the configured output limit. Lower N from failed short output is not useful acceleration. All new panel sparsities are inside the user-approved40–60% range; no completed panel required another sparsity-driven rerun.','',
      '| Model | Benchmark | Threshold | Calibration prompts/candidate | Candidates | Calibration sparsity | Final sparsity |',
      '|---|---|---:|---:|---:|---:|---:|']
    for model,suite in ORDER:
        s=summary[f'{model}/{suite}'];cal=s['calibration']
        lines.append(f"| {NAMES[model]} | {SUITES[suite]} | {s['threshold']:.12g} | {cal['requests']} | {len(cal['trials'])} | {100*cal['sparsity']:.2f}% | {100*s['consumer_sparsity']:.2f}% |")
    lines+=['',
      'Calibration uses physical tile counts only. AIME/HumanEval use six fixed prompts chosen by character-length strata from the examined benchmark, seed42: transductive and exploratory. Retrieval uses13 separately generated development prompts, seed142. No gold or functional score selects thresholds. Calibration runs only those small samples, never the90-request AIME benchmark.','',
      'The new selector processes the mandatory current KV64 tile first, then chronological tiles0..J-2, once each. Gaussian32 projection, conditional normalized-state updates, preserve-mass skip rule, full-dimensional current-V output, Q128/KV64 geometry and four-call support reuse remain. No forced first prefix tile, mass-ranking repair, EOS override or extra model forward is added. All denoising layers are eligible; prompt prefill and committed-block attention remain dense.','',
      '| Model | Native block | Native generation settings | Output cap: AIME / HumanEval / RULER | RULER prompt-token range |',
      '|---|---:|---|---:|---:|',
      '| LLaDA2.2-mini |32|Greedy temperature0; M2T confidence0.5; editing threshold0; max poststeps16; steps32; native editing and EOS|32768 /1024 /128|115092–130776|',
      '| TraDo-8B-Instruct |4|Temperature1; top-p1, top-k0; dynamic confidence0.9; four denoising steps; native EOS|2000 /2000 /128|29798–32472|','',
      'Both use batch1,BF16 on the same H100, the original model revisions, native chat/prompt adapters and source-pinned numerical runtimes. Seeds42/43/44 for AIME,42 for HumanEval/RULER. RULER fits within each model limit without truncation; LLaDA128K and TraDo32K are different tokenizer-specific prompts.','',
      'Sparsity is1−sum(kept)/sum(eligible) across actual native denoising consumers, all eligible layers/heads, including mandatory current tiles and refresh/reuse calls. It excludes selector discovery, dense prefill and commits; it is geometric tile sparsity, not whole-model FLOPs or attention mass. Original dense SDPA and all-kept FA4 remain separate controls. Timings were instrumented; no clean latency/speedup claim is made.','',
      'The largest gain is LLaDA HumanEval:10->119passes,62.9->219.3 mean output tokens and4.634->3.355steps/canvas. Its new55.67% sparsity differs from original49.22%, so this is an initialization-plus-recalibration result within the accepted band, not an exactly matched50% or fixed-threshold ablation. It remains below native141/164. TraDo HumanEval is122/164 versus original125/164; AIME and LLaDA retrieval do not establish a universal improvement.','',
      'Paired problem-bootstrap intervals (10000resamples; AIME clusters its three seeds by problem) are in ../'+marker['analysis']+'/paired.json. LLaDA HumanEval gain over V31 is+21.95pp[12.80,31.10], over original V2+66.46pp[59.15,73.17]. TraDo HumanEval difference from original V2 is−1.83pp[−7.93,4.27]. These intervals are exploratory and do not remove sparsity or runtime reproduction limitations.','',
      'TraDo control limitation: original V2 historical replay differs despite matching pinned source, prompt, seed and threshold; fresh repeats match one another. All-kept replay exactly reproduces historical hashes/counts. Every TraDo panel records this comparison and the disclosed qualification policy. Historical V2 comparison is descriptive, not a bitwise reproduced reference. No intrinsic model explanation is inferred.','',
      'Maintenance recovery: LLaDA RULER128K attempt001 contains two complete33-request shards and21partial records. Attempt002 inherits only the66complete hash/token-verified records and regenerates both remaining shards at the frozen threshold. The21partial records are preserved and excluded. Its completed130-request panel has explicit mixed-attempt lineage. The earlier outside-band LLaDA HumanEval61.35% development result is preserved separately; the accepted campaign panel uses its own calibration.','',
      'Verification:11 focused CPU tests;96 CUDA recurrence cases before and again after maintenance; independent private-token hashes and record/config/protocol/threshold joins;24 complete shard references across6panels; official scorer;768 unique planned cells;138 protected historical files unchanged. No campaign GPU processes remain.','',
      'Full method-by-method work table follows. Every historical denominator belongs to its own native trajectory.','',
      '| Model | Benchmark | Method | Score | Mean T | N total | C total | N/C | Sparsity | EOS / caps |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in allmetrics:
        lines.append(f"| {NAMES[m['model']]} | {SUITES[m['suite']]} | {m['method']} | {100*m['accuracy']:.2f}% | {m['mean_output_tokens']:.1f} | {m['N']} | {m['C']} | {m['N_per_C']:.3f} | {100*m['sparsity']:.2f}% | {m['eos_finished']} / {m['cap_hits']} |")
    lines+=['','Raw prompts,gold,token IDs,completions and unsanitized process paths remain private. Sanitized per-cell scores,complete counts,calibration histories,source identities and completion receipts are retained.','']
    (out/'README.md').write_text('\n'.join(lines))
    verified={'status':'pass','panels':6,'requests':768,'comparison_rows':31,'source_sha256':sha(__file__),
              'analysis_receipt_sha256':sha(analysis/'receipt.json'),'completion_sha256':sha(root/'final_complete.json'),
              'scope_change_sha256':sha(root/'USER_SCOPE_CHANGE_20261104.json'),'metrics_sha256':sha(out/'comparison_metrics.json'),
              'report_sha256':sha(out/'README.md'),'maintenance_recovery_sha256':sha(root/'maintenance_recovery_20261104.json')}
    write(out/'receipt.json',verified)
    print(json.dumps({'status':'pass','comparison_rows':31,'code_completeness':code_stats}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root.resolve())
