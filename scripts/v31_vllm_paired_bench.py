"""Seed-paired vLLM panel worker: dense vs the method inside vLLM 0.30.0's native DiffusionGemma.

Why: vLLM's diffusion sampler draws its initial canvas and per-step Gumbel noise from the default torch CUDA
generator, and vLLM rejects per-request seeds for diffusion models. Without control, one request's denoising
trajectory varies several-fold between runs (V29: 99 vs 374 forwards for one 32K request), so arms cannot be
compared request by request. Here the default torch CPU/CUDA generators are reseeded before every request with
seed = f(dataset, index, panel seed, repeat), identical for every arm. Verified: same mode + same seed gives
token-identical outputs (results/v31_20261003/graphmode001). The method's own randomness uses private generators
(projection bank on CPU, State generator), so it never shifts the sampler's noise.

Arms (one engine per process): dense | native | allkept | method | mage (MAGE port, budget MAGE_K tokens); cudagraph mode per process (CG='default' keeps
vLLM's default FULL+PIECEWISE, 'PIECEWISE' the matched mode). Adapter arms require PIECEWISE.

Per request (public record, no text): prompt tokens, output tokens, sampler calls, canvases C (= commits =
ceil(output / 256)), denoising forwards N = sampler calls - C, prefill / decode / wall time (each engine step is
synchronized), per-step decode times summary, finish reason, adapter + method receipts. Private: raw completion with
special tokens (for the v15 final-channel LongBench scorer), id, finish reason.
Binding (public AND private record): manifest_sha256 (sha256 of the dataset's manifest file), prompt_sha256 (sha256 of
json.dumps(prompt token ids)), budget, max_model_len, chunk, block_size (+ rng_seed, prompt_tokens in the private
record), so a scorer can refuse a completion produced from another pool / budget / setting. MAX_MODEL_LEN pins vLLM's
max_model_len (refused below the cells' longest prompt + budget); unset, it is derived from the cells as before.
usage: python v31_vllm_paired_bench.py MODEL MANIFEST_DIR CELLS_JSON OUT_JSONL PRIVATE_JSONL ARM CG [CONFIG_JSON]
  env: FIX_51994=1 (backport the upstream FULL-graph causal-buffer fix), FA4_LOCAL_FIX=1 (LOCAL sliding-window calls
       of bidirectional canvases skip the KV blocks outside the window; see apply_fa4_local_fix), LOGIT_STATS=fused (one-pass sampler-hook
       statistics, v31_logit_stats; default legacy torch ops), DP_BUILD=chunked (parallel dense-prefix build,
       v31_dp_chunked), OBSERVE=fa4 (FA4 in-kernel observation for compact-mu configs, v31_fa4_observe),
       MAGE_SELECT=fa4 (MAGE selection statistics from the FA4 observation), KV_COPY=triton / MERGE=triton (V30 one-kernel
       paged K/V refresh and alias-split LSE merge), TRACE=1 (per-canvas step counts and mean-entropy trajectories in the
       public record), DENSE_WHEN=conv:THETA|step:S (method arm: dense GLOBAL attention near canvas convergence / from
       step S of a canvas), MAGE_CRIT=TAU (MAGE + per-query-head critical tiles with mass share >= TAU), MAGE_STEP=S (MAGE selects at step S of a canvas after S exact steps; default 0), MAGE_GRAN=kvhead|qhead|qblock|qblock_max|kvblock_max|kvhead_max (selection unit / row aggregation), MAGE_CARRY=1 (with MAGE_STEP >= 1: canvas call 0 runs on the previous canvas's selection, the method's carry_first), RISK_GROUP=kv (method arm, fixed-fraction top-k configs: one top-k decision per (KV head, block), shared by the group's heads), MAGE_FRAC=F (keep fraction of prefix tiles per unit instead of MAGE_K), RESIDUAL=centroid (dropped prefix tiles added back as centroid key / mean value, any sparse arm), DROP_GUARD=THETA (a (head, block) whose dropped tiles hold > THETA of its estimated mass keeps its whole prefix, any sparse arm), DRIFT_DIAG=1 (receipt: step-to-step drift of GLOBAL queries / outputs within a canvas), DENSE_BELOW=K (any sparse arm: GLOBAL calls with fewer than K keys run vLLM's own dense attention), REGROUP_DIAG=1 (kept fraction at 128 / 64 / regrouped-64 rows / per row), REPEATS (default 1), MEM (0.85), BLOCK (32), CHUNK (16384), LIMIT, DATASETS (comma list), SEED_BASE (31),
       V27_ADAPTER_DIR (overlay holding vllm_adapter.py), SHARD=k/K (take cells k, k+K, ...)
"""
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
CANVAS = 256


def request_seed(base, cell, repeat):
    key = f"{base}|{cell['dataset']}|{cell.get('index')}|{cell['seed']}|{repeat}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'little') & 0x7FFFFFFF


def canvas_seed(seed, canvas):
    key = f'{seed}|canvas|{canvas}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'little') & 0x7FFFFFFF


class CanvasForcing:
    """v31 forced-canvas mode (opt-in; timing fields of these records are NOT valid -- the mode synchronizes).

    Free-running arms diverge after the first differing token, so their steps per canvas (N/C) compare different texts
    and stay noisy (72 LongBench-v2 64K cells: N/C CI about +-5%; at 96K below 1). This mode compares arms on the SAME
    canvases:
      * per-canvas reseeding: before every commit step (which draws the next canvas's initial noise) the CUDA generator
        is reseeded from (request seed, canvas index), so canvas i's noise does not depend on how many steps earlier
        canvases took;
      * FORCE_RECORD=path: write this run's committed token ids per cell (private) -- the reference trajectory;
      * FORCE_REF=path: when a canvas converges, record the arm's own denoising-step count and the fraction of its
        converged argmax tokens equal to the reference's, then overwrite the converged canvas (argmax canvas, canvas,
        draft tokens) with the reference's tokens before the commit step, so the commit encodes and emits the
        reference text and every canvas starts from the reference prefix.
    Self-check: the reference configuration forced with its own record must reproduce every step count and agree 1.0.
    One decoding request at a time (max_num_seqs=1)."""

    def __init__(self, ref_path, record_path):
        self.ref = {}
        if ref_path:
            for line in open(ref_path, encoding='utf-8'):
                r = json.loads(line)
                self.ref[(r['dataset'], r['index'], r['panel_seed'], r['repeat'])] = r['token_ids']
        self.record_path = record_path
        self.mode = 'force' if ref_path else ('record' if record_path else 'reseed')
        # no active request until begin(): vLLM's engine start-up warms the sampler up with dummy decode steps (possibly
        # several slots) -- those calls pass straight through
        self.seed = self.cur_ref = None
        self.active = False
        self.canvas, self.steps, self.steps_list, self.agree = 0, 0, [], []

    @classmethod
    def from_env(cls):
        ref, rec = os.environ.get('FORCE_REF') or None, os.environ.get('FORCE_RECORD') or None
        if ref is None and rec is None and os.environ.get('CANVAS_RESEED') != '1':
            return None
        return cls(ref, rec)

    def begin(self, seed, key):
        if self.mode == 'force' and key not in self.ref:
            raise KeyError(f'no reference trajectory for {key}')
        self.seed, self.cur_ref = seed, self.ref.get(key)
        self.canvas, self.steps, self.steps_list, self.agree = 0, 0, [], []
        self.active = True

    def wrap(self, inner):
        import torch

        def step(*args, **kwargs):
            if not self.active:
                return inner(*args, **kwargs)
            slots, canvas, argmax, enc, draft = args[1], args[5], args[6], args[8], args[17]
            if slots.numel() != 1:
                raise RuntimeError('forced-canvas mode needs exactly one decoding request')
            s = slots[0]
            committing = bool(enc[s].item())
            if committing:
                self.canvas += 1
                torch.cuda.manual_seed(canvas_seed(self.seed, self.canvas))
            out = inner(*args, **kwargs)
            if not committing:
                self.steps += 1
                if bool(enc[s].item()):                                  # converged: the next call commits it
                    self._converged(s, canvas, argmax, draft, int(kwargs['CL']))
            return out
        return step

    def _converged(self, s, canvas, argmax, draft, CL):
        import torch
        i = len(self.steps_list)
        self.steps_list.append(self.steps)
        self.steps = 0
        if self.cur_ref is None:
            return
        seg = self.cur_ref[i * CL:(i + 1) * CL]
        if not seg:
            self.agree.append(None)                                      # past the reference's end (never if forced)
            return
        n = len(seg)
        t = torch.tensor(seg, device=argmax.device, dtype=torch.long)
        self.agree.append(round(float((argmax[s, :n].long() == t).float().mean().item()), 5))
        argmax[s, :n] = t.to(argmax.dtype)
        canvas[s, :n] = t.to(canvas.dtype)
        draft[s, :n] = t.to(draft.dtype)

    def receipt(self, token_ids):
        self.active = False
        r = dict(forced_mode=self.mode, forced_canvas_steps=list(self.steps_list), forced_agree=list(self.agree))
        if self.cur_ref is not None:
            r['forced_output_matches_ref'] = list(token_ids) == list(self.cur_ref)
        return r


def main():
    model_dir, manifest_dir, cells_path, out_path, private_path, arm, cg = sys.argv[1:8]
    config_path = sys.argv[8] if len(sys.argv) > 8 else None
    if arm not in ('dense', 'native', 'allkept', 'method', 'mage'):
        raise ValueError(arm)
    if arm != 'dense' and cg != 'PIECEWISE':
        raise ValueError('adapter arms need CG=PIECEWISE')
    cells = json.loads(Path(cells_path).read_text())
    if os.environ.get('DATASETS'):
        keep = set(os.environ['DATASETS'].split(','))
        cells = [c for c in cells if c['dataset'] in keep]
    if os.environ.get('SHARD'):
        k, K = (int(x) for x in os.environ['SHARD'].split('/'))
        cells = cells[k::K]
    if os.environ.get('LIMIT'):
        cells = cells[:int(os.environ['LIMIT'])]
    repeats = int(os.environ.get('REPEATS', '1'))
    seed_base = int(os.environ.get('SEED_BASE', '31'))
    rows, man_sha = {}, {}
    for ds in {c['dataset'] for c in cells}:
        raw = (Path(manifest_dir) / f'{ds}_generation_manifest.json').read_bytes()
        man_sha[ds] = hashlib.sha256(raw).hexdigest()
        for r in json.loads(raw):
            rows[(ds, r['id'])] = r
    import torch
    import vllm
    captures = {'count': 0}
    original_capture_begin = torch.cuda.CUDAGraph.capture_begin
    def count_capture(graph, *args, **kwargs):
        captures['count'] += 1
        return original_capture_begin(graph, *args, **kwargs)
    torch.cuda.CUDAGraph.capture_begin = count_capture
    import vllm.model_executor.models.diffusion_gemma as dg
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    fix_51994 = os.environ.get('FIX_51994') == '1'
    if fix_51994:
        apply_fix_51994()
    fa4_local_fix = apply_fa4_local_fix() if os.environ.get('FA4_LOCAL_FIX') == '1' else None

    adapter, config, adapter_sha, residual_sha = None, None, None, None
    if arm != 'dense':
        import experiments.numerical_qk_reuse as pkg
        if os.environ.get('V27_ADAPTER_DIR'):
            pkg.__path__.append(os.environ['V27_ADAPTER_DIR'])
        from experiments.numerical_qk_reuse import vllm_adapter
        adapter_sha = hashlib.sha256(Path(vllm_adapter.__file__).read_bytes()).hexdigest()
        from experiments.numerical_qk_reuse import v31_residual
        residual_sha = hashlib.sha256(Path(v31_residual.__file__).read_bytes()).hexdigest()
        text = AutoConfig.from_pretrained(model_dir)
        text = getattr(text, 'text_config', text)
        if arm == 'method':
            config = json.loads(Path(config_path).read_text())
        adapter = vllm_adapter.VllmMethodAdapter(text.layer_types, config=config,
                                                 condition=None if config is None else config['condition'], arm=arm,
                                                 profile=os.environ.get('VALUE_AUDIT') == '1',
                                                 mage_k=int(os.environ.get('MAGE_K', '1024')),
                                                 logit_stats=os.environ.get('LOGIT_STATS', 'legacy'),
                                                 dp_build=os.environ.get('DP_BUILD', 'legacy'),
                                                 observe_backend=os.environ.get('OBSERVE', 'triton'),
                                                 mage_select=os.environ.get('MAGE_SELECT', 'torch'),
                                                 kv_copy_backend=os.environ.get('KV_COPY', 'torch'),
                                                 merge_backend=os.environ.get('MERGE', 'torch'),
                                                 trace_canvas=os.environ.get('TRACE') == '1',
                                                 dense_when=os.environ.get('DENSE_WHEN') or None,
                                                 regroup_diag=os.environ.get('REGROUP_DIAG') == '1',
                                                 mage_critical=float(os.environ['MAGE_CRIT']) if os.environ.get('MAGE_CRIT') else None,
                                                 mage_coverage=float(os.environ['MAGE_COV']) if os.environ.get('MAGE_COV') else None,
                                                 mage_select_step=int(os.environ.get('MAGE_STEP', '0')),
                                                 mage_granularity=os.environ.get('MAGE_GRAN', 'kvhead'),
                                                 mage_keep_frac=float(os.environ['MAGE_FRAC']) if os.environ.get('MAGE_FRAC') else None,
                                                 residual=os.environ.get('RESIDUAL') or None,
                                                 drop_guard=float(os.environ['DROP_GUARD']) if os.environ.get('DROP_GUARD') else None,
                                                 drift_diag=os.environ.get('DRIFT_DIAG') == '1',
                                                 dense_below=int(os.environ['DENSE_BELOW']) if os.environ.get('DENSE_BELOW') else None,
                                                 mage_carry_first=os.environ.get('MAGE_CARRY') == '1',
                                                 risk_group=os.environ.get('RISK_GROUP') or None,
                                                 mage_reselect=([int(x) for x in os.environ['MAGE_RESELECT'].split(',')]
                                                                if os.environ.get('MAGE_RESELECT') else None),
                                                 mage_row_weight=os.environ.get('MAGE_ROWW') or None,
                                                 mage_beta=float(os.environ.get('MAGE_BETA', '3')),
                                                 mage_reselect_k=int(os.environ['MAGE_RESELECT_K']) if os.environ.get('MAGE_RESELECT_K') else None,
                                                 mage_reselect_trigger=([float(x) for x in os.environ['MAGE_RESELECT_TRIGGER'].split(',')]
                                                                        if os.environ.get('MAGE_RESELECT_TRIGGER') else None),
                                                 mage_trigger_signal=os.environ.get('MAGE_TRIGGER_SIGNAL', 'accept'),
                                                 mage_reselect_kmin=int(os.environ['MAGE_RESELECT_KMIN']) if os.environ.get('MAGE_RESELECT_KMIN') else None,
                                                 mage_clock_trace=os.environ.get('MAGE_CLOCK_TRACE') == '1',
                                                 mage_sink=int(os.environ.get('MAGE_SINK', '0')),
                                                 mage_recent=int(os.environ.get('MAGE_RECENT', '0')),
                                                 mage_trigger_relative=os.environ.get('MAGE_TRIGGER_RELATIVE') == '1',
                                                 mage_pool=int(os.environ['MAGE_POOL']) if os.environ.get('MAGE_POOL') else None,
                                                 mage_kcover=float(os.environ['MAGE_KCOVER']) if os.environ.get('MAGE_KCOVER') else None,
                                                 mage_kq=float(os.environ.get('MAGE_KQ', '0.75')),
                                                 mage_kmax=int(os.environ.get('MAGE_KMAX', '16384')),
                                                 cg_stop=int(os.environ['CG_STOP']) if os.environ.get('CG_STOP') else None,
                                                 stall_rescue=int(os.environ['STALL_RESCUE']) if os.environ.get('STALL_RESCUE') else None,
                                                 stall_eps=float(os.environ.get('STALL_EPS', '0.01')),
                                                 mage_sticky=float(os.environ['MAGE_STICKY']) if os.environ.get('MAGE_STICKY') else None,
                                                 value_selector=os.environ.get('VALUE_SELECTOR') or None,
                                                 value_audit=os.environ.get('VALUE_AUDIT') == '1',
                                                 value_threshold=(float(os.environ['VALUE_THRESHOLD'])
                                                                  if os.environ.get('VALUE_THRESHOLD') else None),
                                                 local_kernel=os.environ.get('LOCAL_KERNEL', 'compact_triton'),
                                                 local_kv_budget=(int(os.environ['LOCAL_KV_BUDGET'])
                                                                  if os.environ.get('LOCAL_KV_BUDGET') and arm == 'mage'
                                                                  else None))
        vllm_adapter.install_vllm_patches(adapter)
    counter = dict(calls=0)
    inner = dg._compiled_sample_step                    # (already wrapped by the adapter for adapter arms)

    def counting(*args, **kwargs):
        counter['calls'] += 1
        return inner(*args, **kwargs)
    forcing = CanvasForcing.from_env()
    dg._compiled_sample_step = forcing.wrap(counting) if forcing is not None else counting
    rec_out = open(forcing.record_path, 'a', encoding='utf-8') if forcing is not None and forcing.record_path else None

    longest = max(len(rows[(c['dataset'], c['id'])]['prompt_tokens']) + int(rows[(c['dataset'], c['id'])]['generation_budget'])
                  for c in cells)
    max_len = int(os.environ.get('MAX_MODEL_LEN') or min(262144, ((longest + 4096) // 1024 + 1) * 1024))
    if max_len < longest:
        raise ValueError(f'MAX_MODEL_LEN {max_len} is below what the cells need ({longest} = prompt + budget)')
    chunk = int(os.environ.get('CHUNK', '16384'))
    kw = dict(model=model_dir, dtype='bfloat16', max_model_len=max_len, max_num_seqs=1, max_num_batched_tokens=chunk,
              enable_chunked_prefill=True, gpu_memory_utilization=float(os.environ.get('MEM', '0.85')),
              enable_prefix_caching=False, trust_remote_code=False, seed=0, block_size=int(os.environ.get('BLOCK', '32')))
    if cg != 'default':
        kw['compilation_config'] = {'cudagraph_mode': cg}
    llm = LLM(**kw)
    engine = llm.llm_engine
    tok = llm.get_tokenizer()
    meta = dict(schema='v31_vllm_paired_v1', arm=arm, cudagraph_mode=cg, vllm=vllm.__version__, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(), max_model_len=max_len, chunk=chunk, block_size=kw['block_size'],
                max_model_len_source='env' if os.environ.get('MAX_MODEL_LEN') else 'derived', max_model_len_need=longest,
                gpu_memory_utilization=kw['gpu_memory_utilization'], seed_base=seed_base, adapter_sha256=adapter_sha,
                method_fingerprint=None if config is None else config.get('fingerprint'), fix_51994=fix_51994,
                fa4_local_fix=bool(fa4_local_fix),
                fa4_local_fix_sha256=hashlib.sha256(Path(fa4_local_fix).read_bytes()).hexdigest() if fa4_local_fix else None,
                mage_k=int(os.environ.get('MAGE_K', '1024')) if arm == 'mage' else None,
                local_kv_budget=(int(os.environ['LOCAL_KV_BUDGET'])
                                if os.environ.get('LOCAL_KV_BUDGET') and arm == 'mage' else None),
                local_kernel=os.environ.get('LOCAL_KERNEL', 'compact_triton') if arm == 'mage' else None,
                mage_select=os.environ.get('MAGE_SELECT', 'torch') if arm == 'mage' else None,
                value_selector=os.environ.get('VALUE_SELECTOR') if arm == 'mage' else None,
                value_threshold=(float(os.environ['VALUE_THRESHOLD']) if os.environ.get('VALUE_THRESHOLD') and arm == 'mage' else None),
                kv_copy_backend=os.environ.get('KV_COPY', 'torch') if arm != 'dense' else None,
                merge_backend=os.environ.get('MERGE', 'torch') if arm != 'dense' else None,
                logit_stats=os.environ.get('LOGIT_STATS', 'legacy') if arm == 'method' else None,
                dp_build=os.environ.get('DP_BUILD', 'legacy') if arm == 'method' else None,
                observe_backend=os.environ.get('OBSERVE', 'triton') if arm == 'method' else None,
                dense_when=os.environ.get('DENSE_WHEN') or None, residual_sha256=residual_sha,
                # the adapter's EFFECTIVE settings (not the environment), so a silently ignored option shows here
                **{key: (getattr(adapter, key) if adapter is not None and (arm == 'mage' or not key.startswith('mage_'))
                         else None)
                   for key in ('mage_critical', 'mage_coverage', 'mage_select_step', 'mage_granularity',
                               'mage_keep_frac', 'mage_carry_first', 'residual', 'drop_guard', 'drift_diag',
                               'dense_below', 'risk_group', 'mage_reselect', 'mage_row_weight', 'mage_beta',
                               'mage_reselect_k', 'mage_reselect_trigger', 'mage_trigger_signal',
                               'mage_reselect_kmin', 'mage_clock_trace', 'mage_sink', 'mage_recent_tiles',
                               'mage_trigger_relative', 'mage_pool', 'mage_kcover', 'mage_kq', 'mage_kmax',
                               'cg_stop', 'stall_rescue', 'stall_eps', 'mage_sticky')})
    out = open(out_path, 'a', encoding='utf-8')
    priv = open(private_path, 'a', encoding='utf-8')
    schedule = [(True, cells[0], -1)] + [(False, c, r) for r in range(repeats) for c in cells]
    for schedule_index, (warm, cell, rep) in enumerate(schedule):
        row = rows[(cell['dataset'], cell['id'])]
        ids = list(row['prompt_tokens'])
        params = SamplingParams(max_tokens=int(row['generation_budget']), skip_special_tokens=False)
        seed = request_seed(seed_base, cell, max(rep, 0))
        if adapter is not None:
            adapter.begin_request()
            adapter.value_diagnostic = None
            diagnostic_dir = os.environ.get('VALUE_DIAGNOSTIC_DIR')
            if diagnostic_dir and not warm and schedule_index <= 2:
                from experiments.numerical_qk_reuse.v31_value_snapshots import SnapshotRecorder
                adapter.value_diagnostic = SnapshotRecorder(diagnostic_dir, schedule_index)
        counter['calls'] = 0
        if forcing is not None:
            forcing.begin(seed, (cell['dataset'], cell.get('index'), cell['seed'], max(rep, 0)))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        capture_start = captures['count']
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        engine.add_request(f"{'w' if warm else 'r'}{rep}-{cell['dataset']}-{cell.get('index')}-{cell['seed']}",
                           TokensPrompt(prompt_token_ids=ids), params)
        steps, final = [], None
        while engine.has_unfinished_requests():
            a = time.perf_counter()
            outs = engine.step()
            torch.cuda.synchronize()
            steps.append(time.perf_counter() - a)
            for o in outs:
                if o.finished:
                    final = o
        wall = time.perf_counter() - start
        receipts = adapter.end_request() if adapter is not None else None
        if warm:
            if forcing is not None:
                forcing.active = False
            continue
        o = final.outputs[0]
        n_out = len(o.token_ids)
        canvases = math.ceil(n_out / CANVAS)
        k = -(-len(ids) // chunk)                          # prefill engine steps (one chunk each)
        decode = steps[k:]
        bind = dict(manifest_sha256=man_sha[cell['dataset']], prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    budget=int(row['generation_budget']), max_model_len=max_len, chunk=chunk, block_size=kw['block_size'])
        rec = dict(meta, dataset=cell['dataset'], index=cell.get('index'), panel_seed=cell['seed'], repeat=rep,
                   rng_seed=seed, prompt_tokens=len(ids), **bind, output_tokens=n_out,
                   sampler_calls=counter['calls'], canvases=canvases, denoise_forwards=counter['calls'] - canvases,
                   engine_steps=len(steps), prefill_steps=k, prefill_s=round(sum(steps[:k]), 5),
                   decode_s=round(sum(decode), 5), wall_s=round(wall, 5),
                   step_median_ms=round(1000 * statistics.median(decode), 3) if decode else None,
                   finish_reason=o.finish_reason,
                   cuda_graph_captures=captures['count']-capture_start,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   output_hash=hashlib.sha256(json.dumps(list(o.token_ids)).encode()).hexdigest()[:16],
                   **(forcing.receipt(o.token_ids) if forcing is not None else {}),
                   receipts=None if receipts is None else dict(adapter=receipts.get('adapter'),
                                                               method=_slim(receipts.get('method')),
                                                               timing=receipts.get('timing'),
                                                               trace=receipts.get('trace')))
        out.write(json.dumps(rec, default=str) + '\n')
        out.flush()
        priv.write(json.dumps(dict(arm=arm, cudagraph_mode=cg, dataset=cell['dataset'], index=cell.get('index'),
                                   panel_seed=cell['seed'], repeat=rep, id=cell['id'], finish_reason=o.finish_reason,
                                   rng_seed=seed, prompt_tokens=len(ids), **bind,
                                   completion=tok.decode(o.token_ids, skip_special_tokens=False))) + '\n')
        priv.flush()
        if rec_out is not None:
            rec_out.write(json.dumps(dict(dataset=cell['dataset'], index=cell.get('index'), panel_seed=cell['seed'],
                                          repeat=rep, token_ids=list(o.token_ids))) + '\n')
            rec_out.flush()
        print(json.dumps({k2: rec[k2] for k2 in ('arm', 'cudagraph_mode', 'dataset', 'index', 'panel_seed', 'repeat',
                                                 'output_tokens', 'canvases', 'denoise_forwards', 'wall_s',
                                                 'step_median_ms')}), flush=True)


def _slim(m):
    if not m:
        return m
    keep = ('effective_method', 'carried_first_calls', 'fused_observations', 'dp_routes', 'held_decision_calls',
            'bootstrap_dense_calls', 'preqk_consumer_calls', 'attention_calls', 'layer_native_calls', 'fresh_fused_calls',
            'fa4_list_builds', 'async_observation_routes', 'protected_routes', 'v30_sensitivity', 'cgate')
    return {k: m.get(k) for k in keep if k in m}


def apply_fix_51994():
    """Backport of vLLM PR #51994 (merged 2026-09-30, not in 0.30.0): DiffusionGemma's per-request causal buffer was
    bool, so FlashAttentionMetadataBuilder.build() cast it out of place to int32 on every call; FULL CUDA graphs bound
    the capture-time copy and replayed a frozen causal/bidirectional flag. Allocating the buffer as int32 makes
    prepare_attn's slice assignment an in-place update that captured graphs see (the upstream fix also turns the
    builder's silent cast into an error; behaviour with an int32 buffer is identical)."""
    import torch
    import vllm.model_executor.models.diffusion_gemma as dg
    cls = dg.DiffusionGemmaModelState
    if getattr(cls, '_v31_fix_51994', False):
        return
    init = cls.__init__

    def patched(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._causal_buf = torch.zeros(self._causal_buf.shape[0], dtype=torch.int32, device=self._causal_buf.device)
    cls.__init__ = patched
    cls._v31_fix_51994 = True


FA4_LOCAL_FIX_EDITS = (
    # producer (K/V load loop)
    ("""                    if const_expr(self._mDynamicCausal is not None):
                        psc_producer = self._mDynamicCausal[batch_idx]
""", """                    if const_expr(self._mDynamicCausal is not None and not self.is_local):
                        psc_producer = self._mDynamicCausal[batch_idx]
"""),
    # consumer (main loop)
    ("""            if const_expr(self._mDynamicCausal is not None):
                # Per-sequence causal: psc == 0 means this sequence is processed
""", """            if const_expr(self._mDynamicCausal is not None and not self.is_local):
                # Per-sequence causal: psc == 0 means this sequence is processed
"""),
)


def apply_fa4_local_fix(build_dir=None):
    """FA4_LOCAL_FIX=1: skip the KV blocks outside the sliding window in FA4 SM90 LOCAL calls of bidirectional sequences.

    vLLM 0.30 calls FA4 for DiffusionGemma's LOCAL layers with window (1023, 1023), causal False and a per-sequence
    dynamic_causal tensor; a decode canvas is bidirectional (psc == 0). Such a kernel is compiled is_local, and
    BlockInfo.get_n_block_min_max already returns exactly the KV blocks the window mask can reach (the mask of a
    bidirectional sequence applies that same compiled window). But the dynamic-causal branch of the producer (K/V
    load loop) and the consumer (main loop) -- written for kernels compiled causal, whose range stops at the diagonal
    -- resets every bidirectional sequence to the FULL key range. The extra blocks are fully masked (-inf scores leave
    the row max, the row sum and O unchanged), so outputs are right, but every LOCAL canvas call scans the whole
    context (3.2 ms at 131K keys vs 0.10 ms over the window; 25 LOCAL layers per forward). The fix keeps that reset
    for non-local kernels only, identically on both sides (their block counts must agree or the pipeline deadlocks).
    The blocks inside the window run in the same order as before, so outputs are expected to be bitwise unchanged;
    GLOBAL (non-local) kernels and causal sequences are untouched.
    CuTe DSL re-reads kernel source from the defining file, so the patched copy of the installed flash_fwd_sm90.py is
    written to disk (BUILD/<sha of the original>/) and imported as vllm.vllm_flash_attn.cute.flash_fwd_sm90 before any
    kernel compiles. Returns the patched file's path."""
    import importlib
    import importlib.util
    name = 'vllm.vllm_flash_attn.cute.flash_fwd_sm90'
    cur = sys.modules.get(name)
    if cur is not None and getattr(cur, '_v31_fa4_local_fix', False):
        return cur.__file__
    src = Path(importlib.util.find_spec(name).origin).read_text(encoding='utf-8')
    patched = src
    for old, new in FA4_LOCAL_FIX_EDITS:
        assert patched.count(old) == 1, ('FA4_LOCAL_FIX: unexpected flash_fwd_sm90.py', old[:70], patched.count(old))
        patched = patched.replace(old, new)
    build = Path(build_dir or os.environ.get('FA4_LOCAL_FIX_DIR') or Path(__file__).resolve().parent / '_fa4_local_fix')
    path = build / hashlib.sha256(src.encode('utf-8')).hexdigest()[:16] / 'flash_fwd_sm90.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding='utf-8') != patched:
        tmp = path.with_name(f'flash_fwd_sm90.tmp{os.getpid()}')
        tmp.write_text(patched, encoding='utf-8')
        os.replace(tmp, path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        if cur is None:
            del sys.modules[name]
        else:
            sys.modules[name] = cur
        raise
    mod._v31_fa4_local_fix = True
    setattr(importlib.import_module('vllm.vllm_flash_attn.cute'), 'flash_fwd_sm90', mod)
    iface = sys.modules.get('vllm.vllm_flash_attn.cute.interface')
    if iface is not None:                       # interface imported the class by name at its own import time
        iface.FlashAttentionForwardSm90 = mod.FlashAttentionForwardSm90
    return str(path)


if __name__ == '__main__':
    main()
