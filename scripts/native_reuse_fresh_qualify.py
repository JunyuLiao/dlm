"""Fresh-score qualification against the pinned Junyu CUDA/ATen operator.

This is a separate diagnostic, not a generation/timing run. Both operators
receive identical current BF16 Q/K/V, projected current V, reference and T.
Our numerical path receives ONLY the transformed FP32 scores produced by
Attention.observe_scores, current V, Z, reference, and T. The frozen gates
below precede any GPU result; every bitmap disagreement fails qualification
and is reported with distance from the threshold.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

import torch

from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.numerical_qk_reuse.cached_executor import attention as cached_attention
from experiments.numerical_qk_reuse.integration import Attention as NativeAttention
from experiments.value_direction_hopper.cuda import Kernel
from experiments.value_direction_hopper.masks import pack


# Frozen before seeing fresh-kernel results. Exact physical decisions are the
# gate. These output limits accommodate BF16 PV/order differences while still
# detecting material score, support, or normalization errors.
OUTPUT_MAX_ABS = 0.25
OUTPUT_REL_L2 = 0.03
LSE_MAX_ABS = 0.06
RISK_NEAR_THRESHOLD = 0.05  # explanatory classification, never an exemption
POLICY_SHA256 = 'af2688c56f152347265be1a5e59651bb8e4d1815f19838507c03301f87601ef1'
POLICY_THRESHOLDS = {'local': -1.0099318265914916, 'global': -3.1366905212402343}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else '+Infinity' if value > 0 else '-Infinity'
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def errors(actual, expected):
    a, b = actual.float(), expected.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    nonfinite_mismatch = int((((~torch.isfinite(a)) | (~torch.isfinite(b))) & (a != b)).sum())
    if not bool(finite.any()):
        return dict(max_abs=None, relative_l2=None, nonfinite_mismatch=nonfinite_mismatch)
    delta = a[finite] - b[finite]
    return dict(max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm()/b[finite].norm().clamp_min(1e-12)),
                nonfinite_mismatch=nonfinite_mismatch)


def inputs(spec, seed, device):
    qn, kn, d, h, hk = (spec[x] for x in ('q', 'k', 'd', 'h', 'hk'))
    random_device = 'cpu' if spec.get('cpu_seed', False) else device
    gen = torch.Generator(device=random_device).manual_seed(seed)
    q = torch.randn((1, h, qn, d), device=random_device, dtype=torch.bfloat16, generator=gen).to(device)
    k = torch.randn((1, hk, kn, d), device=random_device, dtype=torch.bfloat16, generator=gen).to(device)
    v = torch.randn((1, hk, kn, d), device=random_device, dtype=torch.bfloat16, generator=gen).to(device)
    if spec['mask'] == 'full':
        # Make the first legal physical block genuinely high-mass via current
        # Q/K, so the frozen global T threshold exercises later PV deletion.
        q[..., 0] = 10.
        k[..., :64, 0] = 10.
        k[..., 64:, 0] = 0.
    keys = torch.arange(kn, device=device)
    rows = torch.arange(qn, device=device)
    if spec['mask'] == 'full':
        mask = torch.ones((1, 1, qn, kn), device=device, dtype=torch.bool)
    elif spec['mask'] == 'causal':
        mask = (keys[None, :] <= (kn-qn+rows)[:, None])[None, None].contiguous()
    elif spec['mask'] == 'local_partial':
        end = kn-qn+rows
        allowed = (keys[None, :] <= end[:, None]) & (keys[None, :] >= end[:, None]-95)
        allowed &= ~((rows[:, None] % 7 == 0) & (keys[None, :] % 5 == 0))
        mask = allowed[None, None].contiguous()
    elif spec['mask'] == 'local_causal':
        end = kn-qn+rows
        window = spec['window']
        allowed = (keys[None, :] <= end[:, None]) & (keys[None, :] >= end[:, None]-window+1)
        mask = allowed[None, None].contiguous()
    elif spec['mask'] == 'additive':
        allowed = keys[None, :] <= (kn-qn+rows)[:, None]
        bias = ((keys[None, :] % 7)-3).float().mul_(.125).expand(qn, kn).clone()
        bias.masked_fill_(~allowed, -math.inf)
        mask = bias[None, None].to(torch.bfloat16).contiguous()
    else:
        raise ValueError(spec['mask'])
    scale = spec.get('scale', d**-.5)
    scores = NativeAttention.observe_scores(q, k, mask, scale, False, None, 0)
    valid = torch.isfinite(scores).reshape(1, hk, h//hk, qn, kn).any((2, 3))
    current = v.float()
    matrix = Projections().get(0, hk, d, 'gaussian', 32, 1729, device)
    z = (current @ matrix).contiguous()
    ref = (current.square().sum(-1).masked_fill(~valid, 0.).sum(-1) /
           valid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12).contiguous()
    sensitivity = (1. + 3.*torch.rand((1, qn), device=random_device, generator=gen)).to(device).contiguous()
    if spec['mask'] == 'additive':
        native_mask = mask.expand(1, h, qn, kn).contiguous()
    else:
        native_mask = pack(torch.isfinite(scores).contiguous())
    return q, k, v, z, ref, sensitivity, scores, native_mask


def describe_bitmap(original, cached, threshold):
    eligible_disagreements = int((original.eligible != cached.eligible).sum())
    mismatch = original.skipped != cached.skipped
    count = int(mismatch.sum())
    near = far = 0
    examples = []
    if count:
        places = mismatch.nonzero().cpu().tolist()
        original_risk = original.risk.float().cpu()
        cached_risk = cached.risk.float().cpu()
        original_skip = original.skipped.cpu()
        cached_skip = cached.skipped.cpu()
        for index in places:
            a, b = (float(x[tuple(index)]) for x in (original_risk, cached_risk))
            is_near = min(abs(a-threshold), abs(b-threshold)) <= RISK_NEAR_THRESHOLD
            near += int(is_near)
            far += int(not is_near)
            if len(examples) < 12:
                examples.append(dict(index=index, original_risk=a, cached_risk=b,
                                     original_drop=bool(original_skip[tuple(index)]),
                                     cached_drop=bool(cached_skip[tuple(index)]),
                                     near_threshold=is_near))
    return dict(eligible_disagreements=eligible_disagreements,
                bitmap_disagreements=count, near_threshold=near,
                away_from_threshold=far, examples=examples)


def run_case(kernel, spec, threshold, seed):
    q, k, v, z, ref, sensitivity, scores, mask = inputs(spec, seed, 'cuda')
    original = kernel(q, k, v, z, ref, mask=mask, scale=spec.get('scale', spec['d']**-.5),
                      log_threshold=threshold, mode='value', precision='tf32x3_register',
                      sensitivity=sensitivity, trace=True, overlap=True, tma=False)
    cached = cached_attention(scores, v, z, ref, sensitivity=sensitivity,
                              log_threshold=threshold, trace=True)
    # Force original support into the same cache consumer. This distinguishes
    # output/normalization differences from selector decision differences.
    matched = cached_attention(scores, v, skipped=original.skipped,
                               eligible=original.eligible, trace=True)
    torch.cuda.synchronize()
    bitmap = describe_bitmap(original, cached, threshold)
    same_support_output = errors(matched.output, original.output)
    same_support_lse = errors(matched.log_normalizer, original.log_normalizer)
    direct_output = errors(cached.output, original.output)
    direct_lse = errors(cached.log_normalizer, original.log_normalizer)
    risk = errors(cached.risk, original.risk)
    q16 = (spec['q']+15)//16
    expected_visits = ((original.eligible & ~original.skipped).sum(-1)
                       .repeat_interleave(8, dim=2)[..., :q16]).to(torch.int32)
    visits = matched.counters[..., 0]
    vloads = matched.counters[..., 1]
    pvops = matched.counters[..., 2]
    counter_errors = dict(visited=int((visits != expected_visits).sum()),
                          v_loads=int((vloads != expected_visits).sum()),
                          pv_ops=int((pvops != expected_visits).sum()))
    all_kept = threshold == -math.inf
    gates = dict(eligibility_exact=bitmap['eligible_disagreements'] == 0,
                 bitmap_exact=bitmap['bitmap_disagreements'] == 0,
                 original_all_kept=(not all_kept or not bool(original.skipped.any())),
                 cached_all_kept=(not all_kept or not bool(cached.skipped.any())),
                 no_invalid_scores=not bool(cached.invalid_scores.any()) and not bool(matched.invalid_scores.any()),
                 same_support_output=(same_support_output['nonfinite_mismatch'] == 0 and
                                      same_support_output['max_abs'] is not None and
                                      same_support_output['max_abs'] <= OUTPUT_MAX_ABS and
                                      same_support_output['relative_l2'] <= OUTPUT_REL_L2),
                 same_support_lse=(same_support_lse['nonfinite_mismatch'] == 0 and
                                   same_support_lse['max_abs'] is not None and
                                   same_support_lse['max_abs'] <= LSE_MAX_ABS),
                 physical_counters=all(v == 0 for v in counter_errors.values()))
    return dict(name=spec['name'], seed=seed, input_rng='cpu' if spec.get('cpu_seed') else 'cuda',
                scale=spec.get('scale', spec['d']**-.5),
                shape=dict(q=spec['q'], k=spec['k'], d=spec['d'],
                                             h=spec['h'], hk=spec['hk']), mask=spec['mask'],
                threshold=threshold, sensitivity_range=[float(sensitivity.min()), float(sensitivity.max())],
                eligible=int(original.eligible.sum()), original_skipped=int(original.skipped.sum()),
                cached_skipped=int(cached.skipped.sum()), bitmap=bitmap,
                same_support_output=same_support_output, same_support_lse=same_support_lse,
                direct_output=direct_output, direct_lse=direct_lse, risk=risk,
                original_vs_cached_projected_state=errors(cached.projected_state, original.projected_state),
                counter_errors=counter_errors, score_storage_bytes=scores.numel()*4,
                gates=gates, passed=all(gates.values()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kernel', type=Path, required=True)
    p.add_argument('--bridge', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True,
                   help='Frozen aggregate results/query_adaptive_v3/configs/frozen_policies.json')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError('SM90 CUDA device required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = json.loads(args.policy.read_text())
    observed = {kind: float(policy['policies']['T_s50'][kind]['log_threshold']) for kind in POLICY_THRESHOLDS}
    if observed != POLICY_THRESHOLDS or digest(args.policy) != POLICY_SHA256:
        raise ValueError('Aggregate T_s50 policy does not match the frozen source')
    kernel = Kernel(args.kernel, torch_library=args.bridge)
    signature = inspect.signature(cached_attention)
    if 'q' in signature.parameters or 'k' in signature.parameters:
        raise AssertionError('Cached executor must not accept current Q or K')
    source_files = dict(qualification=Path(__file__).resolve(),
                        cached_executor=Path(inspect.getfile(cached_attention)).resolve(),
                        integration=Path(inspect.getfile(NativeAttention)).resolve(),
                        junyu_cuda=Path(inspect.getfile(Kernel)).resolve())
    source_hashes = {name: dict(path=str(path), sha256=digest(path))
                     for name, path in source_files.items()}
    specs = [
        dict(name='d256_full_q128', q=128, k=192, d=256, h=4, hk=2, mask='full', kind='global'),
        dict(name='d256_local_q129', q=129, k=257, d=256, h=4, hk=2, mask='local_partial', kind='local'),
        dict(name='d512_full_q128', q=128, k=192, d=512, h=4, hk=2, mask='full', kind='global'),
        dict(name='d512_additive_q129', q=129, k=193, d=512, h=4, hk=2, mask='additive', kind='local'),
        dict(name='native_global_q256_h16_hk2_d512_k512', q=256, k=512, d=512, h=16, hk=2,
             mask='causal', kind='global', scale=1.0, cpu_seed=True),
        dict(name='native_local_q256_h16_hk8_d256_k1280', q=256, k=1280, d=256, h=16, hk=8,
             mask='local_causal', window=1024, kind='local', scale=1.0, cpu_seed=True),
    ]
    started = time.monotonic()
    cases = []
    for index, spec in enumerate(specs):
        for label, threshold in [('all_kept', -math.inf),
                                 ('T_s50', POLICY_THRESHOLDS[spec['kind']])]:
            case = run_case(kernel, spec, threshold, seed=1729+index)
            case['mode'] = label
            cases.append(case)
            print(json.dumps(finite_json(dict(name=case['name'], mode=label, bitmap=case['bitmap'],
                                              gates=case['gates'], passed=case['passed']))), flush=True)
    pruned_cases = sum(case['mode'] == 'T_s50' and case['original_skipped'] > 0 for case in cases)
    result = dict(purpose='fresh_same_state_selector_and_cached_score_output_qualification',
                  policy='T_s50', policy_path=str(args.policy),
                  policy_sha256=POLICY_SHA256, policy_thresholds=POLICY_THRESHOLDS,
                  source_hashes=source_hashes,
                  kernel=str(args.kernel), kernel_sha256=digest(args.kernel),
                  bridge=str(args.bridge), bridge_sha256=digest(args.bridge),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  frozen_tolerances=dict(output_max_abs=OUTPUT_MAX_ABS,
                                         output_relative_l2=OUTPUT_REL_L2,
                                         lse_max_abs=LSE_MAX_ABS,
                                         risk_near_threshold=RISK_NEAR_THRESHOLD,
                                         bitmap_disagreements_allowed=0),
                  cached_attention_signature=str(signature),
                  seconds=time.monotonic()-started, cases=cases,
                  pruned_cases=pruned_cases,
                  passed=all(case['passed'] for case in cases) and pruned_cases > 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+'.tmp')
    temporary.write_text(json.dumps(finite_json(result), indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps(dict(output=str(args.output), passed=result['passed'],
                          cases=len(cases))), flush=True)
    if not result['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
