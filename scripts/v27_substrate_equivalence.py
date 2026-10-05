"""v27 correctness check of the piecewise_v1 substrate: the same decoder-call inputs (captured from a real native
request with the D_fa4_allkept control, cache deep-copied at capture) evaluated by the eager model forward and by
the compiled substrate forward (LOCAL attention graphed, then LOCAL eager). Everything the substrate compiles is
outside GLOBAL attention, which stays eager for every arm, so this checks the substrate for all arms.
Reports per captured call: eager-vs-eager repeat (must be exact), eager-vs-compiled max/mean |logit diff|,
argmax agreement over the canvas, mean KL(eager || compiled) of the softmax.
Noise floor: the same eager forward with cuBLAS bf16 reduced-precision reduction toggled (a legitimate numerics
perturbation). BACKEND=eager compiles with dynamo's eager backend (same graph split, eager kernels): any
difference from eager then is a capture/split bug, not numerics.
usage: python -m scripts.v27_substrate_equivalence MODEL MANIFEST ROW_INDEX [CALLS] [BACKEND]
"""
from __future__ import annotations

import copy
import json
import sys


def main(argv=None):
    argv = argv or sys.argv[1:]
    model_path, manifest, row_index = argv[0], argv[1], int(argv[2])
    capture_at = {int(x) for x in (argv[3] if len(argv) > 3 else '0,1,4,9').split(',')}
    backend = argv[4] if len(argv) > 4 else 'inductor'
    import torch
    from dllm.models import create_adapter
    from dllm.models.base import GenerationRequest
    from experiments.numerical_qk_reuse import v27_fast_dense as fast
    from experiments.numerical_qk_reuse import v27_substrate
    row = json.load(open(manifest))[row_index]
    adapter = create_adapter('diffusion_gemma', model_path, device='cuda', precision='bfloat16').load()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = adapter.model
    eager_forward = model.forward
    captured, calls = {}, [0]

    def clone(value):
        return value.clone() if torch.is_tensor(value) else value

    def capture(*a, **k):
        index = calls[0]
        calls[0] += 1
        out = eager_forward(*a, **k)
        if index in capture_at:
            kw = {key: (copy.deepcopy(v) if key == 'past_key_values' else clone(v)) for key, v in k.items()}
            v27_substrate.compact_sliding_cache(kw.get('past_key_values'))
            captured[index] = (tuple(clone(x) for x in a), kw)
        return out
    request = GenerationRequest(prompt=row['prompt'], max_new_tokens=256, temperature=0.0, seed=101,
                                extra={'thinking': bool(row.get('thinking', True))})
    kernel_swap = {}
    with fast.install(adapter, {'control': 'D_fa4_allkept'}, fast.CONDITION):
        model.forward = capture
        try:
            adapter.generate(request)
        finally:
            model.forward = eager_forward
        def compare(test, ref):
            diff = (test - ref).abs()
            kl = torch.nn.functional.kl_div(test.log_softmax(-1), ref.log_softmax(-1), log_target=True,
                                            reduction='none').sum(-1).mean()
            return dict(max_abs=round(float(diff.max()), 5), mean_abs=round(float(diff.mean()), 6),
                        argmax_agree=round(float((test.argmax(-1) == ref.argmax(-1)).float().mean()), 5),
                        mean_kl=float(kl))
        eager = {}
        results = []
        with torch.inference_mode():
            for index, (a, kw) in captured.items():
                first = eager_forward(*a, **kw).logits.float()
                again = eager_forward(*a, **kw).logits.float()
                eager[index] = (first, bool(torch.equal(first, again)))
            flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not flag
            for index, (a, kw) in sorted(captured.items()):
                perturbed = eager_forward(*a, **kw).logits.float()
                results.append(dict(call=index, comparison='eager_noise_floor_bf16_reduction_toggle',
                                    **compare(perturbed, eager[index][0])))
                print(json.dumps(results[-1]), flush=True)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
            # second noise floor: GLOBAL attention through HF's own SDPA path instead of FA4 (both exact)
    with torch.inference_mode():
        for index, (a, kw) in sorted(captured.items()):
            native = eager_forward(*a, **kw).logits.float()      # no plugin: HF SDPA for GLOBAL layers
            results.append(dict(call=index, comparison='eager_noise_floor_global_sdpa_vs_fa4',
                                **compare(native, eager[index][0])))
            print(json.dumps(results[-1]), flush=True)
    with fast.install(adapter, {'control': 'D_fa4_allkept'}, fast.CONDITION):
        v27_substrate.install(model, backend=backend)
        for mode in ('graph', 'eager'):
            v27_substrate.set_local(model, mode)
            with torch.inference_mode():
                for index, (a, kw) in sorted(captured.items()):
                    for _ in range(3 if backend == 'inductor' else 1):      # compile, record, replay
                        torch.compiler.cudagraph_mark_step_begin()
                        compiled = model.forward(*a, **kw).logits.float().clone()
                    ref, repeat_exact = eager[index]
                    results.append(dict(call=index, comparison=f'substrate_{backend}', local=mode,
                                        eager_repeat_exact=repeat_exact, bitwise_equal=bool(torch.equal(compiled, ref)),
                                        **compare(compiled, ref)))
                    print(json.dumps(results[-1]), flush=True)
    print(json.dumps(dict(done=True, prompt_tokens=row.get('prompt_token_count'), torch=torch.__version__,
                          identity=v27_substrate.identity(model))), flush=True)


if __name__ == '__main__':
    main()
