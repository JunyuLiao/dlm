"""Official dense smoke test for the expansion models (no timing claims): start the model's official SGLang engine in
process with its documented dLLM algorithm and FlashInfer attention, generate for one public toy prompt, and record
that it runs, the engine settings, latency and token counts. The answer text is kept only as a correctness check of
the toy prompt.
usage: python v27_sglang_dense_smoke.py MODEL_DIR DLLM_ALGORITHM OUT_JSON [DLLM_ALGORITHM_CONFIG]
"""
import json
import sys
import time


def main():
    model_dir, algorithm, out_path = sys.argv[1:4]
    algo_cfg = sys.argv[4] if len(sys.argv) > 4 else None
    import sglang as sgl
    import torch
    kw = dict(model_path=model_dir, dllm_algorithm=algorithm, attention_backend='flashinfer', trust_remote_code=True,
              mem_fraction_static=0.8, max_running_requests=1, random_seed=0)
    if algo_cfg:
        kw['dllm_algorithm_config'] = algo_cfg
    t0 = time.perf_counter()
    engine = sgl.Engine(**kw)
    load_s = time.perf_counter() - t0
    prompt = 'Compute 17 * 23. Reply with the number only.'
    rec = dict(sglang=sgl.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(), model=model_dir.rsplit('/', 1)[-1],
               dllm_algorithm=algorithm, dllm_algorithm_config=algo_cfg, attention_backend='flashinfer', load_s=round(load_s, 1))
    runs = []
    for i in range(3):
        t = time.perf_counter()
        out = engine.generate(prompt, sampling_params=dict(max_new_tokens=64))
        runs.append(dict(latency_s=round(time.perf_counter() - t, 3),
                         completion_tokens=out.get('meta_info', {}).get('completion_tokens'),
                         contains_391='391' in out.get('text', '')))
    rec['runs'] = runs
    engine.shutdown()
    json.dump(rec, open(out_path, 'w'), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
