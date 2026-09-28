# v25 continuation (aligned route storage) — CPU work complete, GPU awaiting renewal

Authority: user v25 handoff `dllm_claude_v25_route_preparation_20260928.md`; started at `9ded5f80`.
GPU: the 09:30Z renewal expired; `E:/dlm/gpu_authorization.json` (current=null) is the only deadline source,
read by every private launcher via `E:/dlm/gpu_auth.py`. Budget 13,278.8 / 21,600 s, 408 / 512 executions.

## Done (CPU)
- `R/v23_bootstrap6/v25_corrections.md`: nested projection timer, alignment facts (KDIV, ~2.2x/key, no ULP
  measured in v24, producer vs copy), scope, single authorization, v23 qualification selector scope.
- Variant `route_storage=aligned16`: producer unchanged at real K; FP32 bits copied to 16-key pitch (-inf
  tail) for the score cache; sketch padded per route; tiles/prefix/ref/legal/consumer on real K.
  Tests `tests/test_v25_aligned_storage.py` (CPU 43 passed on both hosts).
- Qualification mode `--aligned-qualification`; profile `--arm-set v25`; frozen pilot
  `v25_aligned6_pilot_56d3b98fe1261756` (LB 2x2, AIME 1x2, RULER 2x1; 90 executions).

## Next (after the user renews a GPU window)
1. Deploy HEAD, bind, run `--aligned-qualification --qualify-tests` per host.
2. `python -m scripts.v24_profile --arm-set v25 --v21-profile-config <bindings>/profile.json --out ...`.
3. Pilot only if 1-2 qualify (predeclared: no nonfinite/guard change; decisions and logits compared per call;
   any divergence reported and the variant then treated as a numerical variant scored independently).
