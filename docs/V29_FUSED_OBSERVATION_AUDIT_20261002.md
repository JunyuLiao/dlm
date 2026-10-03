# V29 fused observation correctness audit, 2026-10-02

Current status (2026-10-02 22:51, UTC-5): all44 GPU cases passed at frozen tolerances, source54163f4e7. See the completion receipt below. The protocol/preparation sections retain the pre-execution history. No speed or task-quality conclusion follows; core implementation is unchanged.

## Actual main path

integration.Attention._fused_bootstrap_observation explicitly calls fused_observe with mu_precision='bf16', not its default 'tf32x3'. It computes wholly-prefix tiles as floor(prefix/64), retains the boundary and canvas in the current FP32-score tail, and uses rank32 in exact mu mode. The kernel's BF16 branch rounds both normalized tile probabilities and projected-V sketch to BF16 for the mu dot product. Dense PV also uses BF16 probabilities/V with FP32 accumulation; output is BF16. These are numerical approximations to FP32 math, not bit-exact output or routing trajectory guarantees.

Root checked the frozen main configuration: fused observation, R6/A64, dense-prefix risk, minus-ln2, carry_first and async route are enabled; min_route_keys is absent (core default0), observe_carried is absent/off and adapter Q grouping defaults128. The original four tests exercised default tf32x3 and did not qualify main's explicitly BF16 mu branch.

## Fixed numerical protocol before GPU execution

Retain all four original separate-STORE comparisons. Add40 independent eager-FP32 oracle cases: prefix lengths0,31,65,844,1793, each with splits1/2, dense-output/observe-only modes and scale=512**-0.5/1.0. Root observed official GLOBAL impl.scale=1.0, reflecting its QK normalization convention; both scalar values are covered. All use batch1, Q256, query heads16/KV heads2 (GQA8), head dimension512, rank32, BF16 Q/K/V, FP32 projected sketch and seed2902+prefix. Cases65/844/1793 have non64-aligned prefix and tail; 65 exercises one complete prefix tile and a boundary tile. Primitive cases0/31 exercise no wholly-prefix tiles; they do not establish core eligibility.

The oracle explicitly repeats KV heads, forms FP32 QK with scaling, and evaluates eager softmax/PV and per-prefix-tile logsumexp/softmax-weighted sketch. TF32 is disabled and matmul precision set highest for these oracle matmuls, then restored in finally. The oracle does not call fused_observe, route STORE, SDPA, or a fused consumer. Test complete [1,Q,H,D] output against full eager attention for output=True; output=False must return None. Test every tail score, every prefix tile's z and all32 mu coordinates, and active/bad flags against finite-input expectations. All original tests remain, including their independent STORE/LOAD routing comparison.

Fixed existing tolerances, chosen before execution and not widened after failure:

| Quantity | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| Tail FP32 scores | 1e-4 | 1e-4 |
| Prefix log-mass z | 1e-2 | 1e-3 |
| Rank32 mu | 1e-2 | 1e-2 |
| Full BF16 attention output | 2e-2 | 2e-2 |
| Active/bad finite-input flags | exact | exact |

These are the same thresholds as the pre-existing tests, including the conservative absolute tolerance for coordinates near zero. They assess numerical correctness of this fixed synthetic distribution; they do not bound arbitrary adversarial input error or guarantee equal skip decisions near a threshold. The original STORE routing test still requires >99.5% skip agreement and exact eligibility; that existing rule is not a BF16 trajectory qualification claim. Failures must be preserved and diagnosed without widening tolerances.

## Zero and short prefix audit

allocate_summary returns None when prefix_tiles<=0. The low-level fused primitive uses dummy summary pointers in this case and can still produce dense output and full tail. The core bootstrap observation explicitly raises when summary is None, so generic prefix<64 is unsupported by this fused core path. There is no unconditional short-prefix fallback. Earlier core native gates (nonzero min_route_keys, excluded route_layers, or a fired gate_dense) can bypass it only when actually configured/triggered. Frozen main has no min_route_keys, and must not be described as the earlier HF AIME E7 plus2K length-gated configuration. Current expanded inventory has minimum prefix at least64; that does not imply generic short-prefix support or prove a particular panel has triggered this failure.

## Verification and next GPU receipt

AST syntax passes. The added parameter product is40 BF16 cases, plus4 original cases (44 total). Synthetic random Q/K do not follow the real model QKNorm distribution; these are scalar/geometry correctness cases, not real-model quality qualification. Local system Python has neither Torch nor pytest; actual pytest collection is therefore pending the own vLLM environment after committed source deployment. Attempt001 deployed the prior24case frozen source but stopped before collection because own vLLM lacks pytest; GPU runs and reserved seconds were0. A separate own runner target now imports pytest9.1.1, Torch2.13.0+cu130 and Triton3.7.1 successfully with CUDA hidden and cuda_initialized=False; qualified vLLM packages remain unchanged. The runner dependencies are used only through explicit PYTHONPATH. The expanded44case source must be newly committed before attempt002 deployment. The isolated command is `python -B -m pytest tests/test_v27_fused_observe.py --collect-only -q` with CUDA hidden; then, after root's frozen SHA and idle authorization, `python -B -m pytest tests/test_v27_fused_observe.py -q` on one idle worker with new own caches. The receipt must distinguish collection/CPU success from GPU results, preserve failure/exit code and complete reserved GPU seconds, and state software/source pin. No timing or speed conclusion follows from this correctness run.

## Completed GPU verification (2026-10-02 22:51, UTC-5)

All44cases pass at the frozen tolerances. Source54163f4e7, reserved32.737210GPU seconds. See `results/v29_20261002/fused_observe_bf16_001/summary.json` for attempts and scope; this supersedes the pending status above. No task-quality or speed claim.
