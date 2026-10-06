# Algorithm and execution handoff

This document is the agent handoff for the value-aware and query-sensitivity
protection work. It intentionally omits collaborator-specific hosts, paths,
preferences, and private coordination procedures. Repository source and frozen
experiment records are authoritative.

## Scope

- Model: DiffusionGemma-26B-A4B with the native adaptive denoising sampler.
- Serving path: vLLM 0.30.0 with the pinned SM90 FlashAttention-4 path.
- Attention geometry: 128 query rows × 64 key tokens per physical tile,
  native GQA and structural masks.
- The current V31 adapter sparsifies the five GLOBAL layers only. LOCAL layers,
  prefill, canvas commits, and the sampler remain native dense execution.
- A reported sparsity number must say whether it is GLOBAL prefix sparsity,
  LOCAL sparsity, or count-weighted overall decoder-attention sparsity.
- The next requested arm is LOCAL sparsity as well. The pinned FA4 SM90
  consumer can combine `block_sparse_tensors` with the native bidirectional
  `window_size_left/right=(1023,1023)`, so no separate outside kernel is needed
  for a first implementation. The current adapter does not route LOCAL layers
  or pass that window through its sparse consumer. A correct arm must build
  128x64 maps over structurally window-eligible tiles, preserve the exact
  element-level window mask at boundary tiles, and report LOCAL receipts
  separately from GLOBAL receipts. Do not count the existing `LOCAL_KV_BUDGET`
  metadata as achieved sparsity until this path is active and audited.

## Value-aware routing contract

The Gaussian32 value-aware method is a separate routing candidate and must keep
its own implementation identity when compared with temporal reuse or MAGE:

1. Build one fixed Gaussian/sign/identity projection basis per native KV head,
   using the pinned projection seed and FP32 construction. Record the basis
   identity; do not regenerate it per step.
2. Form the current QK scores and valid mask, then compute the candidate
   block's projected value contribution. Routing information is available only
   after the current QK/softmax work.
3. Compare the candidate contribution with the retained running output. A
   skipped block leaves that retained state unchanged; the first valid support
   and threshold ties are retained.
4. The physical vote is conservative over valid query rows. Only a complete
   unchanged prefix block may reuse a cached sketch. Equality of the relevant
   K/V state and validity mask is required before reuse.
5. The native output remains the original BF16 V attention output. A projected
   diagnostic or a dense-shaped masked PV computation is not evidence that the
   corresponding full-dimensional arithmetic was physically removed.

The Gaussian32 operator therefore must be reported separately from the V31
cached-QK/MAGE selector. Do not claim a value-aware accuracy or kernel speed
gain without matched dense controls, physical tile counts, and a clean timing
pass on the same runtime.

## Query-sensitivity protection (C gate)

The gate is causal: a completed sampler result can affect the next routing
decision, never the call that produced that result. For each canvas row keep:

- `q = gamma_q*q + (1-gamma_q)*(1-accepted)`, initialized to `1`,
  with `gamma_q = 0.65`;
- `r = (r+1)*accepted`, reset when the top-1 token flips;
- `u = sqrt(max(1-p_top1, 0))`;
- `settledness = (1-exp(-r/tau))*(1-q)*(1-u)`, with `tau = 2.5`;
- `s = clip(1 + beta*(1-settledness), 1, 1+beta)`, with `beta = 3`.

The accepted mask must be recomputed from the sampler's own temperature-scaled
logits and entropy-bound rule. Reset all gate state at a canvas boundary. The
first call of a canvas uses maximal protection. A trigger or row weight must
be named explicitly; the settledness trigger at `0.15` and the accepted-fraction
clock are different ablations. The tested row-weight variants are not a free
accuracy improvement: fixed budgets can remove tiles needed by rows that are
already settled.

## Current V31 execution variant

The current integrated baseline uses the same paged FA4 consumer and the
progress-aware re-observation path:

- `qblock_max` selection, 64-key tile granularity;
- The completed AIME26 reproduction used a `4096`-token GLOBAL budget. The
  realized pooled prefix was 3505.6 tokens, so the next half-realized-prefix
  candidate is `MAGE_K=1728` tokens (27 64-token tiles). This is a budget
  candidate; its achieved sparsity must be measured from receipts after a full
  run rather than assumed to be exactly 50%.
- first exact call used for selection, with first-call carry enabled;
- settledness trigger `0.15`, sticky log-share bonus `1.386`;
- FA4 selection, fused logit statistics, chunked DP build, Triton K/V copy and
  LSE merge, and the pinned FA4/local fixes;
- LOCAL budget requests are metadata until a LOCAL routing path exists. In the
  present adapter LOCAL remains dense, so its achieved sparsity is zero and the
  512-token request must not be presented as an effective local skip budget.
- For the planned LOCAL sparse arm, use the same FA4 block-sparse consumer with
  `window_size=(1023,1023)` and a separately frozen local selector/budget. The
  local prefix window is position-dependent at the first and last query rows;
  a map that simply reuses GLOBAL prefix tiles changes the native mask and is
  invalid. Keep an all-kept LOCAL control and a native LOCAL control in every
  timing/accuracy comparison.

Keep the MAGE/V31 arm distinct from Gaussian32. If the value-aware operator is
added to this path, freeze the operator, projection seed, state scope, and
budget as a new named arm and retain the existing arm as its control.

## Local H100 execution

The current server has one NVIDIA H100 80 GB GPU. Run one worker at a time with
CUDA device 0. Use the repository root `/home/exouser/ljy/dlm` for source and
the user's writable private directory `/home/exouser/dyh` for manifests,
caches, raw generations, and run logs. Do not alter shared environments or
other collaborators' files. When the execution sandbox hides CUDA devices,
launch the approved GPU command with the server's elevated execution mode;
verify `torch.cuda.is_available()` and the H100 identity before starting.

Use the local DiffusionGemma snapshot and the pinned tokenizer/model revision.
Build the AIME26 manifest from dataset revision
`79037aebdb6580008fb960d17cb21fd3099083e3`; keep prompts, gold answers,
token IDs, raw completions, and credentials private. The final public record
contains only hashes, sanitized receipts, and aggregate measurements.

For the requested confirmation run:

1. Freeze the source commit, model revision, prompt manifest, sampler settings,
   and cell list before generation.
2. Use all 30 AIME26 problems with seeds `42, 43, 44`. Keep the native canvas
   length 256, maximum 48 denoising calls per canvas, confidence threshold
   `0.005`, stability threshold `1`, entropy bound `0.1`, and the native
   `0.8 -> 0.4` temperature schedule. Thinking and output budget must match the
   pinned V31 protocol.
3. Run a one-cell dense and sparse smoke first. Check finite output, scorer
   identity, GLOBAL-only routing, the C-gate counters, selected/held tile
   counts, and no unexpected CUDA graph capture.
4. Run matched dense and sparse cells on the same H100, then perform a clean
   timing pass with routing diagnostics disabled. A timing pass that includes
   selection traces is diagnostic, not the headline latency result.
5. Require completion markers for every arm and seed before aggregation. A
   partial shard is progress, not a result.

Report per arm and pooled across seeds:

- accuracy on all 30 problems and the per-seed counts;
- total denoising calls and calls per canvas;
- GLOBAL, LOCAL, and count-weighted overall physical sparsity;
- final end-to-end time (prefill + decode) and matched dense speedup;
- decode-only time and matched dense speedup;
- observation, re-selection, C-gate, fallback, and dense-call counters.

Speedup is always measured as the dense time divided by the sparse time on the
same cell set. Never infer it from the fraction of skipped tiles. Preserve
native stopping, acceptance, and re-noising behavior; do not add hidden
forwards or silently change the scorer.

## Required checks before publishing a result

- Unit tests for the value-aware projection/routing and C-gate state update.
- A receipt proving the intended arm and kernel executed, including projection
  identity, trigger parameters, GLOBAL/LOCAL layer coverage, and physical tile
  counters.
- Matched prompt, seed, model, budget, and sampler settings across arms.
- Source and environment fingerprints, GPU model, and run completion markers.
- Separate timing and audit passes, with no newly captured CUDA graphs in the
  timed pass.

Do not merge peer branches, copy peer environment claims, or attribute a peer
kernel to this study. Cite peer algorithms as peer work and label any port or
integration as a new, separately audited arm.
