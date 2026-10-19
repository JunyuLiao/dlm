# Corrected HumanEval164 K256/K128 evaluation

This continues the user's authorized reproducibility-first evaluation. Baseline
qualification must complete before either full method matrix. Preserve the
original K256 results, all failed diagnostic attempts and older LongBench pauses.
No selector redesign, accuracy tuning, eager fallback or sampler-formula change.

## Identity and numerical policy

Use the original frozen20e6c0be selector execution, official164 tasks and exact
thinking-on/off manifests, seed42, output8192 and native adaptive sampler settings.
Apply the qualified numerical-policy worker explicitly to vLLM's compiler and
the three unchanged native compiled helper Python functions. Retain native FULL
dense fix51994, PIECEWISE dense and all-kept FA4 controls. Check actual selected
kernel settings and zero measured CUDA captures; a global flag is not proof.
Reference the completed baseline runtime qualification marker in each protocol.
The qualified policy disables horizontal combo fusion and its unsupported
benchmarking in this pinned runtime. Disclose that numerical/compiler choice;
measure all controls under the same policy. Preserve native FULL/PIECEWISE graphs.
Read per-canvas interval counts from cumulative CPU output metadata already
returned by engine.step; no extra device transfer, synchronization or forward.

Freeze separate new protocols and attempt directories for K256 and K128. For
every query-head/Q128 tile, select min(K/64, complete-prefix-block-count) blocks;
canvas/boundary tiles and all LOCAL layers remain dense. K256 means4 selectable
KV64 prefix blocks; K128 means2. V2 remains explicitly variable-budget, using
its online running-softmax mass-preserving Gaussian32 criterion. V3a is the
existing full-row singleton ranking. Full-value swaps and direction-guarded
residual retain the existing device_exact_v1 full-V512 layer29 implementation,
with V31 at the other GLOBAL layers. Residual changes held outputs explicitly.

## Gates, work calibration and independent replication

1. Add meaningful K2/K4 and short-prefix/tail budget tests to the generalized
   worker, retaining the existing fast-kernel/native-oracle tests. Run relevant
   H100 checks once, with receipts and source fingerprints. No nominal-budget
   claim substitutes for selected-mask accounting.
2. Calibrate only with attention work. K256 first checks the previous frozen
   per-mode threshold; expand calibration only if necessary. K128 starts a fresh
   threshold search. Retain length-stratified24-question weights and the declared
   sparsity tolerance0.25pp plus kept-fraction tolerance1%. No accuracy or timing
   enters threshold choice. Generation remains native with output8192.
3. Verify and, if necessary, refine first-canvas matching on all164 tasks using
   only V31/V2 work-calibration requests. Freeze the final threshold before the
   eight-method primary panel. This transductive work calibration is explicitly
   separate from final method accuracy. Inspect full-trajectory absolute work too;
   first-canvas matching is not a whole-trajectory work match.
4. Run native qualification for all requested methods and controls, both passes
   and thinking modes, with fixed-budget/layer-scope/kernel receipts and bounded
   native same-state refinement-oracle checks. Fail early on malformed receipts.
   Bound oracle work to the first two anchors per request; independent full-V512
   tests include the long-prefix K2 case. Primary requests contain no oracle.
   Use the six metadata-selected diagnostic questions, seed42 first, followed
   by their predeclared seed43 confirmation. Seed43 remains descriptive; no
   seed-robust superiority is inferred from six previously examined questions.
5. Keep two82-question primary shards. Shard001 begins with those same six
   questions in the qualification order; assign its remaining76 and shard002's82
   deterministically without scores. Identical per-family/pass warmups and the
   frozen final threshold make this prefix an independent engine replication of
   qualification, including request/arm order. Each task still contributes one
   seed42 primary result per method/pass. Compare seed42 prefix outputs, N/C/T
   and correctness exactly; report technical repeats separately from seed43.
6. Generate all164 primary tasks for V31, V2, V3a, full-value swaps, directional
   residual, FULL dense, PIECEWISE dense and all-kept, both passes/modes. Preserve
   audit/clean separation. Associate audit work with clean latency only after
   per-item output/trajectory agreement is verified; otherwise retain a failed
   qualification and separate populations, without a reliable speedup claim.

The prefix replication avoids eight extra cold engine launches. In the original
study,32 primary engines consumed9751.45s,16 small qualifications2725.18s, and
calibration/refinements2555.52s. The design spends calibration on V31/V2 before
the complete panel and reuses its established scheduling machinery; it does not
select questions or configurations by observed correctness.

## Budget, resumability and reporting

Charge both panels to the existing shared43200s ledger: diagnosis10800,
K25616200 and K12816200 under the bound allocation addendum. Include startup,
warmup and failed attempts. One worker/H100; immutable attempts, explicit stage
markers, bounded per-job reservations and a hard phase gate. Inspect progress
about every30 minutes. Do not overwrite or resume a killed native attempt in
place. Do not silently borrow or expand a phase budget.

Official scoring plus independent rescoring; sanitized per-item metrics and
paired question-bootstrap uncertainty. Report pass@1, transitions, N/problem,
N/C, C, T, tails/caps, S/W/S:N, actual GLOBAL/LOCAL/overall tile accounting,
first-canvas and full-trajectory work, and complete selector/correction envelopes.
Keep instrumented envelopes separate from clean marginal-cost claims. Include
technical reproducibility across independent processes/passes and the bounded
extra seed, a paper table/paragraph, source/protocol identities and completion
markers. New negatives are retained; no favorable method ranking is promised.
