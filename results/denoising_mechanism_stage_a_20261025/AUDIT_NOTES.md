# Interpretation and implementation audit notes

1. The frozen README describes mean paired log ratios over routing units within
   question. The simultaneously frozen executable instead divides arithmetic
   mean unit losses, then aggregates question log ratios. The executable
   decisions are preserved. Closeout reports both calculations in
   `aggregation_sensitivity.json`; neither is used to change selected supports.
   Identification coverage is independent of this averaging discrepancy.

2. Small prefixes structurally limit the bank. With two or three wholly-prefix
   KV64 blocks and a fixed one-block deletion, four alternatives cannot exist.
   Guards remove additional choices at early layers. Such states remain
   non-identifying; no deletion reallocation or guard relaxation occurs.

3. Public banks retain every candidate's position-wise bias magnitude, score,
   mass, physical support and matched-pair signed-vector cosine. Ignored private
   files retain full position-wise signed vectors for the selected supports,
   plus all-candidate mean bias directions, compressed to FP16. They do not
   provide lossless full-vector replay for every unselected candidate. All
   prediction losses are calculated from the live FP32 diagnostic state before
   it is released. This storage limitation must accompany any later vector audit.

4. `48-call caps` means canvases that reach 48 denoising calls. Confidence and
   stability can also hold on that final call; reaching the cap alone does not
   attribute termination exclusively to it. Output-token caps and missing final
   responses are separate fields. Native commits define canvases, and an
   uncommitted denoising tail remains included in N.

5. Native confidence/history/convergence flags are read directly. Entropy is
   reconstructed from actual returned scaled logits in float32 and is reported
   separately from native flags. No exact compiled acceptance count is claimed.
   Top-two values provide an unsigned auxiliary native margin; there are no
   sparse-treatment signed reference-winner margins unless A2 actually runs.

6. A1 uses counterfactual attention operators along an unchanged common trajectory.
   Its candidate deletion budget is a physical diagnostic control, not removed
   work from model execution. Every A1 model consumer remains all-kept. Clean
   W/S/S-per-N and native AA/SA/AS/SS effects cannot be inferred from these data.

7. The initial launcher ended with the interrupted interactive turn. Its already
   running FULL worker survived and finished. Recovery observes the original
   identity and launches only unstarted jobs. FULL has a completion marker and
   exact counted forwards but no recoverable parent wait status; allocation is
   disclosed as worker duration and a polled launch-to-exit upper bound.
