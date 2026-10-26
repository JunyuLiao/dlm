## 2026-10-26 UTC — Native support discovery complete; readout rule does not advance

Verified results checkpoint `ed6730cf5ca83f18f9b3124976c2305fbb00c644`; generation source `c3f3fcdf`, protocol
`10398a912c355dabf3d6511475a7bdf0fb3c3434c0e7055e439a724fd5bc0d70`.
Read [the complete handoff](results/native_support_discovery_20261026/summary.md)
and [method decisions](results/native_support_discovery_20261026/method_decisions.json).

The new authorized study completed the 368-control census, 216 fresh baseline
profiles on 36 questions, and all 202 frozen single-pulse branches on six
independent development questions with seeds 42/43. Local calls: 52 fewer,
102 unchanged, 48 more. Of 131 executed equal-work/error/mass-matched pairs,
66 differ in remaining calls. This supports propagation effects, not a useful
selection rule or reuse mechanism. Every target canvas stops adaptively.

Readout scores 10/12 in every configuration, matching all-kept/current-error;
FULL/PW score 9/12. One sham support regresses a correct LongBench answer to a
wrong capped answer. No readout configuration passes repeated opportunity and
identifying prediction gates. No strict lead over both CE/sham repeats on both
seeds for any question. Confidence/instability and budget failures stay recorded.

All 31 focused CPU cases pass; source/model/runtime/scorer checks pass. All 202
prefix/state/current-QKV/support and immediate RNG checks pass; no extra
in-request model forwards. Counts include 523 complete native requests,
117,399 model forwards and 1.903 allocated GPU-worker hours, including failures.
The final H100 is idle; no worker remains running. No resource ceiling, paused
campaign resume, production kernel work or push. Nine over-limit identities
remain blocked. Six confirmation questions remain unexposed to sparse treatment.

Next: independent review of matched contrasts and weak readout forecasts. Do
not launch confirmation, Stage B/C, an exhaustive MarginGuard oracle or kernel
optimization. Reuse remains parked. Historical mixed/negative reports are intact.

