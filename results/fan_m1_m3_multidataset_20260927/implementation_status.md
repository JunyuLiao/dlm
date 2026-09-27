# v20 implementation status — CP0 in progress

Parent e3f14520eea98e45998fb369f9027f3b2707ec4a; new branch
research/fan-m1-m3-cost-refresh-20260927. The v18 campaign is closed and unchanged.
One remote ref check found peer Junyu 053441c6 and Haowei b23f969 unchanged.
Both original H100 UUIDs were idle at preflight. No old controller is relaunched.

| Item | Existing implementation | v20 work / current evidence |
|---|---|---|
| M1 current-output pre-QK | Present in numerical_qk_reuse/integration.py | Scope/numerical qualification pending |
| M3 R2 | Present, GLOBAL plugin hard-coded R2 | New explicit R1/R2/R3 API in progress |
| A/R separation | ScoreCache age-based decisions already present | Anchor8 then R3 decision11 regression tests in progress |
| common fast_t | Fresh/CVM pass flag; historical installers omit it | Sol-A plumbing explicit flag, preserve old defaults |
| native legal all layers | M1 native_mask and v18 fresh adapter exist | Explicit masks still force anchors or fail closed; never fake reuse |
| complete forward per dataset | Missing same-contract table | State-restored v9/v14 harness extension in progress |
| adaptive first84/700 | NOT_RUN | Sol-B freezes task-aware schedule/identities |
| G75/L30 first bitmap | Historical FIXED16 vLLM source identified | Native-adaptive reference separately gated; no old speed multiplied in |
| candidate / M2 | Ball STOP from v18; M2 NOT_RUN | No candidate/kernel expansion |

Timing decision: per-forward event/synchronization replay runs separately from
clean adaptive first/warm requests. This avoids changing E2E launch gaps. A
validated existing prefill-end boundary may still supply decode-generation span.
No request/calls amortization is labeled direct forward.

Fan source: read-only speaker notes on slide yuhan_20260923_fixed in megakernel
confirm reuse QK with current projected V, then hold last skip information and
periodically return to M1. Notes identify historical G75/L30 as FIXED16 and
amortized generation/calls. The named ASR transcript was not found in the local
workspace/Downloads or two exact Drive searches; this missing corroboration
does not block the explicit v20 method contract. No shared slide was edited.
