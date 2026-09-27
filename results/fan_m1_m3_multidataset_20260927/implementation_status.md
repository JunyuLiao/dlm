# v20 implementation and qualification status — first84 running

Parent e3f14520eea98e45998fb369f9027f3b2707ec4a; branch
research/fan-m1-m3-cost-refresh-20260927. Production deployment is immutable
00a2c4d3c93a7f685a4f75fba8ea47991fa5c383 (CP3); auxiliary archive/scoring/parity
helpers are05f99473406ef9b21d75b20d52d8e36b0124ddd1 (CP5), with child cwd pinned
to CP3. Earlier v18 CP4 remains closed. Junyu053441c6/Haowei b23f969 unchanged
at the single preflight ref check; no peer branch/environment changes.

| Item | Implemented | Qualification / measured evidence |
|---|---|---|
| M1 current-output pre-QK | Existing engine, explicit v20 PREQK mode | Actual QKV/bitmap output checks against pinned v11 reference envelope on real states |
| M3 R1/R2/R3 | Explicit v20 API; old GLOBAL R2 label preserved | CPU phase tests; actual A/D/H layer calls and operator probes; R1=M1, R>=A held reduction |
| A/R separation | Existing score age and decision age, A8 anchors reset decisions | R3 anchor8 then decision11 tested; numerical/decision ages reported |
| common fast_t | Flag threaded to all historical installers; old defaults preserved | GPU fast/slow T equivalence among11 qualification tests per host; public shared optimization |
| native legality | ALL and GLOBAL native-mask paths | Both scopes measured; explicit is_causal=False/mask=None qualified, invalidation not suppressed |
| current attention output | Retained tiles compute current QK/PV | No cached-final-weight fallback, no QKV projection skipping; same-support independent reference |
| complete model forward | State-correct one-canvas capture/replay | screen002 N4 bothscopes/26arms; selected001 native-reached long sequences; RULER natural4call supplement |
| whole denoising step | Native wrapper includes T/sampler/stop | Both boundaries measured for selected main7+G75, three blocks, no accepted new JIT |
| physical work | Separate untimed exact input/output-matched twins | QK/PV legal-pair counts, partial tiles, GLOBAL/LOCAL/prefix/canvas, A/D/H ages; GLOBAL denominator only routed5layers |
| samehost/crosshost bridge | Immutable frozen config/manifest/source checks | All8 bridge arms exact tokens, calls, termination, router phases across original H100 UUIDs |
| initial-prefill-excluded span | Existing two-event InitialPrefillTimeline | ON/OFF parity passed native/R2/G75 bothhosts. Device-clock span includes host gaps/later commits, NOT decode wall/TBT |
| adaptive first84 | Frozen task-aware sevenarm first/warm worker | RUNNING42executions perhost, first outputs immutable; not yet a scored quality/E2E claim |
| remaining700 | Same frozen panel/policy/source | NOT_STARTED, follows first84 publication and remaining budget |
| G75/L30 reference | Explicit ALL native-Q128/KV64 transplant | Direct and operator+bridge+timeline qualified; optional100answer executions NOT_STARTED; not oldvLLMgrouping |
| M2 / ball / CVM expansion | No new implementation | M2 NOT_RUN; ball stopped beforeCUDA; noCVM expansion |

Selected answer point BEFORE task outcomes: GLOBAL_ONLY_NATIVE_LOCAL, P0,
generic Triton. ALL loses at the measured point. P1 does not yield repeatable
cross-family cost improvement. GLOBAL consumer aggregate difference is ~0.03%,
not a claimed backend advantage. Exact policies/masks/stopping in method_contract
and frozen binding; no output cap, seed, temperature or native stop changes.

Direct profiling has per-call synchronization and runs separately from accepted
adaptive request timing. Sequence sums and outer replay spans (which include
fixture restoration) are separate quantities. Common-input replay does not
constitute a naturally generated answer. RULER missingN16 is preserved and
reconciled with actualN4; no forced calls. Qualified complete gate is in
qualified_numerical_gate.json and bridge001_compare.json.

Source/config value hashes differ across hosts only for the previously qualified
private ATen relocation/buildJSON; active code and kernel hashes match. Model
index filenames are repacked with previously verified identical tensor identity.
Host paths enter config fingerprints, so different fingerprints do not claim
bit-identical paths. No cross-machine absolute latency pooling is used.

The read-only Fan speaker notes corroborate periodic M1 intent. Named ASR
transcript remains unavailable; this is documented missing corroboration.
Shared slides/messages untouched. Drafts remain local. Runnable checkpoints are
committed; GitHubpush is currently blocked by automatic review pending explicit
user confirmation of the named remote destination.
