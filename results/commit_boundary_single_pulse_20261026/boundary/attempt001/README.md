# Observation-only exact H replay across native commit

Run references on prior H cells0,1,3: minimal A, boundary A, repeat A. Require
historical output,N,C,T,CPU/CUDA RNG and model-forward equality. Reuse EXACT
historical banks,anchors,supports,aliases and budgets for ten unique pulses,
then A_after eachgroup. Full requests finish naturally; current-QKV/GQA/error<=.03
and actual mask/protected-support checks stay. No RNG realignment, no restoration.
Capture target-window sampler state plus own native commit/model input and next
initialization/first ordinary call, active GLOBAL and LOCAL cache spans, inputs,
outputs,valid lengths,RNG/counters,emitted canvas/prefix hashes. Numerical copies
canonicalize signedzero/NaN only for hashing. Inactive history slots are excluded.
Caches are compared in active logical order, not allocation capacity.

Pre-pulse fingerprints and aligned draws stay exact; branch-dependent later RNG
is allowed. Source/observer qualification must reproduce the historical request
before interpreting any phase differences. Complete boundary observation does not
prove RNG is the sole causal mediator or establish expected quality/work gains.
