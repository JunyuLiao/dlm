# v25 corrections to v24 component-level wording (append-only)

The v23 answers and the v24 direct full-forward / denoising-step profiles are unaffected. These corrections narrow the component-level interpretation only.

1. **The projection proxy timer was nested.**
   - `_observation_components` timed `_project_full(...)` with `_event`, but `_project_full` itself runs its own `_event` and then does diagnostic reductions and host scalar reads.
   - The "full current-V projection" column therefore includes synchronizations and checks. It is not a pure production projected-V price, nor the price of the leased `Sketches.get`.
   - The column is relabelled as an upper-bound proxy. It is not re-measured by rerunning the old panel.
2. **Alignment facts, stated precisely.**
   - K = 13,703 has KDIV = 1; K = 17,540 has KDIV = 4. The same-state padded probe is the causal evidence; cross-host per-key ratios are not.
   - 6.6 ms / 13,703 keys vs 3.9 ms / 17,540 keys is about **2.2×** per key, not 2.6×. Raw measured times are preferred to any universal odd-K multiplier.
   - The v24 probe recorded boolean summary equality only. It did **not** measure ULP/abs differences, so "last-ulp" was an unmeasured description. v25 measures max abs, relative and ULP.
   - The padded score **producer** (zero-padded K) is a separate perturbation from **copying** an already computed score tensor into padded storage. v25's candidate is only the copy (`route_storage=aligned16`); the producer is unchanged at the real K.
   - The component probes rebuild current-score observations at captured states. They are BO/A-like evidence, not the cached D/H trajectory.
3. **Scope.** Matched B being cheaper per forward does not by itself defeat M3: B omits redecision and had different quality. No M3 win is claimed either. The paper question is a quality–time frontier.
4. **Authorization.** `STATE.json` kept a stale top-level 05:52Z stop next to a nested 09:30Z renewal, and one RULER launch was refused by the stale value. There is now a single coordinator file, `E:/dlm/gpu_authorization.json`, read through `gpu_auth.deadline()` by every private launcher. It refuses absent or expired windows, and the history keeps both past windows. The 09:30Z window has expired; `current` is null until the user renews.
5. **v23 bootstrap qualification scope.** The v23 bootstrap parity stage built M1/M3 from the diagnostic arm with the default `legacy_recompute` selector. It qualified native logits at calls 0/1 and the A/D/H schedule, but not the prefix-summary path used by the scored panel. v25's aligned16 qualification builds all arms with the panel's selectors.
