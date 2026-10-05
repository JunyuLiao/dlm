# v25 aligned16 qualification contract (frozen before any GPU measurement, 2026-09-28T11:25Z)

The candidate is `route_storage=aligned16` (see `v25_corrections.md` and `integration.py`). Method policy, thresholds, A8/R3, bootstrap, Q128/KV64, output numerics and native stopping are fixed. Evidence is gathered in three separately reported levels.

**Level 1: intermediate bits.**
- Stored score bits for the real K: bitwise equal by construction, since only the physical pitch changes (verified by tests).
- Stored prefix summaries z/mu at every BO/A anchor: report bitwise equality, differing-element count, max abs, max relative and max ULP. Summary equality is not required.

**Level 2: behaviour on the tested states.** Parent vs aligned16 are replayed on identical native incoming states from canvas start, for up to 16 reached calls, for bootstrapped M1, M3 and B. Per call we report:
- the phase string;
- logits bitwise equality, otherwise max abs diff and argmax agreement;
- per-layer decision (skip/eligible) equality and the number of differing tiles.

If every call's logits are bitwise equal in every tested sequence, the variant is called execution-equivalent **on those states only**.

**Level 3: numerical variant.** If any logits or decisions differ, aligned16 is a numerical variant. It may enter the pilot only if all of the following hold:
1. no non-finite value or guard/invalid-tile behaviour changes;
2. summary max relative difference ≤ 1e-5 wherever the parent value's magnitude is ≥ 1e-30 (TF32x3-dot reduction scale);
3. every decision difference is reported, with per-call counts.

In that case the parent's quality is **not** reused, and all pilot first outputs are scored independently.

**Economic gate** (before the pilot): the v25 direct profile (`--arm-set v25`, same states, native brackets) must show aligned16 cheaper than its logical parent at BO/A on at least one odd-K LB state, with no slowdown beyond bracket drift elsewhere. All padding copies, sketch pads and allocations are included because they sit inside the timed forwards.

If either gate fails, the pilot is not run and the measured evidence is reported. The parallel-summary builder probe (v25 §5.2) is considered only if aligned16 qualifies numerically but its measured BO/A gain is too small to matter. Thresholds are not changed after seeing results.
