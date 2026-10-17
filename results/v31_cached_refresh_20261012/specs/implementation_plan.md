# Frozen implementation plan

1. Run the normal V31 dense observation on the initial call and every existing
   dense/progress refresh.  Cache raw per-row prefix-tile log-mass, the current
   fixed-budget map, zero ages, and optional diagnostic value/output tensors.
2. On a cheap event, execute the held-map sparse FA4 output first.  Compute QK
   statistics only for retained prefix tiles, combine them with cached
   statistics for discarded tiles, normalize over complete prefix support, and
   select the next map.  The new map is used only by the next attention call.
3. `global_topk` ranks the hybrid qblock-max score globally.  `bounded_swaps`
   uses a total top-M frontier and exchanges at most B=8 tiles, preserving K.
   Physical tile ids are stable tie breakers.
4. The primary arm is bounded M=256, no decay, refresh period two.  The decay
   arm uses lambda `log(2)/2` and floor `0.25`.  Phase protection, C-gate
   ranking/ties, online/full-information ties, strict/filled intersections,
   and periods one/four/progress are separate arms.
5. Non-finite retained statistics or drift above the frozen operator threshold
   `10,000,000` keep the current map for the current output and schedule a
   dense refresh next call.  This threshold is fixed from synthetic stress
   replay and is never fitted to target answers.

The implementation module is
`experiments/numerical_qk_reuse/v31_cached_refresh.py`.  Its
`CachedRefreshState` is the selector-only state machine; the adapter's FA4
consumer remains responsible for native attention output.
