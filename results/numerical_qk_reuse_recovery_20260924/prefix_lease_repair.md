# Prefix V/norm lease repair (Phase C, code + invariance proof only)

Source finding (v6 prompt, "REPEATED PREFIX WORK"): the numerical decision-
refresh step in `experiments/numerical_qk_reuse/integration.py` recomputed
`v.float()`, a full `matmul` against the Gaussian-32 projection, and the RMS
norm over the **entire** V tensor (encoder prefix + canvas) on every
decision-refresh call (every M1 call, since `decision_interval=1`), even
though the encoder-owned prefix portion of V does not change within a canvas.
`experiments/value_direction_hopper/integration.py`'s `Sketches` class
already solves exactly this for the fresh-kernel (T) path: it leases the
projected-V/norm sketch for the aligned (KV64-floor) prefix boundary, keyed on
`(encoder_epoch, cache_tensor._version, value.shape, aligned_boundary)`, and
only reprojects from that boundary onward (current canvas plus the small
unaligned remainder) when the lease is valid.

## Repair

`Attention.__init__` now constructs `self.sketches = Sketches(adapter,
self.projections, fused=False)` -- the same class, not a reimplementation --
and the decision-refresh block calls `self.sketches.get(layer, v, valid,
prefix - crop)` instead of the inline full-tensor recompute. `crop` is 0 in
the executed model regime (verified in the native-mask audit:
`DynamicSlidingWindowLayer` already caps the stored prefix at
`sliding_window - 1`, so the integration's own additional crop line is always
a no-op), so this passes the full concatenated V tensor, matching what
`Sketches.get` expects. `close()` now also releases the sketches' hooks.
`counters()` reports `projected_current_v_tokens` /
`reused_current_v_tokens` directly from the lease's own cumulative counters
instead of assuming every call reprojects `nk` tokens.

## Invariance proof (not yet a GPU cost measurement)

This is a pure engineering change: it must not alter M1's numerical output
or decisions, only how much redundant work produces them.
`tests/test_numerical_reuse_prefix_lease.py`:

- `test_leased_prefix_matches_brute_force_across_unaligned_boundary_and_changing_canvas`:
  runs 3 decoder calls in one canvas with an intentionally KV64-unaligned
  prefix (65), a changing canvas V each call (simulating token acceptance),
  and asserts the router's output equals a brute-force recompute (the exact
  pre-repair formula) via `torch.testing.assert_close`, and that
  `sketches.reused_tokens > 0` by the second call -- proving the lease is
  actually engaged, not just present and inert.
- `test_commit_invalidates_the_lease_even_with_identical_content`: a new
  encoder forward (simulating a canvas commit) with bit-identical prefix
  values but a new tensor object must still force a full reproject
  (`reused_tokens` unchanged), proving content-based silent reuse never
  happens -- identity/version is the only authority, matching Sketches'
  existing contract.

Both pass locally under CUDA (`pytest tests/test_numerical_reuse_prefix_lease.py`).

## Not yet done

This proves the repair is *safe* (output/phase invariant). It does **not**
yet show it is the *right* explanation for M1/M3's quality collapse, nor does
it have a warm-forward cost measurement -- both require the Phase B 5-cell
diagnostic and Phase C PROFILE, which need the real model on the remote GPU
and have not run yet as of this commit. Do not cite this as the selected
Phase D fix until that evidence lands.
