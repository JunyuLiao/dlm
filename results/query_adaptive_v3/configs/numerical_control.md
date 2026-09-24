# Frozen v3 versus new v4 numerical control

The new kernel adds an optional FP32 `[batch, query_position]` sensitivity
pointer and multiplies each row risk before the physical 128×64 tile maximum.
With the pointer null, it implements the unweighted rule. With all ones,
v4 routing decisions and BF16 outputs match the v4 null-pointer call exactly.

On a deterministic synthetic H100 state (batch 1, four query heads, two native
KV heads, 256 queries, 512 keys, width 256, projection rank 32), the frozen v3
and v4 kernels had identical physical skipped-tile masks at log thresholds
`−∞`, `−1.1`, `−0.18`, and `0`. Their BF16 attention outputs were not bitwise
identical: maximum absolute difference was `0.0009765625` at each threshold.
The added branch can change compiler scheduling/rounding even when the
sensitivity pointer is null.

In the two-prompt model smoke, native dense reproduced the archive on 2/2
prompts. The new v4 all-retained kernel reproduced archived output plus step
count on 0/2, and v4 unweighted Gaussian-32 on 1/2. Instrumented versus
untraced v4 parity passed for every smoke method, and v4 unit sensitivity
reproduced unweighted v4 exactly on 2/2 prompts.

This is why the full study reruns native dense, v4 all-retained, and v4
unweighted controls under the new binary. Archived v3 curves are historical
context, not exact matched-backend controls for the v4 adaptive methods.
The original native SDPA backend also retains its previously documented local
mask and numerical differences from the matched H100 kernel.

Margin sensitivity uses the previous processed top-two gap multiplied by the
known native scalar temperature to recover the pre-temperature gap without an
extra vocabulary pass. In 13 archived all-retained trajectories (17,408 token
positions), comparison with the separately recorded raw-logit gap had mean
absolute difference `9.84e−7` and maximum `4.13e−6`. This is a very small
numerical reconstruction error, not exact raw-logit identity.
