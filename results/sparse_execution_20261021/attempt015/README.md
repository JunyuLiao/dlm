# Register-resident Q producer, 2026-10-21

Synthetic-only unique QK/routing producer uses WGMMA register-source Q and
two separate256-channel PV CTAs, each128threads. Removing64KB shared Q makes
the cooperative192-CTA grid fit two blocks per SM. Current Q is loaded once;
the producer retains its register fragments across the ordered KV64 traversal.
Rejected blocks do not enter the two-slot PV queue. No dense PV or extra forward.

qualification001 passes80changed-input graph replays with exact masks and
maximum relativeL2<0.003. Complete8K refresh636.42us vs359.18us plain dense is
a negative. The unroll32 producer has24bytes compiler spills. Unroll1 removes
spills but places its dynamically indexed512-byte query array in local memory;
it is preserved as a resource-negative build, not promoted or native-qualified.
