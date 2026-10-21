# Within-CTA QK/PZ lookahead, two output groups, 2026-10-21

Synthetic optimization only. Follow attempt008's unique lookahead QK/PZ schedule
with two output-channel groups, each owning256channels across two128-channel
PV warpgroups. This repeats QK/routing twice rather than the selected reference's
four times. Q64 per CTA and128CTAs at Q256/H16 avoid the underoccupied64-CTA
full-output layout. The kernel uses242registers and no spills. Shared next-QK
FP32 scores reuse packed-Z/low-P storage after PZ completes, with exact current
Q/K/V and fixed online thresholds. Rejected tiles never execute full PV.

Qualification: exact ordered masks, finite output, relativeL2<0.01, changed
Q/K/V/Z/held graph replay. Timing follows qualification at8Kkeys/50% physical
sparsity, including currentV projection/RMS, sketch pack and compact masks.
No native model or benchmark is launched by this attempt. Every source/build
and failed variant is frozen separately and the prior campaign remains partial.

qualification001 passes80changed-input graph replays with exact masks and
relativeL2<0.003. refresh001 complete8K refresh614.42us vs362.14us dense is a
negative. Timed captures0, no other GPU client, no hidden dense PV.
