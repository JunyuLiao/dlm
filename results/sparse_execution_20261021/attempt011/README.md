# Compact QK assembly sequence, 2026-10-21

Synthetic-only optimization of best interior/tail 3D-TMA kernel. One inline PTX
sequence issues32identical QK WGMMA operations while directly advancing the
shared descriptors. It preserves original BF16 operands and accumulation order;
no approximate selector, dense PV on rejected tiles or extra forward. This
reduces descriptor liveness and compiles around200registers with no spills.
Initial build17869972 failed the unsupported sm90 generic PTX target; guarded
SM90a build1045b6d6 succeeds and the failed build remains preserved.

qualification001 passes80changed-input graph replays with exact masks and
maximum relativeL2<0.003. Primary8K/50% complete refresh timing is frozen in
refresh001, charging currentV projection/RMS, sketch packing and compact masks.
All graph capture/compilation precedes timing; actual capture counters required.
The native benchmark remains paused until the user's ordered gates pass.

refresh001 complete8K refresh435.32us vs362.45us dense: no improvement over
the best interior specialization429.32us. Actual timed capture count0.
