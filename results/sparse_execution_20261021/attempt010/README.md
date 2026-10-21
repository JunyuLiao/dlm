# Earlier K/V staging with explicit reuse handshakes, 2026-10-21

Synthetic-only follow-up to the best interior/tail kernel. A WG1 completion
epoch protects shared V from overwrite before previous PV finishes. This allows
the next V copy at the loop entrance. A separate WG1 QK-completion epoch lets
WG0 prefetch the next K immediately after both WGs finish current QK, before
softmax/sketch work. Release/acquire shared-memory signals preserve buffer
lifetime without an added full-CTA barrier. Fixed thresholds, Q128 admission,
current Q/K/V, Gaussian32 precision and conditional full PV are unchanged.

Two separately frozen builds compare earlier V alone and earlier K+V. Both
compile without spills. Changing-input numerical and CUDA-graph qualification
must precede8K/50% timing with projection, sketch packing, masks, dense controls
and actual capture accounting. No native evaluation, calibration or speed claim.

qualification001 passes80changed-input graph replays, exact masks and relative
L2<0.003. Both-early build refresh001 complete8K refresh479.68us vs362.54us
plain dense is negative. Earlier-V-only build remains compile-only, no timing
claim. Actual timed capture count0 on measured both-early build.
