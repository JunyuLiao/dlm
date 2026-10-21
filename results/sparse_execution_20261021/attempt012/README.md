# Static descriptor negative, 2026-10-21

Synthetic build43acdd45 specializes literal shared-memory addresses and guards
the assumed zero dynamic-shared base. qualification001 fails immediately with
CUDA launch failure; no timing or promotion. The explicit address assumption is
not qualified and must not be reused. Static SASS also serializes WGMMA with
inserted dependency waits, so this design is abandoned. Preserve failed build,
source and operator attempt; independent subsequent processes reinitialize CUDA.
