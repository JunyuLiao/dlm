# Full historical-state Q64 alias sweep (2026-10-02)

All 144 historical real-prefix-need snapshots were tested: 48 each in the nearest
nominal 32K / 64K / 96K bin. Q/K/V were synthetic BF16 with native token-major Q,
interleaved paged K/V strides, and randomized physical page order. All canvas
tiles were kept. This is a component diagnostic; the snapshots are not current
vLLM request trajectories and do not establish accuracy or request speed.

Source: `8c9cd31ae06b8b54ef6dcf91f792c864139afa5c`, following qualified parent
deployment `dcfdb8730efff131f92d0a8bbbab514ef1eedf18`. The unchanged own vLLM
environment and previously qualified FA4 paged-sparse patch were reused. The
script SHA is recorded in [summary.json](summary.json). The worker completed
with exit code zero. Reserved GPU time was **27.0913 s**, including process
startup and CPU preparation; the component script's span was **21.6885 s**.

Each S1/2/4 configuration used its own adapter and split cache, the same Q64
support, Q/K/V, and exact generic softmax/LSE merge. S1 was not optimized to skip
the merge. Eight warm calls preceded 30 rotated timing repeats per configuration
and state. Timings include page-table/used allocation and merge, but exclude
selector, first-use keep-map/list/split construction, KV copies, request lifecycle
and model work. Every adapter built its held list once, with no timed rebuild.

## Descriptive results

Ratios are geometric means of per-state median CUDA-event time relative to S2;
smaller is faster. Wins count strictly smaller per-state medians, without a
noise threshold or statistical significance claim.

| Nominal bin | States | S1 / S2 | S4 / S2 | S4 faster / slower than S2 |
|---|---:|---:|---:|---:|
| 32K | 48 | 1.34601 | 1.01731 | 19 / 29 |
| 64K | 48 | 1.38828 | 0.98315 | 28 / 20 |
| 96K | 48 | 1.38074 | 0.96533 | 33 / 15 |
| All | 144 | 1.37155 | 0.98836 | 80 / 64 |

S1 was slower in all 144 states. S4's overall component gain was about 1.16%,
with a 1.73% regression at 32K and a 3.47% gain at 96K. This mixed result does
not select a new default or establish an end-to-end benefit. A prospective
length-conditioned S4 request variant would require a separately frozen
qualification and matched controls. Default S2 remains unchanged.

Each output passed the same masked IEEE FP32 reference, with TF32 explicitly
disabled and nonfinite outputs/references rejected. Maximum relative error
over the 144 states was 0.4204% / 0.6051% / 0.5397% for S1 / S2 / S4. These
are numerical consumer checks, not model accuracy measurements.

States from the same historical requests are related. The geometric means and
win counts are descriptive; there are no independent-request confidence
intervals or quality claims. No prompts, completions, item identities, need
bits, token hashes, raw snapshot filenames or private paths are published.

The separate [qualification note](qualification_note.json) records only that
one timed private completion matched between legacy and canvas-release modes
for each of Q128 and Q64 in qualification004. N was 72 / 72 for Q128 and
66 / 66 for Q64. This small equality check does not establish general method
equivalence or accuracy.
