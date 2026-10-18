# Preserved attempt001 — profiling startup failure

Frozen source: `6aab86325990edf6e5ab6bb72bcd909781234e97`.
Protocol SHA256:
`1e7714be3fd4690b41e37727a7b8ae283800c24f579f90eec03d90e37bfb964e`.

Engines 001–005 completed: the six-request native qualification and all 20 clean
headroom measurements. Every completion marker and its bound artifacts remain
unchanged. Qualification passed; clean measurements have zero measured graph
captures and no measured JIT warnings. These are diagnostic results on selected
inputs, not population accuracy or speed claims.

Engine 006 failed before writing any request record. An audit hook accessed
`HeadroomCollector.phase` during model initialization, before the first request
initialized that field. Its receipt records zero measured requests, worker
failure and 70.189 s GPU reservation. No profile or factorial result is available
from this attempt. Engines 007–012 were not launched.

The attempt is failed and has no final completion marker. Total charged GPU
reservation is 1,521.995 s: qualification 471.963 s and headroom 1,050.032 s, including
the failed worker. Recovery uses a fresh attempt and explicit references to
completed artifacts under `docs/REFRESH_DIAGNOSTIC_RECOVERY_20261018.md`.
The original protocol, source snapshot, failed leaf and state are preserved.

The completed clean measurements show consumer and trajectory sensitivity.
At 41K, dynamic V31, FULL dense and PIECEWISE dense are correct; fixed16 and
all-kept FA4 are wrong. Their mean request times are 8.628,6.901,7.108,4.433
and 7.480 s respectively. At 112K, every arm is wrong; corresponding times are
10.875,13.561,16.419,11.600 and 19.006 s. Both repetitions reproduce their arm's
output hash and call count. These two inputs do not justify promoting fixed16
or treating consumer paths as numerically interchangeable.
