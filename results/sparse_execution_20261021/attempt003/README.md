# Unique Q128 producer follow-up, 2026-10-21

Synthetic optimization only; no native benchmark relaunch. Starting source
commit5a5611e2 plus this attempt's explicitly frozen source/build identities.
The previous benchmark attempt004 remains partial and paused.

Q128 routing uses both query warpgroups in one producer CTA, with a local
Q128 risk reduction. A retained-only GPU queue feeds two independent Q64/full-D
PV CTAs in the same cooperative kernel launch. This removes output-channel QK
duplication and cross-CTA risk voting; rejected tiles never enqueue full PV.
The hypothesis is that two active routing warpgroups per producer improve
scheduling compared with attempt002's Q64 producer, despite fewer producers.
No extra model forward, dense PV, two-pass discovery or sampler change.

Qualification precedes timing: independent exact ordered masks, finite output,
same-mask FA4 relativeL2<0.01, changed Q/K/V/Z and held-mask graph replays.
Primary timing: synthetic Q256/H16/Hkv2/D512 BF16,8Kkeys,50% actual tile sparsity,
fresh projection/RMS, sketch packing, compact masks, actual capture counter,
matched dense and all-kept controls. Warm compilation and graph capture outside
timed regions. Save failures and compiler resource reports. No full evaluation
or per-workload calibration unless preceding gates pass.

Q128 queue qualification001 passes80changed-input graph replays; refresh001
measures598.60us complete8K refresh vs361.63us plain dense, a negative. Further
interleaved design computes even/odd KV tiles with two unique QK producers per
Q64 unit, then an ordered Gaussian32 router and four128-channel PV consumers.
Seven128-thread CTAs per unit share one cooperative launch. A32KB per-CTA shared
workspace and128register launch cap allow four CTAs per SM, so all448CTAs fit
concurrently on H100. QK, probability and risk queues stay resident with two
slots; full attention logits are never materialized. PV only receives retained
tiles. This investigates parallelism in unique QK work rather than duplicating
that work across output channels. Actual occupancy and correctness must qualify.

interleaved_qualification001 passes80changed-input graph replays with exact
masks and relativeL2<0.003. interleaved_refresh001 complete8K refresh764.50us
vs360.08us plain dense is a negative; timed captures0. Initial2be28678build
failed a C++ auto declaration combining different tensor layouts; fixed in
fdd80a7f. Both preserved. No promotion or native benchmark relaunch.
