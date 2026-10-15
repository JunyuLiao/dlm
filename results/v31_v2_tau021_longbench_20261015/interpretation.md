# Interpretation of the complete V2 tau=0.021 LongBench run

V2 scores 254/503 (50.50%), compared with inherited Yuhan V31 268/503
(53.28%). The observed difference is -14 answers / -2.783 percentage points;
the paired 20,000-draw problem bootstrap interval is [-6.362, +0.795] pp,
with exact McNemar p=0.1654. This is no demonstrated accuracy gain or accuracy
preservation versus V31. Compared with FULL/PIECEWISE dense, V2 scores seven
more answers, but its interval also spans zero: [-2.584, +5.368] pp.

V2 uses 124,538 native denoising calls versus V31 140,936 (-11.635%),
6,504 visible canvases versus 6,833 (-4.815%), and 19.148 calls/canvas versus
20.626 (-7.165%). Output tokens decrease from 1,684,180 to 1,598,393
(-5.094%). Fewer calls are an aggregate result: 253 problems use fewer calls,
247 more, and three the same; the median paired change is only -1 call.
Thus the mean reduction is not a uniform per-problem improvement.

Absolute retained GLOBAL work is 5,243,209,214 versus 5,833,185,760 tiles
(-10.114%). GLOBAL eligible work is 30,032,440,160 versus 34,713,640,000
(-13.485%). V2's physical GLOBAL sparsity is actually lower (82.54% versus
83.20%); its smaller native trajectory reduces the absolute denominator.
This comparison is not equal retained work, despite the nearby sparsity ratios.

The first-canvas kept-work ratio is 0.96087 across all 503 items. Through
canvas 4 it is 1.00438 on 443 paired eligible items; through canvas 8 it is
0.99703 on 263; through canvas 20 it is 1.00936 on only 42. These are different
native survivor cohorts, not a curve over a fixed 503-item cohort. The full
per-item work-difference distributions are retained. Matching an early mean
on a survivor cohort neither matches every item nor establishes equal total work.

Held-phase retained work falls from 2,564,783,360 to 2,101,668,777 tiles
(-18.06%). Its eligible work falls from 30,004,449,760 to 25,554,089,120;
the kept fraction changes from 8.548% to 8.224%. Initial and refresh work also
fall with the shorter trajectory. Carried work slightly rises from 149,766,720
to 154,161,877 tiles. Source phase receipts preserve these distinctions.

Accuracy overlap is 217 both correct, 37 V2-only correct, 51 V31-only correct,
and 198 both wrong. All 503 output hashes differ; 499 output lengths and 500
denoising-call counts differ. The aggregate accuracy loss consists of four
more unparsed responses (34 versus 30) and ten more parsed-but-wrong responses
(215 versus 205). Capped outputs decrease from 18 to 13. The loss therefore
cannot be attributed simply to more output caps or parser failures. No causal
claim about a particular selector mechanism follows from these aggregate counts.

V2 discovery and candidate evaluation each count 2,881,671,520 rectangles.
Recorded value-discovery and selection times are 601.56 s and 515.37 s;
these are additional instrumented selector costs, separate from FA4 retained
tiles. Instrumented request wall time is 6,008.42 s versus V31 5,242.81 s.
These are audit measurements, not clean latency or throughput. Zero V31
value-specific counters do not mean the V31 mass selector has zero overhead.

All 503 scorer bindings match. Five control/V2 paths were qualified on two
cells each; all eight repeated control outputs and their N/T/C matched the
completed reference records exactly. All 886 execution files, 24 inputs, and
76 original reference artifacts were independently verified. GLOBAL-only,
LOCAL-dense, exact selector/state semantics, native FA4 output and sampler
remain fixed. No invalid rows, missing scorer cells, or request-time graph
captures occurred. Other thresholds and the separate refresh campaign remain
paused. This previously examined benchmark is not held-out confirmation.
