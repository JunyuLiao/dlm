# Primary frontier checkpoint — RULER and AIME complete

Generation and offline scoring are complete for RULER2808/2808 and AIME1080/1080.
This checkpoint interprets the verified primary summaries; secondary results
are interpreted separately in morning_brief.md. No policy, question, seed or gate
is changed in response to these results. The single ball candidate remains STOP
before CUDA implementation.

RULER:130 questions,13 official task strata,3 paired seeds;390 first outputs/arm.
Official task macro includes partial credit. Timing uses the preselected26 questions,
3 seeds,78 accepted warm pairs/arm (39/host). All390 first canvases/arm are retained.

| Arm | Official task macro % | Decoder calls/canvas | Whole PV support skipped % |
|---|---:|---:|---:|
| D_native |89.4872|4.0718|N/A|
| D_matched |89.6539|4.2513|0|
| U50 |90.0598|4.4872|50.255|
| U60 |90.0086|4.7821|59.586|
| T50 |89.7009|4.3436|49.054|
| T60 |89.8376|4.7077|60.219|

T60 versus U50 skips about9.96 percentage points more whole PV support, with
about4.9% more first-output decoder calls per canvas. Quality difference is
-0.222pp,95% question-cluster interval[-0.855,+0.256]pp. This does not establish
quality noninferiority: no prospective margin was specified. Accepted-warm
whole-request wall ratio is1.0049,[0.9720,1.0401]; no clear speed advantage overU50.

Against U60, T60 quality difference is-0.171pp,[-0.855,+0.427]pp. Warm wall ratio
is0.9674,[0.9384,0.9971], a small timing advantage on this26-question subset.
The shuffled/uniform controls are still needed before attributing value causally
to the query-specific allocation. Do not infer their outcome from this comparison.

Against native dense, T60 quality difference is+0.350pp,[-0.188,+1.034]pp; warm
wall ratio1.1114,[1.0703,1.1549], so T60 is slower on the accepted timing subset.
T60 has about15.6% more first-output calls/canvas. Those390-cell work totals are
not the78-cell timing subset and must not be combined into a causal decomposition.
The JSON provides exact matched timing-cell decompositions separately by host.

Every timing ratio pairs methods on the same GPU first; the reported geometric
mean combines those ratios with question clusters/task strata. Absolute seconds
are kept separate by host. Intervals are descriptive10,000-resample bootstrap.
PV support counters do not demonstrate QK savings; allU/T paths compute freshQK.
Qualified generation latency, TBT and direct full-forward cost remain unavailable.
No inference of zero overhead or full-forward speedup is warranted.

Source: ruler_redacted_summary.json SHA256
6ee11dfca05d2d958a9c0a40917abc64546c33ca2bf9d62fcf576dca97d35049.
Frozen primary scorer sourcef45fa83; identities and missing metric fields are
preserved in the summary. Final interpretation is in morning_brief.md.

## AIME complete primary score (90 first outputs/arm, 90 strict warm/arm)

| Arm | Correct/90 | Calls/canvas | Length-capped | Whole PV support skipped |
|---|---:|---:|---:|---:|
| D_native |46|13.349|35|N/A|
| D_matched |49|13.237|33|0%|
| U50 |45|15.320|39|39.216%|
| U60 |48|18.388|40|49.680%|
| T50 |57|14.218|31|28.913%|
| T60 |49|15.396|37|40.302%|

T60/U50 first-answer quality difference +4.44 percentage points, bootstrap95%
interval[-3.33,+13.33]pp; accepted-warm request-wall geomean0.9887
[0.9283,1.0529]. T60/U60 quality+1.11pp[-6.67,+8.89]pp; wall
ratio0.8238[0.7680,0.8848]. T60/native quality+3.33pp[-6.67,+13.33]pp;
wall ratio1.3102[1.2380,1.3912]. These are paired withinGPU and clustered
by question. The intervals are descriptive, not a proved noninferiority claim.

The actual AIME PV support skip rate differs sharply from RULER calibration:
T60 40.30% versus U50 39.22%, only1.09pp apart. The selected70% RULER
controls still need to run; their outcomes cannot repair this AIME work point.
T50 has57/90 correct, an exploratory arm total; it was not the prespecified
T60-versus-U50/U60 gate and needs paired uncertainty before interpretation.
Many AIME outputs hit the8192-token cap or remained unparsed; all count under
the frozen scorer. The candidate remains stopped beforeCUDA implementation.

Source: aime_redacted_summary.json SHA256
c3cb6cdc70d9699635439e74672b8a832140caf2c99a48957d78120b2ecc1c88.
