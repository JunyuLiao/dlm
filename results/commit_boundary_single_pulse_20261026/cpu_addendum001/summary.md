# CPU addendum: verified historical single pulses

All ten terminal argmax canvases agree. Five earlier-stop requests change complete outputs; five unchanged-stop requests reproduce complete outputs. This is consistent with, but does not isolate, native RNG-position mediation. Old hashes do not resolve signed-zero SC differences.

|ID / seed|Support|A remaining|S remaining|Output equal|CUDA RNG equal|Adaptive / capped|Correct / parsed|
|---|---|---:|---:|---|---|---|---|
|longbench_v2/66ec2755821e116aacb1b6f2 / 42|cell000_current_error|15|12|False|False|True / False|False / False|
|longbench_v2/66ec2755821e116aacb1b6f2 / 42|cell000_direction_high|15|13|False|False|True / False|False / False|
|longbench_v2/66ec2755821e116aacb1b6f2 / 42|cell000_direction_low|15|13|False|False|True / False|False / False|
|longbench_v2/66ec2755821e116aacb1b6f2 / 43|cell001_current_error|44|44|True|True|False / True|False / False|
|longbench_v2/66ec2755821e116aacb1b6f2 / 43|cell001_direction_high|44|44|True|True|False / True|False / False|
|longbench_v2/66ec2755821e116aacb1b6f2 / 43|cell001_direction_low|44|44|True|True|False / True|False / False|
|longbench_v2/66ec2755821e116aacb1b6f2 / 43|cell001_support_sham|44|44|True|True|False / True|False / False|
|longbench_v2/6724645dbb02136c067d73ad / 43|cell003_current_error|20|17|False|False|True / False|False / False|
|longbench_v2/6724645dbb02136c067d73ad / 43|cell003_direction_high|20|17|False|False|True / False|False / False|
|longbench_v2/6724645dbb02136c067d73ad / 43|cell003_direction_low|20|20|True|True|True / False|False / False|

Deleting GQA maximum 0.003998036; all-control maximum 0.004263655. Canonical slack =0.005−entropy. The canonical420-row work correction is bound by input SHA256; raw stale fields stay historical. Historical failed gate stays failed; its quality/work omissions are not reused.
