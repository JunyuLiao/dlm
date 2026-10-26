# Exact replay commit-boundary diagnostic

|ID / seed|Support|Remaining A→S|Terminal argmax equal|Commit cache numerical equal (30layers)|Next init canvas equal|Init CUDA RNG equal|Full output equal|
|---|---|---|---|---|---|---|---|
|longbench_v2/66ec2755821e116aacb1b6f2 / 42|cell000_current_error|15→12|True|True|False|False|False|
|longbench_v2/66ec2755821e116aacb1b6f2 / 42|cell000_direction_high|15→13|True|True|False|False|False|
|longbench_v2/66ec2755821e116aacb1b6f2 / 42|cell000_direction_low|15→13|True|True|False|False|False|
|longbench_v2/66ec2755821e116aacb1b6f2 / 43|cell001_current_error|44→44|True|True|True|True|True|
|longbench_v2/66ec2755821e116aacb1b6f2 / 43|cell001_direction_high|44→44|True|True|True|True|True|
|longbench_v2/66ec2755821e116aacb1b6f2 / 43|cell001_direction_low|44→44|True|True|True|True|True|
|longbench_v2/66ec2755821e116aacb1b6f2 / 43|cell001_support_sham|44→44|True|True|True|True|True|
|longbench_v2/6724645dbb02136c067d73ad / 43|cell003_current_error|20→17|True|True|False|False|False|
|longbench_v2/6724645dbb02136c067d73ad / 43|cell003_direction_high|20→17|True|True|False|False|False|
|longbench_v2/6724645dbb02136c067d73ad / 43|cell003_direction_low|20→20|True|True|True|True|True|

All requests finish natively and reproduce historical output,N,C,T,RNG/forward fingerprints. Same-target tokens are not task correctness. Complete conditional phase differences and byte/numerical/nonfinite comparisons are in summary.json. No restoration, RNG compensation, extra model forward or new selector.
