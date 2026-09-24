# Native stop diagnostic summary

Separate diagnostic only; these runs do not replace formal quality or timing receipts.

Observer gate: **PASS**.

| Arm | Status | Calls | Stable/Confident FF, FT, TF, TT | Accepted mean | Renoised mean | Termination |
| --- | --- | ---: | --- | ---: | ---: | --- |
| native_dense | diagnostic | 146 | 115, 10, 1, 20 (n=146) | 129.71 | 126.29 | eos |
| fresh_junyu_T | diagnostic | 208 | 164, 13, 6, 25 (n=208) | 122.84 | 133.16 | eos |
| M1 | diagnostic | 908 | 215, 0, 670, 23 (n=908) | 13.93 | 242.07 | eos |
| M3 | diagnostic | 1289 | 409, 0, 855, 25 (n=1289) | 22.33 | 233.67 | length |

native_dense: 20 canvases; calls per canvas [12, 6, 7, 15, 6, 8, 5, 5, 9, 7, 11, 11, 4, 4, 9, 8, 4, 4, 5, 6]; iteration-cap finals 0; joint-stop finals 20.
  Primary parity: tokens=True, calls=True, prompt=True.
  Primary receipt NOT_RUN under /home/exouser/dyh/numerical_qk_reuse_native_20260924/runs/integrated_smoke_01.
  Primary receipt NOT_RUN under /home/exouser/dyh/numerical_qk_reuse_native_20260924/runs/m3_smoke_01.
fresh_junyu_T: 25 canvases; calls per canvas [10, 9, 9, 7, 6, 12, 7, 8, 9, 9, 7, 12, 8, 5, 5, 9, 11, 8, 7, 5, 8, 10, 13, 6, 8]; iteration-cap finals 0; joint-stop finals 25.
  Primary receipt NOT_RUN under /home/exouser/dyh/numerical_qk_reuse_native_20260924/runs/dense_smoke_02.
  Primary parity: tokens=True, calls=True, prompt=True.
  Primary receipt NOT_RUN under /home/exouser/dyh/numerical_qk_reuse_native_20260924/runs/m3_smoke_01.
M1: 27 canvases; calls per canvas [48, 48, 48, 46, 48, 42, 43, 37, 42, 35, 33, 26, 26, 28, 31, 27, 36, 30, 28, 27, 38, 20, 22, 25, 28, 38, 8]; iteration-cap finals 4; joint-stop finals 23.
  Primary receipt NOT_RUN under /home/exouser/dyh/numerical_qk_reuse_native_20260924/runs/dense_smoke_02.
  Primary parity: tokens=True, calls=True, prompt=True.
  Primary receipt NOT_RUN under /home/exouser/dyh/numerical_qk_reuse_native_20260924/runs/m3_smoke_01.
M3: 32 canvases; calls per canvas [48, 48, 48, 48, 48, 45, 46, 47, 45, 37, 42, 39, 48, 48, 38, 35, 30, 35, 43, 38, 40, 31, 34, 35, 42, 34, 41, 36, 28, 35, 34, 43]; iteration-cap finals 7; joint-stop finals 25.
  Primary receipt NOT_RUN under /home/exouser/dyh/numerical_qk_reuse_native_20260924/runs/dense_smoke_02.
  Primary receipt NOT_RUN under /home/exouser/dyh/numerical_qk_reuse_native_20260924/runs/integrated_smoke_01.
  Primary parity: tokens=True, calls=True, prompt=True.

JSON contains input-file, source, and output hashes; no prompt, answer, or token content.
