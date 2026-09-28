# v24 bootstrap6 output-numerics audit

Generated: 2026-09-28T06:21:00Z

> First-generation ('first') outputs only were used for quality scoring; the warm repeat is a timing measurement only, never rescored. In this panel host is confounded with seed by design: seed 101 always ran on 149.165.151.254, seed 202 always ran on 149.165.159.64. All contrasts below are exploratory question-cluster comparisons on a frozen, correctness-gated but small development panel -- not a noninferiority test.

## Union audit result: **PASS**

- protocol_id: `v21_bootstrap6_b8b2c59449fd0ebf`
- protocol sha256: `e447785e6fbd181050ee4fbc1aea475ab44b188dba5f30aff70a3b9a21a31955`
- cells total: 120, missing: 0, invalid: 0
- no violations found: exactly one segment executed each cell, all host/gpu/cell_id/ledger cross-checks agree.

## Lineage

- **preview_aime**: protocol sha256 `e447785e6fbd181050ee4fbc1aea475ab44b188dba5f30aff70a3b9a21a31955`, binding panel_protocol_sha256 matches: True, source_commits: `{'149.165.151.254': '6af4d42374c0a2d6d71ce91b4f020255b120b800', '149.165.159.64': '6af4d42374c0a2d6d71ce91b4f020255b120b800'}`
- **lb_rest**: protocol sha256 `e447785e6fbd181050ee4fbc1aea475ab44b188dba5f30aff70a3b9a21a31955`, binding panel_protocol_sha256 matches: True, source_commits: `{'149.165.151.254': 'dc525227a89651a44ba61f93ab7175a576fb32ef', '149.165.159.64': 'dc525227a89651a44ba61f93ab7175a576fb32ef'}`

## Per-dataset corrected table

### aime26

| arm | correct/valid | failed | missing | unscored | invalid | calls | canvases | calls/canvas | B0 layer-calls | BO layer-calls | A layer-calls | D layer-calls | H layer-calls | model-calls (sum/5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D_native | 4/8 | 0 | 0 | 0 | 0 | 1961 | 168 | 11.67 | n/a | n/a | n/a | n/a | n/a | n/a |
| T_scope | 6/8 | 0 | 0 | 0 | 0 | 2020 | 162 | 12.47 | n/a | n/a | n/a | n/a | n/a | n/a |
| M3_R3_A8_incumbent | 6/8 | 0 | 0 | 0 | 0 | 1855 | 173 | 10.72 | 0 | 0 | 1530 | 2090 | 5655 | 1855.0 |
| M1_native_bootstrap2_observe1 | 6/8 | 0 | 0 | 0 | 0 | 1847 | 156 | 11.84 | 780 | 780 | 600 | 7075 | 0 | 1847.0 |
| M3_native_bootstrap2_observe1 | 6/8 | 0 | 0 | 0 | 0 | 1967 | 164 | 11.99 | 820 | 820 | 655 | 2040 | 5500 | 1967.0 |
| B_native_bootstrap2_observe1 | 5/8 | 0 | 0 | 0 | 0 | 1931 | 169 | 11.43 | 845 | 845 | 605 | 0 | 7360 | 1931.0 |

### longbench_v2

| arm | correct/valid | failed | missing | unscored | invalid | calls | canvases | calls/canvas | B0 layer-calls | BO layer-calls | A layer-calls | D layer-calls | H layer-calls | model-calls (sum/5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D_native | 6/12 | 0 | 0 | 0 | 0 | 2360 | 140 | 16.86 | n/a | n/a | n/a | n/a | n/a | n/a |
| T_scope | 6/12 | 0 | 0 | 0 | 0 | 2223 | 128 | 17.37 | n/a | n/a | n/a | n/a | n/a | n/a |
| M3_R3_A8_incumbent | 5/12 | 0 | 0 | 0 | 0 | 2604 | 145 | 17.96 | 0 | 0 | 1940 | 3070 | 8010 | 2604.0 |
| M1_native_bootstrap2_observe1 | 5/12 | 0 | 0 | 0 | 0 | 2623 | 147 | 17.84 | 735 | 735 | 1140 | 10505 | 0 | 2623.0 |
| M3_native_bootstrap2_observe1 | 6/12 | 0 | 0 | 0 | 0 | 2415 | 143 | 16.89 | 715 | 715 | 995 | 2685 | 6965 | 2415.0 |
| B_native_bootstrap2_observe1 | 4/12 | 0 | 0 | 0 | 0 | 2524 | 144 | 17.53 | 720 | 720 | 1060 | 0 | 10120 | 2524.0 |

## Estimand table (paired, same-host/gpu, accepted-warm cells)

| dataset | arm_x | arm_y | n_pairs | geoW | geoN | geoW/N | identity_ok | sum_calls_ratio | discordant x-correct/y-wrong | discordant x-wrong/y-correct |
|---|---|---|---|---|---|---|---|---|---|---|
| aime26 | M3_native_bootstrap2_observe1 | D_native | 8 | 1.0415 | 1.0217 | 1.0194 | True | 1.0031 | 2 | 0 |
| aime26 | M3_native_bootstrap2_observe1 | M3_R3_A8_incumbent | 8 | 1.0169 | 1.0172 | 0.9997 | True | 1.0604 | 0 | 0 |
| aime26 | M3_native_bootstrap2_observe1 | B_native_bootstrap2_observe1 | 8 | 0.9839 | 0.9772 | 1.0068 | True | 1.0186 | 1 | 0 |
| aime26 | M3_native_bootstrap2_observe1 | T_scope | 8 | 1.0307 | 1.0207 | 1.0097 | True | 0.9738 | 0 | 0 |
| aime26 | B_native_bootstrap2_observe1 | D_native | 8 | 1.0586 | 1.0455 | 1.0125 | True | 0.9847 | 2 | 1 |
| aime26 | M1_native_bootstrap2_observe1 | D_native | 8 | 0.9750 | 0.9457 | 1.0309 | True | 0.9419 | 2 | 0 |
| aime26 | T_scope | D_native | 8 | 1.0105 | 1.0009 | 1.0096 | True | 1.0301 | 2 | 0 |
| longbench_v2 | M3_native_bootstrap2_observe1 | D_native | 12 | 0.9901 | 1.0399 | 0.9521 | True | 1.0233 | 0 | 0 |
| longbench_v2 | M3_native_bootstrap2_observe1 | M3_R3_A8_incumbent | 12 | 0.9997 | 0.9936 | 1.0061 | True | 0.9274 | 1 | 0 |
| longbench_v2 | M3_native_bootstrap2_observe1 | B_native_bootstrap2_observe1 | 12 | 0.9968 | 0.9755 | 1.0218 | True | 0.9568 | 2 | 0 |
| longbench_v2 | M3_native_bootstrap2_observe1 | T_scope | 12 | 1.0952 | 1.1451 | 0.9564 | True | 1.0864 | 0 | 0 |
| longbench_v2 | B_native_bootstrap2_observe1 | D_native | 12 | 0.9934 | 1.0661 | 0.9318 | True | 1.0695 | 0 | 2 |
| longbench_v2 | M1_native_bootstrap2_observe1 | D_native | 12 | 1.0577 | 1.0687 | 0.9897 | True | 1.1114 | 0 | 1 |
| longbench_v2 | T_scope | D_native | 12 | 0.9041 | 0.9081 | 0.9955 | True | 0.9419 | 0 | 0 |

### Host-stratified W/N decomposition (host confounded with seed)

- **aime26: M3_native_bootstrap2_observe1 vs D_native**
  - host `149.165.151.254` (n=4): sum W ratio=0.9406, sum N ratio=0.9171 (sumW_x=146.56 sumW_y=155.81 sumN_x=951 sumN_y=1037)
  - host `149.165.159.64` (n=4): sum W ratio=1.1248, sum N ratio=1.0996 (sumW_x=146.09 sumW_y=129.88 sumN_x=1016 sumN_y=924)
- **aime26: M3_native_bootstrap2_observe1 vs M3_R3_A8_incumbent**
  - host `149.165.151.254` (n=4): sum W ratio=1.0247, sum N ratio=1.0182 (sumW_x=146.56 sumW_y=143.03 sumN_x=951 sumN_y=934)
  - host `149.165.159.64` (n=4): sum W ratio=1.1073, sum N ratio=1.1031 (sumW_x=146.09 sumW_y=131.93 sumN_x=1016 sumN_y=921)
- **aime26: M3_native_bootstrap2_observe1 vs B_native_bootstrap2_observe1**
  - host `149.165.151.254` (n=4): sum W ratio=0.9518, sum N ratio=0.9463 (sumW_x=146.56 sumW_y=153.98 sumN_x=951 sumN_y=1005)
  - host `149.165.159.64` (n=4): sum W ratio=1.1088, sum N ratio=1.0972 (sumW_x=146.09 sumW_y=131.76 sumN_x=1016 sumN_y=926)
- **aime26: M3_native_bootstrap2_observe1 vs T_scope**
  - host `149.165.151.254` (n=4): sum W ratio=0.8858, sum N ratio=0.8709 (sumW_x=146.56 sumW_y=165.45 sumN_x=951 sumN_y=1092)
  - host `149.165.159.64` (n=4): sum W ratio=1.1015, sum N ratio=1.0948 (sumW_x=146.09 sumW_y=132.63 sumN_x=1016 sumN_y=928)
- **aime26: B_native_bootstrap2_observe1 vs D_native**
  - host `149.165.151.254` (n=4): sum W ratio=0.9883, sum N ratio=0.9691 (sumW_x=153.98 sumW_y=155.81 sumN_x=1005 sumN_y=1037)
  - host `149.165.159.64` (n=4): sum W ratio=1.0145, sum N ratio=1.0022 (sumW_x=131.76 sumW_y=129.88 sumN_x=926 sumN_y=924)
- **aime26: M1_native_bootstrap2_observe1 vs D_native**
  - host `149.165.151.254` (n=4): sum W ratio=0.9008, sum N ratio=0.8717 (sumW_x=140.35 sumW_y=155.81 sumN_x=904 sumN_y=1037)
  - host `149.165.159.64` (n=4): sum W ratio=1.0656, sum N ratio=1.0206 (sumW_x=138.41 sumW_y=129.88 sumN_x=943 sumN_y=924)
- **aime26: T_scope vs D_native**
  - host `149.165.151.254` (n=4): sum W ratio=1.0619, sum N ratio=1.0530 (sumW_x=165.45 sumW_y=155.81 sumN_x=1092 sumN_y=1037)
  - host `149.165.159.64` (n=4): sum W ratio=1.0211, sum N ratio=1.0043 (sumW_x=132.63 sumW_y=129.88 sumN_x=928 sumN_y=924)
- **longbench_v2: M3_native_bootstrap2_observe1 vs D_native**
  - host `149.165.151.254` (n=6): sum W ratio=0.8628, sum N ratio=0.8951 (sumW_x=208.97 sumW_y=242.20 sumN_x=1144 sumN_y=1278)
  - host `149.165.159.64` (n=6): sum W ratio=1.1090, sum N ratio=1.1747 (sumW_x=222.52 sumW_y=200.66 sumN_x=1271 sumN_y=1082)
- **longbench_v2: M3_native_bootstrap2_observe1 vs M3_R3_A8_incumbent**
  - host `149.165.151.254` (n=6): sum W ratio=0.8361, sum N ratio=0.8207 (sumW_x=208.97 sumW_y=249.93 sumN_x=1144 sumN_y=1394)
  - host `149.165.159.64` (n=6): sum W ratio=1.0564, sum N ratio=1.0504 (sumW_x=222.52 sumW_y=210.63 sumN_x=1271 sumN_y=1210)
- **longbench_v2: M3_native_bootstrap2_observe1 vs B_native_bootstrap2_observe1**
  - host `149.165.151.254` (n=6): sum W ratio=0.9521, sum N ratio=0.9226 (sumW_x=208.97 sumW_y=219.48 sumN_x=1144 sumN_y=1240)
  - host `149.165.159.64` (n=6): sum W ratio=1.0095, sum N ratio=0.9899 (sumW_x=222.52 sumW_y=220.43 sumN_x=1271 sumN_y=1284)
- **longbench_v2: M3_native_bootstrap2_observe1 vs T_scope**
  - host `149.165.151.254` (n=6): sum W ratio=1.0249, sum N ratio=1.0612 (sumW_x=208.97 sumW_y=203.89 sumN_x=1144 sumN_y=1078)
  - host `149.165.159.64` (n=6): sum W ratio=1.0718, sum N ratio=1.1100 (sumW_x=222.52 sumW_y=207.62 sumN_x=1271 sumN_y=1145)
- **longbench_v2: B_native_bootstrap2_observe1 vs D_native**
  - host `149.165.151.254` (n=6): sum W ratio=0.9062, sum N ratio=0.9703 (sumW_x=219.48 sumW_y=242.20 sumN_x=1240 sumN_y=1278)
  - host `149.165.159.64` (n=6): sum W ratio=1.0985, sum N ratio=1.1867 (sumW_x=220.43 sumW_y=200.66 sumN_x=1284 sumN_y=1082)
- **longbench_v2: M1_native_bootstrap2_observe1 vs D_native**
  - host `149.165.151.254` (n=6): sum W ratio=0.9404, sum N ratio=0.9421 (sumW_x=227.76 sumW_y=242.20 sumN_x=1204 sumN_y=1278)
  - host `149.165.159.64` (n=6): sum W ratio=1.2836, sum N ratio=1.3115 (sumW_x=257.56 sumW_y=200.66 sumN_x=1419 sumN_y=1082)
- **longbench_v2: T_scope vs D_native**
  - host `149.165.151.254` (n=6): sum W ratio=0.8418, sum N ratio=0.8435 (sumW_x=203.89 sumW_y=242.20 sumN_x=1078 sumN_y=1278)
  - host `149.165.159.64` (n=6): sum W ratio=1.0347, sum N ratio=1.0582 (sumW_x=207.62 sumW_y=200.66 sumN_x=1145 sumN_y=1082)

