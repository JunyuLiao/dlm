
### longbench_v2_32k: reference dense_default_fix_ref medians: W 6.3 s, prefill 0.8 s, decode 5.4 s, N 176, C 12, N/C 14.5, S/N 30.3 ms

| arm | cells | W | prefill | decode S | N | C | N/C | S/N | logW: prefill + decode | logS: C + N/C + S/N |
|---|---|---|---|---|---|---|---|---|---|---|
| mage4096_PIECEWISE_fix_plain | 72 | 0.960 | 0.994 | 0.956 | 1.028 | 0.976 | 1.054 | 0.930 | -0.001 -0.040 | -0.025 +0.052 -0.073 |
| mage4096_PIECEWISE_fix_s20 | 72 | 0.962 | 0.994 | 0.956 | 1.021 | 0.971 | 1.052 | 0.937 | -0.001 -0.038 | -0.030 +0.051 -0.066 |
| method_m2c_k5_mass_PIECEWISE_fix_s20 | 72 | 1.098 | 0.992 | 1.114 | 1.171 | 1.039 | 1.128 | 0.951 | -0.001 +0.095 | +0.038 +0.120 -0.050 |
| method_m2c_r_PIECEWISE_fix_plain | 72 | 1.035 | 0.991 | 1.050 | 1.079 | 1.011 | 1.068 | 0.973 | -0.001 +0.036 | +0.011 +0.066 -0.027 |
| method_m2c_r_PIECEWISE_fix_s20 | 72 | 1.022 | 0.992 | 1.033 | 1.060 | 1.005 | 1.055 | 0.975 | -0.001 +0.023 | +0.005 +0.054 -0.025 |

### longbench_v2_64k: reference dense_default_fix_ref medians: W 7.9 s, prefill 2.2 s, decode 6.2 s, N 158, C 10, N/C 15.6, S/N 40.9 ms

| arm | cells | W | prefill | decode S | N | C | N/C | S/N | logW: prefill + decode | logS: C + N/C + S/N |
|---|---|---|---|---|---|---|---|---|---|---|
| mage4096_PIECEWISE_fix_plain | 72 | 0.887 | 1.000 | 0.850 | 1.005 | 0.975 | 1.031 | 0.846 | -0.000 -0.120 | -0.026 +0.031 -0.168 |
| mage4096_PIECEWISE_fix_s20 | 72 | 0.898 | 1.000 | 0.864 | 1.006 | 0.966 | 1.041 | 0.859 | -0.000 -0.107 | -0.034 +0.040 -0.152 |
| method_m2c_k5_mass_PIECEWISE_fix_s20 | 72 | 0.959 | 0.999 | 0.942 | 1.065 | 1.004 | 1.060 | 0.884 | -0.000 -0.041 | +0.004 +0.059 -0.123 |
| method_m2c_r_PIECEWISE_fix_plain | 72 | 0.904 | 0.998 | 0.867 | 0.983 | 0.955 | 1.030 | 0.882 | -0.000 -0.101 | -0.047 +0.029 -0.125 |
| method_m2c_r_PIECEWISE_fix_s20 | 72 | 0.870 | 0.998 | 0.828 | 0.924 | 0.943 | 0.980 | 0.896 | -0.001 -0.139 | -0.059 -0.020 -0.110 |

### longbench_v2_96k: reference dense_default_fix_ref medians: W 13.8 s, prefill 3.6 s, decode 10.4 s, N 215, C 12, N/C 19.5, S/N 47.4 ms

| arm | cells | W | prefill | decode S | N | C | N/C | S/N | logW: prefill + decode | logS: C + N/C + S/N |
|---|---|---|---|---|---|---|---|---|---|---|
| mage4096_PIECEWISE_fix_plain | 22 | 0.750 | 0.999 | 0.672 | 0.827 | 0.880 | 0.939 | 0.813 | -0.000 -0.288 | -0.128 -0.063 -0.207 |
| mage4096_PIECEWISE_fix_s20 | 22 | 0.863 | 0.999 | 0.800 | 0.945 | 0.961 | 0.984 | 0.847 | -0.000 -0.147 | -0.040 -0.016 -0.166 |
| method_m2c_k5_mass_PIECEWISE_fix_s20 | 22 | 0.744 | 1.000 | 0.671 | 0.773 | 0.827 | 0.935 | 0.867 | -0.000 -0.296 | -0.190 -0.068 -0.143 |
| method_m2c_r_PIECEWISE_fix_plain | 22 | 0.839 | 0.997 | 0.797 | 0.951 | 0.904 | 1.052 | 0.838 | -0.001 -0.175 | -0.101 +0.050 -0.177 |
| method_m2c_r_PIECEWISE_fix_s20 | 22 | 0.907 | 0.997 | 0.876 | 1.009 | 1.010 | 0.999 | 0.868 | -0.001 -0.096 | +0.010 -0.001 -0.141 |
