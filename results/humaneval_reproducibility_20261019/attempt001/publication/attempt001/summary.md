# HumanEval baseline reproducibility diagnostic

A runtime choice of reduction configuration changes rounding in fused post-MLP normalization on identical inputs. R0_BLOCK512 versus1024 changes11 BF16 layer-output elements,86596 internal FP32 elements and10 BF16 elements entering the next Q/K/V projection (481536 valid elements). Each configuration repeats exactly and matches its corresponding saved output.

The qualified policy explicitly applies deterministic Inductor options to the model and three unchanged native helper Python functions. Horizontal combo fusion and benchmarking are disabled; native FULL/PIECEWISE graphs and sampler formulas/stopping remain. All53 recorded kernels receipt deterministic settings. Environment-only and model-only failures are retained.

Independent unobserved FULL engines match12/12 output/N/C/T pairs. Eleven further dense/V31 engines, both thinking modes, pass all declared full pre/post sampler-state, valid-history, emission and RNG comparisons; bookends match and measured captures are zero. This is a complete six-question baseline diagnostic, not a164-question accuracy study.

Diagnosis reservation: 10477.365/10800 seconds. Original results and older pauses are preserved. Corrected K256/K128 matrices require their own method, replication and official-scoring completion seals.
