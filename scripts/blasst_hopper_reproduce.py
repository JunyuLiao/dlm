#!/usr/bin/env python3
"""Build and benchmark the paper authors' exact Hopper BLASST kernels.

The artifact pins separate TensorRT-LLM revisions for the warp-specialized
prefill FMHA kernel and the K-first/conditional-V-load decode XQA kernel.  This
driver makes the BF16 decode experiment reproducible on an H100 without the
100 GB TensorRT-LLM container and writes machine-readable results.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "reference" / "blasst-ae-mlsys26"
ARTIFACT_URL = "https://github.com/cameronshinn/blasst-ae-mlsys26.git"
ARTIFACT_COMMIT = "108bf9b3f29e29c5b5c055edc52f5c42491a4509"
DECODE_COMMIT = "ae012977159bfa84e27b0305358d365fb97ee927"
PREFILL_COMMIT = "617440d385f091007daafe8bb7206e9a748bb84b"
XQA = ARTIFACT / "hopper_decode" / "TensorRT-LLM" / "cpp" / "kernels" / "xqa"
PREFILL_REPO = ARTIFACT / "hopper_prefill" / "TensorRT-LLM"
FMHA = PREFILL_REPO / "cpp" / "kernels" / "fmha_v2"
XQA_PATCH = ROOT / "patches" / "blasst-xqa-host-build.patch"
PREFILL_PATCH = ROOT / "patches" / "blasst-prefill-minimal-build.patch"
PREFILL_FACTORS = [0.0, 0.8, 1.2, 2.0, 4.0]
FULL_PREFILL_FACTORS = [
    0.0, 0.5, 0.6, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2,
    1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0,
]


def run(
    cmd: list[str], *, cwd: Path = ROOT, capture: bool = False, extra_env: dict[str, str] | None = None
) -> str:
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        env={
            **os.environ,
            "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
            **(extra_env or {}),
        },
    )
    return result.stdout or ""


def prepare() -> None:
    if not (ARTIFACT / ".git").exists():
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", ARTIFACT_URL, str(ARTIFACT)])
    run(["git", "checkout", ARTIFACT_COMMIT], cwd=ARTIFACT)
    run(
        [
            "git",
            "submodule",
            "update",
            "--init",
            "--depth",
            "1",
            "hopper_prefill/TensorRT-LLM",
            "hopper_decode/TensorRT-LLM",
        ],
        cwd=ARTIFACT,
    )
    assert run(["git", "rev-parse", "HEAD"], cwd=ARTIFACT, capture=True).strip() == ARTIFACT_COMMIT
    assert run(["git", "rev-parse", "HEAD"], cwd=XQA.parents[2], capture=True).strip() == DECODE_COMMIT
    assert run(["git", "rev-parse", "HEAD"], cwd=PREFILL_REPO, capture=True).strip() == PREFILL_COMMIT

    decode_repo = XQA.parents[2]
    for repo, patch in ((decode_repo, XQA_PATCH), (PREFILL_REPO, PREFILL_PATCH)):
        if subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=repo).returncode != 0:
            run(["git", "apply", str(patch)], cwd=repo)


def configure_and_build(skip: bool, jobs: int) -> Path:
    build = XQA / ("build_skip_bf16" if skip else "build_noskip_bf16")
    flags = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_XQA_TESTS=ON",
        "-DBUILD_XQA_NVRTC_TESTS=OFF",
        "-DHEAD_ELEMS=128",
        "-DHEAD_GRP_SIZE=16",
        "-DINPUT_FP16=0",
        "-DCACHE_ELEM_ENUM=0",
        f"-DSKIP_SOFTMAX_ATTN={'ON' if skip else 'OFF'}",
        f"-DSKIP_SOFTMAX_ATTN_BLOCK_STATS={'ON' if skip else 'OFF'}",
    ]
    run(["cmake", "-S", str(XQA), "-B", str(build), "-U", "NVRTC_LIB", *flags])
    run(["cmake", "--build", str(build), "-j", str(jobs)])
    return build


def benchmark(binary: Path, log: Path) -> str:
    output = run([str(binary), "--gtest_filter=Perf.skip_softmax_attn"], cwd=binary.parent, capture=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(output)
    print(output)
    return output


def parse_dense(text: str) -> dict[int, float]:
    pattern = re.compile(
        r"seqLen: (\d+), original attention kernel\n"
        r"dramSolRatio: [\d.]+% \(([\d.]+) ms,"
    )
    return {int(seq): float(ms) for seq, ms in pattern.findall(text)}


def parse_skip(text: str) -> list[dict[str, float | int]]:
    pattern = re.compile(
        r"seqLen: (\d+), skipSoftmaxThreshold: ([\d.]+)\n"
        r"kernel skippedBlockCount: (\d+)/(\d+) \(([\d.]+)%\)\n"
        r"dramSolRatio: [\d.]+% \(([\d.]+) ms,"
    )
    rows: list[dict[str, float | int]] = []
    for seq, threshold, skipped, total, sparsity, ms in pattern.findall(text):
        rows.append(
            {
                "sequence_length": int(seq),
                "threshold": float(threshold),
                "skipped_blocks": int(skipped),
                "total_blocks": int(total),
                "sparsity_percent": float(sparsity),
                "time_ms": float(ms),
            }
        )
    return rows


def write_results(dense_text: str, skip_text: str, output_dir: Path) -> None:
    dense = parse_dense(dense_text)
    rows = parse_skip(skip_text)
    if not dense or not rows:
        raise RuntimeError("Could not parse the XQA benchmark output")
    for row in rows:
        baseline = dense[int(row["sequence_length"])]
        row["dense_time_ms"] = baseline
        row["speedup"] = baseline / float(row["time_ms"])
    path = output_dir / "decode_bf16_h100.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def build_prefill(enable_stats: bool, jobs: int) -> Path:
    env = {
        "CUDA": "/usr/local/cuda",
        "TMPDIR": "/tmp/blasst-prefill-build",
        "ENABLE_SM90": "1",
        "BLASST_AE_MINIMAL": "1",
    }
    if (FMHA / "generated" / "makefile").exists():
        run(["make", "clean"], cwd=FMHA, extra_env=env)
    run(["python3", "setup.py"], cwd=FMHA, extra_env=env)
    run(["make", "dirs"], cwd=FMHA, extra_env=env)
    cmd = ["make", "-j", str(jobs)]
    if enable_stats:
        cmd.extend(["CXXFLAGS=-DSKIP_SOFTMAX_STAT", "CUDAFLAGS=-DSKIP_SOFTMAX_STAT"])
    cmd.append("bin/fmha.exe")
    run(cmd, cwd=FMHA, extra_env=env)
    return FMHA / "bin" / "fmha.exe"


def run_prefill_case(binary: Path, seq: int, factor: float, *, repeats: int, warmups: int) -> str:
    cmd = [
        str(binary),
        "-bf16",
        "-d",
        "128",
        "-b",
        "1",
        "-h",
        "64",
        "-gqa",
        "4",
        "-s",
        str(seq),
        "-runs",
        str(repeats),
        "-warm-up-runs",
        str(warmups),
        "-skip-checks",
    ]
    if factor > 0:
        cmd.extend(["-skip-softmax-threshold-scale-factor", str(factor * seq)])
    return run(cmd, cwd=FMHA, capture=True)


def benchmark_prefill(
    output_dir: Path, jobs: int, factors: list[float], sequences: tuple[int, ...]
) -> None:
    sparsity: dict[tuple[int, float], float] = {}
    stats_binary = output_dir / "fmha_stats.exe"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not stats_binary.exists():
        shutil.copy2(build_prefill(True, jobs), stats_binary)
    stats_log: list[str] = []
    stats_log_path = output_dir / "prefill_stats.log"
    for seq in sequences:
        for factor in factors:
            text = run_prefill_case(stats_binary, seq, factor, repeats=1, warmups=0)
            stats_log.append(f"seq={seq} factor={factor}\n{text}")
            stats_log_path.write_text("\n".join(stats_log))
            if factor == 0:
                sparsity[(seq, factor)] = 0.0
            else:
                match = re.search(r"Skip-Softmax \.\:\s*\d+\s*/\s*\d+\s*=\s*([\d.]+)%", text)
                if not match:
                    raise RuntimeError(f"Missing prefill sparsity for seq={seq}, factor={factor}")
                sparsity[(seq, factor)] = float(match.group(1))

    perf_binary = output_dir / "fmha_perf.exe"
    if not perf_binary.exists():
        shutil.copy2(build_prefill(False, jobs), perf_binary)
    rows: list[dict[str, float | int]] = []
    perf_log: list[str] = []
    perf_log_path = output_dir / "prefill_perf.log"
    for seq in sequences:
        times: dict[float, float] = {}
        for factor in factors:
            text = run_prefill_case(perf_binary, seq, factor, repeats=10, warmups=3)
            perf_log.append(f"seq={seq} factor={factor}\n{text}")
            perf_log_path.write_text("\n".join(perf_log))
            match = re.search(r"Fused time\s*\.*\:\s*([\d.]+)\s*us", text)
            if not match:
                raise RuntimeError(f"Missing prefill time for seq={seq}, factor={factor}")
            times[factor] = float(match.group(1)) / 1000.0
        baseline = times[0.0]
        for factor in factors:
            rows.append(
                {
                    "sequence_length": seq,
                    "threshold_factor": factor,
                    "sparsity_percent": sparsity[(seq, factor)],
                    "dense_time_ms": baseline,
                    "time_ms": times[factor],
                    "speedup": baseline / times[factor],
                }
            )

    stats_log_path.write_text("\n".join(stats_log))
    perf_log_path.write_text("\n".join(perf_log))
    path = output_dir / "prefill_bf16_h100.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--phase", choices=("decode", "prefill", "both"), default="both")
    parser.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument(
        "--prefill-full-sweep",
        action="store_true",
        help="run the artifact's 17 factors at both 16K and 64K (slow)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "blasst_hopper_reproduction",
    )
    args = parser.parse_args()

    prepare()
    if args.prepare_only:
        return
    if args.phase in ("decode", "both"):
        dense_build = XQA / "build_noskip_bf16"
        skip_build = XQA / "build_skip_bf16"
        if not args.skip_build:
            dense_build = configure_and_build(False, args.jobs)
            skip_build = configure_and_build(True, args.jobs)
        for build in (dense_build, skip_build):
            if not (build / "unitTests").exists():
                raise FileNotFoundError(f"Missing {build / 'unitTests'}; rerun without --skip-build")

        dense_text = benchmark(dense_build / "unitTests", args.output_dir / "decode_dense.log")
        skip_text = benchmark(skip_build / "unitTests", args.output_dir / "decode_blasst.log")
        write_results(dense_text, skip_text, args.output_dir)
    if args.phase in ("prefill", "both"):
        if args.skip_build:
            raise ValueError("--skip-build is only supported for the decode phase")
        factors = FULL_PREFILL_FACTORS if args.prefill_full_sweep else PREFILL_FACTORS
        sequences = (16384, 65536) if args.prefill_full_sweep else (65536,)
        benchmark_prefill(args.output_dir, args.jobs, factors, sequences)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
