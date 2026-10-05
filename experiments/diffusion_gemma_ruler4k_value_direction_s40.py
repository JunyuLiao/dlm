"""Focused RULER4K value-direction sweep at 40% physical tile sparsity.

This is a thin, versioned adapter over the audited RULER4K implementation.  It
keeps the frozen routing kernels and calibration code unchanged while selecting
one intermediate target and retaining BLASST plus the full-dimensional and
Gaussian projected centered routers.
"""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import shutil
from unittest.mock import patch
import json
import time
import sys

from experiments import diffusion_gemma_ruler4k_gaussian_sweep as source

ROOT = Path("results/diffusion_gemma_ruler4k_value_direction_s40_v23")
MODULE = "experiments.diffusion_gemma_ruler4k_value_direction_s40"
TARGETS = (0.40,)
BASELINES = {"blasst": {"method": "blasst"}}
PROJECTED = {"jl_gaussian_r32": {"family": "gaussian", "rank": 32}}
CONFIGS = {**BASELINES, **PROJECTED}
CONDITIONS = ["dense"] + [f"{name}_s40" for name in CONFIGS]
EXPECTED = 130 * len(CONDITIONS)

_ORIGINAL = {
    "prepare": source.prepare,
    "execution": source.execution,
    "work": source.work,
    "report": source.report,
    "verify": source.verify,
    "launch": source.launch,
    "supervise": source.supervise,
}


def _ensure_tests(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dst = root / "tests.xml"
    if not dst.exists():
        shutil.copy2(Path("results/diffusion_gemma_ruler4k_gaussian_rank_sweep_v16/tests.xml"), dst)


@contextmanager
def _constants():
    attrs = {
        "ROOT": ROOT,
        "MODULE": MODULE,
        "TARGETS": TARGETS,
        "BASELINES": BASELINES,
        "PROJECTED": PROJECTED,
        "CONFIGS": CONFIGS,
        "CONDITIONS": CONDITIONS,
        "EXPECTED": EXPECTED,
    }
    with ExitStack() as stack:
        for key, value in attrs.items():
            stack.enter_context(patch.object(source, key, value))
        yield


def prepare(root=ROOT):
    _ensure_tests(root)
    with _constants():
        return _ORIGINAL["prepare"](root)


def execution(root=ROOT):
    _ensure_tests(root)
    contract_path = Path(root) / "execution_contract.json"
    if contract_path.exists():
        existing = source.base.read(contract_path)
        if (existing.get("schema") == "ruler4k_value_direction_s40_v23"
                and str(Path(__file__)) in existing.get("sources", {})):
            return existing
    with _constants():
        contract = _ORIGINAL["execution"](root)
    # Include this adapter in provenance while retaining the inherited frozen
    # implementation source hashes.
    contract = dict(contract)
    sources = dict(contract["sources"])
    sources[str(Path(__file__))] = source._sha(Path(__file__).read_bytes())
    contract["sources"] = sources
    contract["schema"] = "ruler4k_value_direction_s40_v23"
    contract["provenance_note"] = "Uses the current working-tree JL helper sources; prior 50/60/70 runs remain frozen historical controls."
    contract["fingerprint"] = source.base._fingerprint(contract)
    source.base._write(contract_path, contract)
    return contract


@contextmanager
def _patched():
    attrs = {
        "ROOT": ROOT,
        "MODULE": MODULE,
        "TARGETS": TARGETS,
        "BASELINES": BASELINES,
        "PROJECTED": PROJECTED,
        "CONFIGS": CONFIGS,
        "CONDITIONS": CONDITIONS,
        "EXPECTED": EXPECTED,
        "prepare": prepare,
        "execution": execution,
        "report": report,
        "verify": verify,
    }
    with ExitStack() as stack:
        for key, value in attrs.items():
            stack.enter_context(patch.object(source, key, value))
        yield


def work(root=ROOT):
    _ensure_tests(root)
    with _patched():
        return _ORIGINAL["work"](root)


def launch(root=ROOT):
    _ensure_tests(root)
    with _patched():
        return _ORIGINAL["launch"](root)


def supervise(root=ROOT):
    _ensure_tests(root)
    with _patched():
        return _ORIGINAL["supervise"](root)


def report(root=ROOT):
    _ensure_tests(root)
    with _patched():
        result = _ORIGINAL["report"](root)
    # The parent wrapper has a fixed output-count prose replacement.
    path = Path(root) / "report.md"
    text = path.read_text().replace("2470", str(EXPECTED))
    path.write_text(text)
    result["artifacts"]["report.md"] = source._sha(path.read_bytes())
    source.base._write(Path(root) / "audit.json", result)
    return result


def verify(root=ROOT):
    before = source.base.read(Path(root) / "audit.json")
    if not before.get("complete") or before.get("completed") != EXPECTED:
        raise ValueError("Incomplete focused final sweep")
    source.evidence.check_sources({str(Path(root) / p): h for p, h in before["artifacts"].items()})
    if report(root) != before:
        raise ValueError("Raw-only regeneration differs")
    result = dict(passed=True, completed=EXPECTED, inference_performed=False,
                  audit_sha256=source._sha((Path(root) / "audit.json").read_bytes()))
    source.base._write(Path(root) / "regeneration_verification.json", result)
    return result


def main():
    raise RuntimeError('Superseded failed attempt: use experiments.value_direction_hopper.sparsity_steps. '
                       'The original source contract is preserved for audit; it must not be resumed.')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "execution", "work", "launch", "supervise", "report", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.output
    if args.command == "prepare": out = prepare(root)
    elif args.command == "execution": out = execution(root)
    elif args.command == "work": out = work(root)
    elif args.command == "launch": out = launch(root)
    elif args.command == "supervise": out = supervise(root)
    elif args.command == "report": out = report(root)
    else: out = verify(root)
    if out is not None and args.command not in ("work", "launch", "supervise"):
        print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
