"""Resumable native adaptive DiffusionGemma generation, one attempt-0 receipt/request.

M1/M3 integration is supplied by ``--plugin module:function``. Its factory is
called as ``factory(adapter, config, condition)`` and returns a context manager
yielding an object (or mapping) with ``binding``, ``router``, ``state``, and
``counters``. ``state`` is a Junyu-compatible State, and this runner applies
``observe(model, state)`` exactly once. ``counters`` may be a callable returning
a JSON-compatible mapping after generation. The native dense control has no
attention binding; fresh T uses Junyu's original install/kernel/State.

Generation never reads expected answers. Score the immutable receipts offline.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import time
from types import MethodType
from typing import Any, Mapping


CONDITIONS = ("native_dense", "fresh_junyu_T", "M1", "M3")
GOLD_FIELDS = frozenset(("expected", "expected_answer", "answer", "reference_solution",
                         "solution", "gold", "target"))
SOURCE = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = SOURCE / "results/query_adaptive_v3/configs/frozen_policies.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _member(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty JSON array or JSONL")
    identities = set()
    for row in rows:
        if not isinstance(row, dict) or not all(k in row for k in ("id", "prompt")):
            raise ValueError("Every manifest row needs id and prompt")
        if str(row["id"]) in identities:
            raise ValueError(f"Duplicate manifest id: {row['id']}")
        identities.add(str(row["id"]))
        if not isinstance(row["prompt"], str) or not row["prompt"]:
            raise ValueError(f"Empty prompt: {row['id']}")
        if row.get("thinking", True) is not True:
            raise ValueError("This run requires thinking ON in every manifest row")
        if row.get("generation_budget", 8192) != 8192:
            raise ValueError("This run requires an 8192-token output budget")
    return rows


def _selected(rows: list[dict[str, Any]], ids: list[str], phase: str) -> list[dict[str, Any]]:
    requested = list(dict.fromkeys(ids))
    if len(requested) != len(ids) or not requested:
        raise ValueError("Pass distinct explicit --ids; implicit manifest-order selection is forbidden")
    by_id = {str(row["id"]): row for row in rows}
    missing = set(requested) - by_id.keys()
    if missing:
        raise ValueError(f"IDs absent from authorized manifest: {sorted(missing)}")
    if phase == "smoke" and len(requested) != 4:
        raise ValueError("Smoke is exactly four preselected authorized questions")
    return [by_id[id_] for id_ in requested]


def _source_hashes(plugin: str | None, *, library: Path | None, torch_library: Path | None,
                   extra_sources: list[Path]) -> dict[str, str]:
    paths = [Path(__file__), SOURCE / "src/dllm/models/adapters/diffusion_gemma.py",
             SOURCE / "experiments/value_direction_hopper/query_adaptive.py",
             SOURCE / "experiments/value_direction_hopper/integration.py",
             SOURCE / "experiments/value_direction_hopper/cuda.py",
             SOURCE / "experiments/value_direction_hopper/csrc/value_direction.cu",
             SOURCE / "experiments/value_direction_hopper/csrc/value_direction.h",
             SOURCE / "experiments/value_direction_hopper/csrc/torch_bridge.cpp",
             SOURCE / "experiments/diffusion_gemma_jl_output_aware/projections.py"]
    # Hash the whole small integration package, including cache/math files
    # imported by the plug-in, rather than only its public factory module.
    paths.extend(sorted(Path(__file__).parent.glob("*.py")))
    if plugin:
        module = importlib.import_module(plugin.split(":", 1)[0])
        module_file = inspect.getsourcefile(module)
        if module_file:
            paths.append(Path(module_file))
    if library is not None:
        paths.append(library)
    if torch_library is not None:
        paths.append(torch_library)
    paths.extend(extra_sources)
    try:
        from transformers.models.diffusion_gemma import generation_diffusion_gemma
        paths.append(Path(generation_diffusion_gemma.__file__))
    except ImportError:
        # The actual CUDA runner imports transformers before model loading.
        raise RuntimeError("Installed transformers lacks native DiffusionGemma generation")
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Required source/binary absent: {absent}")
    return {str(path.resolve()): _sha(path) for path in dict.fromkeys(paths)}


def _config(args: argparse.Namespace) -> dict[str, Any]:
    policy_file = args.policy.resolve()
    model_path = args.model.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Native model snapshot directory absent: {model_path}")
    frozen = json.loads(policy_file.read_text(encoding="utf-8"))
    policy = frozen["policies"][args.policy_name]
    if set(policy) != {"local", "global"}:
        raise ValueError("Policy must contain local and global thresholds")
    if args.condition == "fresh_junyu_T" and (args.library is None or args.torch_library is None):
        raise ValueError("Fresh Junyu T requires explicit matching --library and --torch-library")
    if args.condition in ("M1", "M3") and not args.plugin:
        raise ValueError("M1/M3 require a cache-consuming integration --plugin")
    if args.condition == "M3" and args.decision_interval not in (2, 3):
        raise ValueError("M3 decision interval must be 2, or 3 for the bounded sensitivity check")
    if args.condition == "M1" and args.decision_interval != 1:
        raise ValueError("M1 decision interval must be 1")
    model_metadata = [model_path / name for name in ("config.json", "generation_config.json",
                      "tokenizer_config.json", "preprocessor_config.json",
                      "model.safetensors.index.json")]
    config = dict(schema="numerical_qk_native_runner_v1", condition=args.condition,
                  phase=args.phase, ids=args.ids, seeds=args.seeds,
                  manifest=str(args.manifest.resolve()), manifest_sha256=_sha(args.manifest),
                  policy_file=str(policy_file), policy_sha256=_sha(policy_file),
                  policy_name=args.policy_name, policy=policy,
                  m_ref=float(frozen["m_ref"]), beta=float(frozen["beta"]), gamma=float(frozen["gamma"]),
                  model=str(model_path), revision=args.revision, dtype="bfloat16",
                  model_metadata_hashes={p.name: _sha(p) for p in model_metadata if p.is_file()},
                  max_new_tokens=8192, thinking=True, temperature=0.0, eos_enabled=True,
                  native_adaptive=True, decision_interval=args.decision_interval,
                  score_refresh_period=args.score_refresh_period, support=args.support,
                  output_mode=args.output_mode,
                  library=str(args.library.resolve()) if args.library else None,
                  torch_library=str(args.torch_library.resolve()) if args.torch_library else None,
                  plugin=args.plugin, diagnostic=args.diagnostic,
                  timing_events=args.timing_events,
                  source_hashes=_source_hashes(args.plugin, library=args.library,
                                               torch_library=args.torch_library,
                                               extra_sources=args.extra_source))
    config["fingerprint"] = _fingerprint(config)
    return config


class CanvasCalls:
    """Count actual native decoder calls without per-step device/host sync."""

    def __init__(self) -> None:
        self.canvases: list[dict[str, Any]] = []
        self._stops: list[Any] = []
        self._last_schedule_step: int | None = None

    @contextmanager
    def observe(self, model: Any):
        original = model._denoising_step
        had = "_denoising_step" in model.__dict__
        saved = model.__dict__.get("_denoising_step")

        def step(this: Any, **kwargs: Any):
            schedule_step = int(kwargs["cur_step"])
            if self._last_schedule_step is None or schedule_step >= self._last_schedule_step:
                self.canvases.append(dict(canvas_index=len(self.canvases), decoder_calls=0,
                                          schedule_steps=[]))
            current = self.canvases[-1]
            current["decoder_calls"] += 1
            current["schedule_steps"].append(schedule_step)
            self._last_schedule_step = schedule_step
            result = original(**kwargs)
            # Result[3] is the native stop tensor. Transfer all stop flags once
            # after generation; .item() here would synchronize every iteration.
            # Preserve the value even if the native sampler reuses a buffer.
            # clone() enqueues device work but does not synchronize the host.
            self._stops.append(result[3].detach().reshape(-1).clone())
            return result

        model._denoising_step = MethodType(step, model)
        try:
            yield self
        finally:
            if had:
                model._denoising_step = saved
            else:
                del model._denoising_step

    def finish(self, output: Any) -> list[dict[str, Any]]:
        import torch
        if not self.canvases:
            raise RuntimeError("No native decoder calls observed")
        flags = torch.cat(self._stops).to(device="cpu", dtype=torch.bool).tolist()
        if len(flags) != sum(c["decoder_calls"] for c in self.canvases):
            raise AssertionError("Native stop flag count differs from decoder calls")
        offset = 0
        canvas_length = int(output.metadata["native_canvas_length"])
        for canvas in self.canvases:
            steps = canvas["schedule_steps"]
            if any(a <= b for a, b in zip(steps, steps[1:])):
                raise AssertionError("Schedule steps did not strictly descend within a canvas")
            stop = flags[offset:offset + len(steps)]
            offset += len(steps)
            canvas["native_stop_final_call"] = bool(stop[-1])
            canvas["iteration_cap_final_call"] = steps[-1] == 1
            canvas["completion_slice_tokens"] = len(output.completion_tokens[
                canvas["canvas_index"] * canvas_length:(canvas["canvas_index"] + 1) * canvas_length])
        reported = output.metadata.get("actual_denoising_step_count")
        observed = sum(c["decoder_calls"] for c in self.canvases)
        if reported is not None and reported != observed:
            raise AssertionError(f"Adapter reports {reported} calls; observed {observed}")
        return self.canvases


class InitialPrefillTimeline:
    """Optional first encoder-end -> final generation-end CUDA event span.

    It is a device-clock timeline, including possible host gaps. It is neither
    request wall time nor a token/chunk emission clock. Later encoder forwards
    remain inside this span and are never subtracted as initial prefill.
    """

    def __init__(self, model: Any, enabled: bool) -> None:
        self.model = model
        self.enabled = enabled
        self.first_end: Any = None
        self.final_end: Any = None
        self.handles: list[Any] = []

    def __enter__(self):
        if self.enabled:
            import torch
            def after_encoder(*_args: Any) -> None:
                if self.first_end is None:
                    self.first_end = torch.cuda.Event(enable_timing=True)
                    self.first_end.record()
            for module in self.model.modules():
                if type(module).__name__ == "DiffusionGemmaEncoderModel":
                    self.handles.append(module.register_forward_hook(after_encoder))
        return self

    def mark_final(self) -> None:
        if self.enabled:
            import torch
            self.final_end = torch.cuda.Event(enable_timing=True)
            self.final_end.record()

    def seconds(self) -> float | None:
        if self.first_end is None or self.final_end is None:
            return None
        # Caller synchronizes once after generation before asking for elapsed.
        return self.first_end.elapsed_time(self.final_end) / 1000.0

    def __exit__(self, *_exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@contextmanager
def _runtime(adapter: Any, condition: str, config: Mapping[str, Any]):
    if condition == "native_dense":
        state = None
        if config["diagnostic"]:
            from experiments.value_direction_hopper.query_adaptive import State
            state = State("native_dense", None, m_ref=config["m_ref"], beta=config["beta"],
                          gamma=config["gamma"], diagnostics=True)
        yield dict(binding=None, router=None, state=state, counters=None)
        return
    if condition == "fresh_junyu_T":
        from experiments.diffusion_gemma_jl_output_aware.projections import Projections
        from experiments.value_direction_hopper.integration import install
        from experiments.value_direction_hopper.query_adaptive import State
        with install(adapter, config["library"], config["policy"], mode="value",
                     projections=Projections(), torch_library=config["torch_library"], collect=True) as (binding, router):
            state = State("T", router, m_ref=config["m_ref"], beta=config["beta"],
                          gamma=config["gamma"], diagnostics=bool(config["diagnostic"]))
            yield dict(binding=binding, router=router, state=state, counters=None)
        return
    module_name, function_name = str(config["plugin"]).split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    with factory(adapter, config, condition) as value:
        if _member(value, "router") is None or _member(value, "state") is None:
            raise ValueError("Reuse plugin must yield router and State")
        yield value


def _receipt_path(root: Path, phase: str, condition: str, seed: int, id_: str) -> Path:
    digest = hashlib.sha256(id_.encode()).hexdigest()[:20]
    return root / phase / condition / f"seed_{seed}" / f"{digest}.attempt0.json"


def _one(adapter: Any, row: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from dllm.models import GenerationRequest
    from experiments.value_direction_hopper.query_adaptive import observe

    condition = str(config["condition"])
    request = GenerationRequest(prompt=row["prompt"], max_new_tokens=8192,
                                temperature=0.0, seed=seed, extra={"thinking": True})
    calls = CanvasCalls()
    with _runtime(adapter, condition, config) as runtime:
        binding, router, state = (_member(runtime, key) for key in ("binding", "router", "state"))
        if binding is not None:
            from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _set_context
            _set_context(binding, {"benchmark": row.get("benchmark", "aime26"),
                                   "id": row["id"], "seed": seed})
        with (observe(adapter.model, state) if state is not None else nullcontext()):
            with InitialPrefillTimeline(adapter.model, bool(config["timing_events"])) as timeline:
                with calls.observe(adapter.model):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    output = adapter.generate(request)
                    timeline.mark_final()
                    torch.cuda.synchronize()
                    request_wall_seconds = time.perf_counter() - started
                generation_gpu_timeline_seconds = timeline.seconds()
        per_canvas = calls.finish(output)
        routing = router.records() if router is not None and hasattr(router, "records") else None
        counter_source = _member(runtime, "counters")
        counters = counter_source() if callable(counter_source) else counter_source
        diagnostics = (dict(steps=state.steps, canvases=state.canvases)
                       if config["diagnostic"] and state is not None else None)
    expected_hash = row.get("prompt_hash")
    prompt_hash = hashlib.sha256(row["prompt"].encode()).hexdigest()
    if expected_hash is not None and expected_hash != prompt_hash:
        raise AssertionError(f"Manifest prompt_hash mismatch for {row['id']}")
    if "prompt_tokens" in row and output.prompt_tokens != row["prompt_tokens"]:
        raise AssertionError(f"Tokenized prompt mismatch for {row['id']}")
    if output.metadata.get("thinking") is not True:
        raise AssertionError("Native generation did not use thinking ON")
    if output.metadata.get("sampling", {}).get("native_temperature_schedule") is not True:
        raise AssertionError("Native temperature schedule was overridden")
    if output.metadata.get("native_canvas_length") != 256:
        raise AssertionError("Native canvas changed; review protocol before continuing")
    if not output.metadata.get("special_token_ids", {}).get("eos_token_ids"):
        raise AssertionError("Native EOS token IDs absent")
    effective = output.metadata.get("denoising_configuration", {})
    expected_adaptive = dict(max_denoising_steps=48, t_min=.4, t_max=.8,
                             confidence_threshold=.005, stability_threshold=1,
                             entropy_bound=.1)
    for name, expected in expected_adaptive.items():
        observed = effective.get(name)
        if observed is None or abs(float(observed) - expected) > 1e-7:
            raise AssertionError(f"Native adaptive {name} changed: {observed}; expected {expected}")
    return dict(schema="numerical_qk_attempt0_v1", fingerprint=config["fingerprint"],
                attempt=0, phase=config["phase"], condition=condition, id=str(row["id"]),
                source_id=row.get("source_id"), seed=seed, prompt_hash=prompt_hash,
                prompt_token_hash=_fingerprint(output.prompt_tokens),
                request_wall_seconds=request_wall_seconds,
                generation_gpu_timeline_seconds=generation_gpu_timeline_seconds,
                generation_gpu_timeline_note=(
                    "First actual DiffusionGemmaEncoderModel forward end to final CUDA event after generate; "
                    "device timeline includes host gaps and later encoder/commit work; no token emission boundary"
                    if config["timing_events"] else "CUDA timing events disabled pending observer parity qualification"),
                generation_excluding_initial_prefill_seconds=None,
                generation_time_note="No nonintrusive synchronized wall boundary after initial prefill; separate profile required",
                completion_tokens=output.completion_tokens, output_tokens=len(output.completion_tokens),
                raw_completion=adapter.tokenizer.decode(output.completion_tokens, skip_special_tokens=False),
                prediction=output.text, termination_reason=output.termination_reason,
                metadata=output.metadata, per_canvas=per_canvas,
                total_decoder_calls=sum(c["decoder_calls"] for c in per_canvas),
                routing=routing, counters=counters, diagnostics=diagnostics)


def run(args: argparse.Namespace) -> None:
    import torch
    from dllm.models import create_adapter
    if not torch.cuda.is_available():
        raise RuntimeError("Native reproduction requires CUDA; CPU validates configuration only")
    # Drop answer fields before model loading or any generation. The offline
    # scorer reopens the manifest in a different process after receipts exist.
    rows = _selected([{key: value for key, value in row.items() if key not in GOLD_FIELDS}
                      for row in _rows(args.manifest)], args.ids, args.phase)
    config = _config(args)
    config_path = args.output / "configs" / f"{args.phase}.{args.condition}.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise RuntimeError(f"Frozen config changed; choose a new output directory: {config_path}")
    else:
        _atomic(config_path, config)
    adapter = create_adapter("diffusion_gemma", config["model"], device="cuda",
                             precision="bfloat16", revision=config["revision"]).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for row in rows:
        for seed in args.seeds:
            destination = _receipt_path(args.output, args.phase, args.condition, seed, str(row["id"]))
            if destination.exists():
                saved = json.loads(destination.read_text(encoding="utf-8"))
                if (saved.get("fingerprint"), saved.get("id"), saved.get("seed")) != (
                    config["fingerprint"], str(row["id"]), seed):
                    raise RuntimeError(f"Incompatible attempt-0 receipt: {destination}")
                print(json.dumps(dict(event="reuse_attempt0", condition=args.condition,
                                      id=row["id"], seed=seed, path=str(destination))), flush=True)
                continue
            receipt = _one(adapter, row, seed, config)
            _atomic(destination, receipt)
            print(json.dumps(dict(event="attempt0", condition=args.condition, id=row["id"],
                                  seed=seed, wall_seconds=receipt["request_wall_seconds"],
                                  decoder_calls=receipt["total_decoder_calls"],
                                  path=str(destination))), flush=True)


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--phase", choices=("smoke", "panel", "diagnostic"), default="smoke")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--policy-name", default="T_s50")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--torch-library", type=Path)
    parser.add_argument("--plugin", help="M1/M3 integration factory as module:function")
    parser.add_argument("--extra-source", type=Path, action="append", default=[],
                        help="Additional cache/math/executor source to hash in the frozen config")
    parser.add_argument("--decision-interval", type=int, default=1)
    parser.add_argument("--score-refresh-period", type=int, default=8)
    parser.add_argument("--support", choices=("legacy_junyu_mask", "native_mask"),
                        default="legacy_junyu_mask",
                        help="M1/M3 native-mask correction; native_dense/fresh_junyu_T are unaffected")
    parser.add_argument("--output-mode", choices=("cached_scores", "routing_only_current_output"),
                        default="cached_scores",
                        help="routing_only_current_output: stale-score routing decision, but current "
                             "QK for the final softmax/PV (Phase D successor); does not skip QK compute")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--timing-events", action="store_true",
                        help="Optional first encoder-end to final CUDA event device timeline")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate generation seeds")
    if args.phase == "smoke" and args.seeds != [42]:
        parser.error("First smoke uses exactly one generation seed, 42")
    if args.diagnostic != (args.phase == "diagnostic"):
        parser.error("--diagnostic is required only for a separate diagnostic phase")
    if args.score_refresh_period <= 0:
        parser.error("Score refresh period must be positive")
    if args.plugin and ":" not in args.plugin:
        parser.error("--plugin must be module:function")
    return args


if __name__ == "__main__":
    run(parse())
