"""Separate one-question native DiffusionGemma stopping-signal diagnostic.

The installed native StableAndConfidentStoppingCriteria is called exactly once
per decoder iteration. This observer records its two inputs/predicates before
delegating to it, and records the native entropy-bound sampler's accepted mask.
It does not replace the sampler, change model weights, or produce quality rows.
All device values are transferred only after the request completes. A plain vs
observed native-dense run on the same model instance is a mandatory parity gate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import time
from types import MethodType
from typing import Any
import uuid

from experiments.numerical_qk_reuse import runner


EXPECTED_NATIVE_SOURCE_SHA256 = "b814a6fc41492794c1f50ba71c750206684e25dfdd53932e0842675871fef0de"
PLUGIN = "experiments.numerical_qk_reuse.integration:install"
CONDITIONS = ("native_dense", "fresh_junyu_T", "M1", "M3")
IDENTITY = runner.SOURCE / "results/numerical_qk_reuse_20260924/smoke_manifest_identity.json"


def _scalar(value: Any) -> int | float | bool:
    values = value.detach().to(device="cpu").reshape(-1).tolist()
    if len(values) != 1:
        raise ValueError("Stop diagnostic currently supports batch size one")
    return values[0]


class SamplerSpy:
    """Observe native acceptance and renoising while delegating unchanged."""

    def __init__(self, inner: Any, record: dict):
        self.inner, self.record = inner, record

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def accept_canvas(self, current: Any, proposed: Any, logits: Any, cur_step: Any):
        result = self.inner.accept_canvas(current, proposed, logits, cur_step)
        mask = self.inner.accepted_token_mask
        self.record["accepted"] = mask.sum().detach().clone()
        self.record["canvas_positions"] = int(mask.shape[-1])
        return result

    def renoise_canvas(self, accepted_canvas: Any, cur_step: Any):
        result = self.inner.renoise_canvas(accepted_canvas, cur_step)
        self.record["renoised"] = (~self.inner.accepted_token_mask).sum().detach().clone()
        return result


class StopSpy:
    """Read the exact native stable/confident predicates, then call native code."""

    def __init__(self, inner: Any, record: dict, previous_top: Any):
        if type(inner).__name__ != "StableAndConfidentStoppingCriteria":
            raise ValueError(f"Unsupported native stopper: {type(inner).__name__}")
        self.inner, self.record, self.previous_top = inner, record, previous_top

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def __call__(self, argmax_canvas: Any, logits: Any, **kwargs: Any):
        import torch
        if self.inner.stability_threshold == 0:
            stable = torch.ones((logits.shape[0],), device=logits.device, dtype=torch.bool)
        else:
            history = self.inner.argmax_canvas_history
            if history is None:
                stable = torch.zeros((argmax_canvas.shape[0],), device=argmax_canvas.device,
                                     dtype=torch.bool)
            else:
                stable = (history == argmax_canvas[None, :, :]).all(dim=-1).all(dim=0)
        entropy = torch.distributions.Categorical(logits=logits).entropy().mean(dim=-1)
        confident = entropy < self.inner.confidence_threshold
        self.record["stable"] = stable.detach().clone()
        self.record["confident"] = confident.detach().clone()
        self.record["mean_processed_entropy"] = entropy.detach().clone()
        if self.previous_top is not None:
            flips = (argmax_canvas != self.previous_top).sum(dim=-1)
            self.record["top1_flips_vs_previous_completed"] = flips.detach().clone()
        else:
            self.record["top1_flips_vs_previous_completed"] = None
        result = self.inner(argmax_canvas, logits, **kwargs)
        self.record["native_criterion_stop"] = result.detach().clone()
        return result


class NativeStopObserver:
    def __init__(self):
        self.steps: list[dict] = []
        self.canvas = -1
        self.last_schedule_step: int | None = None
        self.previous_top: Any = None
        self.call_in_canvas = 0

    @contextmanager
    def observe(self, model: Any):
        original = model._denoising_step
        had = "_denoising_step" in model.__dict__
        saved = model.__dict__.get("_denoising_step")

        def step(this: Any, **kwargs: Any):
            schedule_step = int(kwargs["cur_step"])
            if self.last_schedule_step is None or schedule_step >= self.last_schedule_step:
                self.canvas += 1
                self.previous_top = None
                self.call_in_canvas = 0
            self.last_schedule_step = schedule_step
            self.call_in_canvas += 1
            record = dict(canvas_index=self.canvas,
                          decoder_call=self.call_in_canvas,
                          remaining_schedule_step=schedule_step)
            criterion = kwargs.get("diffusion_stopping_criteria")
            if criterion is None:
                raise RuntimeError("Native stable/confident stopper is disabled")
            kwargs["sampler"] = SamplerSpy(kwargs["sampler"], record)
            kwargs["diffusion_stopping_criteria"] = StopSpy(criterion, record, self.previous_top)
            result = original(**kwargs)
            record["native_finished_denoising"] = result[3].detach().clone()
            self.previous_top = result[1].detach().clone()
            self.steps.append(record)
            return result

        model._denoising_step = MethodType(step, model)
        try:
            yield self
        finally:
            if had:
                model._denoising_step = saved
            else:
                del model._denoising_step

    def finish(self, output: Any) -> tuple[list[dict], list[dict]]:
        if not self.steps:
            raise RuntimeError("No native denoising calls were observed")
        decoded = []
        required = ("accepted", "renoised", "stable", "confident",
                    "mean_processed_entropy", "native_criterion_stop",
                    "native_finished_denoising")
        for record in self.steps:
            missing = set(required) - record.keys()
            if missing:
                raise RuntimeError(f"Native diagnostic hook missed {sorted(missing)}")
            row = {key: (_scalar(value) if hasattr(value, "detach") else value)
                   for key, value in record.items()}
            row["top1_agreement_positions"] = (None if row["top1_flips_vs_previous_completed"] is None
                                                  else row["canvas_positions"] - row["top1_flips_vs_previous_completed"])
            row["iteration_cap"] = row["remaining_schedule_step"] == 1
            if row["accepted"] + row["renoised"] != row["canvas_positions"]:
                raise AssertionError("Native sampler accepted/renoised partition changed")
            if (row["stable"] and row["confident"]) != row["native_criterion_stop"]:
                raise AssertionError("Observed native stop predicates differ from criterion result")
            if row["native_criterion_stop"] != row["native_finished_denoising"]:
                raise AssertionError("Native criterion result differs from finished_denoising in batch-one run")
            decoded.append(row)
        observed_count = len(decoded)
        if output.metadata.get("actual_denoising_step_count") != observed_count:
            raise AssertionError("Native reported decoder calls differ from observed calls")
        canvas_length = int(output.metadata["native_canvas_length"])
        canvases = []
        for index in range(self.canvas + 1):
            group = [row for row in decoded if row["canvas_index"] == index]
            if not group:
                raise AssertionError("Empty observed canvas")
            final = group[-1]
            canvases.append(dict(canvas_index=index, actual_decoder_calls=len(group),
                                 native_stable_final_call=final["stable"],
                                 native_confident_final_call=final["confident"],
                                 native_stop_final_call=final["native_criterion_stop"],
                                 iteration_cap_final_call=final["iteration_cap"],
                                 completion_slice_tokens=len(output.completion_tokens[
                                     index*canvas_length:(index+1)*canvas_length]),
                                 request_eos_boundary=(index == self.canvas and output.termination_reason == "eos"),
                                 request_output_cap_boundary=(index == self.canvas and
                                                              output.termination_reason == "length" and
                                                              len(output.completion_tokens) >= 8192)))
        return decoded, canvases


def _args_for(args: argparse.Namespace, condition: str) -> argparse.Namespace:
    return argparse.Namespace(manifest=args.manifest, output=args.output, model=args.model,
                              revision=args.revision, phase="stop_diagnostic",
                              condition=condition, ids=[args.id], seeds=[42], policy=args.policy,
                              policy_name=args.policy_name, library=args.library,
                              torch_library=args.torch_library,
                              plugin=PLUGIN if condition in ("M1", "M3") else None,
                              extra_source=[Path(__file__)], decision_interval=2 if condition == "M3" else 1,
                              score_refresh_period=8, support="legacy_junyu_mask",
                              diagnostic=False, timing_events=False)


def _config(args: argparse.Namespace, condition: str) -> dict:
    config = runner._config(_args_for(args, condition))
    source = [(path, digest) for path, digest in config["source_hashes"].items()
              if path.endswith("generation_diffusion_gemma.py")]
    if len(source) != 1 or source[0][1] != EXPECTED_NATIVE_SOURCE_SHA256:
        raise RuntimeError("Installed native generation source differs from audited stopper implementation")
    return config


def _validate_native_output(output: Any) -> None:
    metadata = output.metadata
    if metadata.get("thinking") is not True:
        raise AssertionError("Native diagnostic did not use thinking ON")
    if metadata.get("sampling", {}).get("native_temperature_schedule") is not True:
        raise AssertionError("Native temperature schedule was overridden")
    if metadata.get("native_canvas_length") != 256:
        raise AssertionError("Native canvas length changed")
    if not metadata.get("special_token_ids", {}).get("eos_token_ids"):
        raise AssertionError("Native EOS token IDs absent")
    expected = dict(max_denoising_steps=48, t_min=.4, t_max=.8,
                    confidence_threshold=.005, stability_threshold=1, entropy_bound=.1)
    actual = metadata.get("denoising_configuration", {})
    for name, value in expected.items():
        if name not in actual or abs(float(actual[name]) - value) > 1e-7:
            raise AssertionError(f"Native adaptive {name} changed: {actual.get(name)}")


def _plain(adapter: Any, row: dict, config: dict) -> dict:
    import torch
    from dllm.models import GenerationRequest
    request = GenerationRequest(prompt=row["prompt"], max_new_tokens=8192,
                                temperature=0.0, seed=42, extra={"thinking": True})
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = adapter.generate(request)
    torch.cuda.synchronize()
    _validate_native_output(output)
    return dict(schema="numerical_qk_stop_plain_v1", fingerprint=config["fingerprint"],
                id=row["id"], seed=42,
                prompt_hash=hashlib.sha256(row["prompt"].encode()).hexdigest(),
                completion_tokens=output.completion_tokens,
                raw_completion=adapter.tokenizer.decode(output.completion_tokens, skip_special_tokens=False),
                metadata=output.metadata, termination_reason=output.termination_reason,
                request_wall_seconds=time.perf_counter()-started,
                quality_eligible=False, timing_eligible=False)


def _diagnose(adapter: Any, row: dict, config: dict) -> dict:
    import torch
    from dllm.models import GenerationRequest
    from experiments.value_direction_hopper.query_adaptive import observe
    request = GenerationRequest(prompt=row["prompt"], max_new_tokens=8192,
                                temperature=0.0, seed=42, extra={"thinking": True})
    trace = NativeStopObserver()
    with runner._runtime(adapter, config["condition"], config) as runtime:
        state = runner._member(runtime, "state")
        with (observe(adapter.model, state) if state is not None else nullcontext()):
            with trace.observe(adapter.model):
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = adapter.generate(request)
                torch.cuda.synchronize()
                wall = time.perf_counter() - started
        _validate_native_output(output)
        steps, canvases = trace.finish(output)
        router = runner._member(runtime, "router")
        counters_source = runner._member(runtime, "counters")
        counters = counters_source() if callable(counters_source) else counters_source
        routing = router.records() if router is not None and hasattr(router, "records") else None
    return dict(schema="numerical_qk_stop_diagnostic_v1", fingerprint=config["fingerprint"],
                id=row["id"], seed=42, condition=config["condition"],
                prompt_hash=hashlib.sha256(row["prompt"].encode()).hexdigest(),
                completion_tokens=output.completion_tokens,
                raw_completion=adapter.tokenizer.decode(output.completion_tokens, skip_special_tokens=False),
                termination_reason=output.termination_reason, metadata=output.metadata,
                request_wall_seconds_diagnostic_only=wall, quality_eligible=False,
                timing_eligible=False, actual_decoder_calls=len(steps),
                per_step=steps, per_canvas=canvases, routing=routing, counters=counters)


def run(args: argparse.Namespace) -> None:
    import torch
    from dllm.models import create_adapter
    if not torch.cuda.is_available():
        raise RuntimeError("Separate stop diagnostic requires CUDA")
    allowed = json.loads(IDENTITY.read_text(encoding="utf-8"))["selected_ids"]
    if args.id not in allowed:
        raise ValueError("Diagnostic ID must be one of the four frozen smoke IDs")
    rows = [{key: value for key, value in row.items() if key not in runner.GOLD_FIELDS}
            for row in runner._rows(args.manifest)]
    row = next((value for value in rows if value["id"] == args.id), None)
    if row is None:
        raise ValueError("Diagnostic ID absent from authorized manifest")
    session = uuid.uuid4().hex
    base = args.output / "stop_diagnostic" / session
    failure = base / "failure.json"
    condition = "plain_native_dense"
    try:
        configs = {name: _config(args, name) for name in set(args.conditions) | {"native_dense"}}
        runner._atomic(base / "configs.json", configs)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        adapter = create_adapter("diffusion_gemma", str(args.model.resolve()), device="cuda",
                                 precision="bfloat16", revision=args.revision).load()
        plain = _plain(adapter, row, configs["native_dense"])
        runner._atomic(base / "plain_native_dense.json", plain)
        condition = "observed_native_dense"
        observed = _diagnose(adapter, row, configs["native_dense"])
        runner._atomic(base / "observed_native_dense.json", observed)
        counts_match = (plain["metadata"].get("actual_denoising_step_count") ==
                        observed["actual_decoder_calls"])
        tokens_match = plain["completion_tokens"] == observed["completion_tokens"]
        gate = dict(schema="numerical_qk_stop_observer_gate_v1", id=args.id,
                    same_model_instance=True, tokens_match=tokens_match,
                    decoder_counts_match=counts_match,
                    passed=bool(tokens_match and counts_match),
                    note="Diagnostic records are separate from quality and formal timing")
        runner._atomic(base / "gate.json", gate)
        if not gate["passed"]:
            raise AssertionError("Native stop observer changed tokens or decoder calls")
        for name in args.conditions:
            if name == "native_dense":
                continue  # observed_native_dense is the diagnostic record
            condition = name
            result = _diagnose(adapter, row, configs[name])
            runner._atomic(base / f"{name}.json", result)
        print(json.dumps(dict(event="stop_diagnostic_complete", session=session,
                              id=args.id, conditions=args.conditions, directory=str(base))), flush=True)
    except Exception as error:
        if not failure.exists():
            runner._atomic(failure, dict(schema="numerical_qk_stop_failure_v1",
                                         id=args.id, session=session, condition=condition,
                                         error_type=type(error).__name__, error=str(error)))
        raise


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--id", default="aime26/2",
                        help="One frozen smoke ID; defaults to aime26/2")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=["M1"],
                        help="Diagnostic arms after plain/observed dense parity; defaults to M1 only")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--torch-library", type=Path)
    parser.add_argument("--policy", type=Path, default=runner.DEFAULT_POLICY)
    parser.add_argument("--policy-name", default="T_s50")
    args = parser.parse_args(argv)
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("Duplicate conditions")
    return args


if __name__ == "__main__":
    run(parse())
