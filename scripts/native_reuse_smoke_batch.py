"""One-load, four-question native T/M1/M3 smoke with an optional observer gate.

Each arm uses the existing runner's immutable config and attempt-0 receipt
format. The optional observer qualification runs three native-dense generations
on the SAME model instance before enabling CUDA timing events for the arms.
Optional dense timing controls reuse the third qualification output and run the
remaining three questions with events on; they never replace quality outputs.
Generation never receives answer fields and stops on the first failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid
from typing import Any

from experiments.numerical_qk_reuse import runner


ARM_ORDER = ("fresh_junyu_T", "M1", "M3")
PLUGIN = "experiments.numerical_qk_reuse.integration:install"
IDENTITY = runner.SOURCE / "results/numerical_qk_reuse_20260924/smoke_manifest_identity.json"


def _immutable(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Frozen output differs: {path}")
        return
    runner._atomic(path, value)


def _arm_args(args: argparse.Namespace, condition: str, *, timing_events: bool,
              phase: str = "smoke") -> argparse.Namespace:
    return argparse.Namespace(manifest=args.manifest, output=args.output, model=args.model,
                              revision=args.revision, phase=phase, condition=condition,
                              ids=args.ids, seeds=[42], policy=args.policy,
                              policy_name=args.policy_name, library=args.library,
                              torch_library=args.torch_library,
                              plugin=PLUGIN if condition in ("M1", "M3") else None,
                              extra_source=[Path(__file__)],
                              decision_interval=2 if condition == "M3" else 1,
                              score_refresh_period=args.score_refresh_period,
                              support=getattr(args, "support", "legacy_junyu_mask"),
                              diagnostic=False, timing_events=timing_events)


def _qualify(adapter: Any, row: dict, args: argparse.Namespace,
             directory: Path) -> dict:
    import torch
    from dllm.models import GenerationRequest

    plain_cfg = runner._config(_arm_args(args, "native_dense", timing_events=False,
                                         phase="observer_qualification"))
    event_cfg = runner._config(_arm_args(args, "native_dense", timing_events=True,
                                         phase="observer_qualification"))
    request = GenerationRequest(prompt=row["prompt"], max_new_tokens=8192,
                                temperature=0.0, seed=42, extra={"thinking": True})
    torch.cuda.synchronize()
    started = time.perf_counter()
    plain = adapter.generate(request)  # no CanvasCalls or CUDA event hooks
    torch.cuda.synchronize()
    plain_wall = time.perf_counter() - started
    plain_record = dict(schema="numerical_qk_plain_observer_control_v1",
                        fingerprint=plain_cfg["fingerprint"], id=row["id"], seed=42,
                        prompt_hash=hashlib.sha256(row["prompt"].encode()).hexdigest(),
                        completion_tokens=plain.completion_tokens,
                        raw_completion=adapter.tokenizer.decode(plain.completion_tokens,
                                                                 skip_special_tokens=False),
                        prediction=plain.text, metadata=plain.metadata,
                        termination_reason=plain.termination_reason,
                        request_wall_seconds=plain_wall,
                        reported_decoder_calls=plain.metadata.get("actual_denoising_step_count"))
    _immutable(directory / "plain.json", plain_record)

    no_events = runner._one(adapter, row, 42, plain_cfg)
    _immutable(directory / "observed_no_events.json", no_events)
    no_event_tokens_match = plain.completion_tokens == no_events["completion_tokens"]
    no_event_counts_match = (plain_record["reported_decoder_calls"] is not None and
                             plain_record["reported_decoder_calls"] == no_events["total_decoder_calls"])
    if not (no_event_tokens_match and no_event_counts_match):
        gate = dict(schema="numerical_qk_observer_gate_v1", id=row["id"], seed=42,
                    same_model_instance=True, tokens_match=no_event_tokens_match,
                    decoder_counts_match=no_event_counts_match, timeline_available=None,
                    passed=False, controls=["plain", "observed_no_events"],
                    reason="Observer without timing events changed native output or call count; event-on run was not attempted",
                    plain_fingerprint=plain_cfg["fingerprint"])
        _immutable(directory / "gate.json", gate)
        raise AssertionError("Observer qualification failed before CUDA timing events")
    with_events = runner._one(adapter, row, 42, event_cfg)
    _immutable(directory / "observed_with_events.json", with_events)

    tokens_match = (plain.completion_tokens == no_events["completion_tokens"] ==
                    with_events["completion_tokens"])
    counts_match = (plain_record["reported_decoder_calls"] is not None and
                    plain_record["reported_decoder_calls"] == no_events["total_decoder_calls"] ==
                    with_events["total_decoder_calls"] and
                    [c["decoder_calls"] for c in no_events["per_canvas"]] ==
                    [c["decoder_calls"] for c in with_events["per_canvas"]])
    timeline_available = with_events["generation_gpu_timeline_seconds"] is not None
    gate = dict(schema="numerical_qk_observer_gate_v1", id=row["id"], seed=42,
                same_model_instance=True, tokens_match=tokens_match,
                decoder_counts_match=counts_match, timeline_available=timeline_available,
                passed=bool(tokens_match and counts_match and timeline_available),
                controls=["plain", "observed_no_events", "observed_with_events"],
                plain_fingerprint=plain_cfg["fingerprint"],
                event_fingerprint=event_cfg["fingerprint"])
    _immutable(directory / "gate.json", gate)
    if not gate["passed"]:
        raise AssertionError("Observer qualification failed; no smoke arm may run")
    return gate


def _frozen_config(args: argparse.Namespace, condition: str,
                   timing_events: bool) -> dict:
    config = runner._config(_arm_args(args, condition, timing_events=timing_events))
    path = args.output / "configs" / f"smoke.{condition}.json"
    _immutable(path, config)
    return config


def _existing_receipt(path: Path, config: dict, row: dict) -> bool:
    if not path.exists():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (saved.get("fingerprint"), saved.get("id"), saved.get("seed"), saved.get("attempt")) != (
            config["fingerprint"], row["id"], 42, 0):
        raise RuntimeError(f"Incompatible immutable attempt-0 receipt: {path}")
    return True


def _dense_timing_path(output: Path, id_: str) -> Path:
    digest = hashlib.sha256(id_.encode()).hexdigest()[:20]
    return output / "timing_controls" / "native_dense" / f"{digest}.json"


def _dense_quality_source(root: Path, row: dict) -> tuple[Path, dict]:
    path = runner._receipt_path(root, "smoke", "native_dense", 42, row["id"])
    if not path.is_file():
        raise FileNotFoundError(f"Original dense attempt-0 quality receipt absent: {path}")
    source = json.loads(path.read_text(encoding="utf-8"))
    if (source.get("attempt"), source.get("condition"), source.get("id"), source.get("seed")) != (
            0, "native_dense", row["id"], 42):
        raise RuntimeError(f"Invalid original dense quality identity: {path}")
    if source.get("prompt_hash") != hashlib.sha256(row["prompt"].encode()).hexdigest():
        raise RuntimeError(f"Original dense quality prompt differs: {path}")
    return path, source


def _dense_timing_controls(adapter: Any, rows: list[dict], args: argparse.Namespace,
                           gate_dir: Path, session: str) -> None:
    event_cfg = runner._config(_arm_args(args, "native_dense", timing_events=True,
                                        phase="observer_qualification"))
    first_record_path = gate_dir / "observed_with_events.json"
    first_record = json.loads(first_record_path.read_text(encoding="utf-8"))
    if first_record.get("fingerprint") != event_cfg["fingerprint"]:
        raise RuntimeError("First event-on observer record has unexpected source fingerprint")
    for index, row in enumerate(rows):
        quality_path, original = _dense_quality_source(args.dense_quality_root, row)
        destination = _dense_timing_path(args.output, row["id"])
        original_sha = runner._sha(quality_path)
        if destination.exists():
            prior = json.loads(destination.read_text(encoding="utf-8"))
            if (prior.get("id"), prior.get("retry_source_fingerprint"),
                prior.get("original_quality_receipt_sha256"), prior.get("quality_eligible"),
                prior.get("timing_retry_only")) != (
                    row["id"], event_cfg["fingerprint"], original_sha, False, True):
                raise RuntimeError(f"Incompatible immutable dense timing control: {destination}")
            print(json.dumps(dict(event="reuse_dense_timing_control", id=row["id"],
                                  path=str(destination))), flush=True)
            continue
        retry = first_record if index == 0 else runner._one(adapter, row, 42, event_cfg)
        if retry.get("generation_gpu_timeline_seconds") is None:
            raise RuntimeError(f"Dense timing event boundary missing: {row['id']}")
        if (retry.get("condition"), retry.get("id"), retry.get("seed"),
            retry.get("fingerprint")) != (
                "native_dense", row["id"], 42, event_cfg["fingerprint"]):
            raise RuntimeError(f"Dense timing retry identity differs: {row['id']}")
        original_token_hash = runner._fingerprint(original["completion_tokens"])
        retry_token_hash = runner._fingerprint(retry["completion_tokens"])
        control = dict(schema="numerical_qk_dense_timing_control_v1", id=row["id"], seed=42,
                       condition="native_dense", quality_eligible=False,
                       timing_retry_only=True, independent_question_count_increment=0,
                       original_quality_receipt=str(quality_path.resolve()),
                       original_quality_receipt_sha256=original_sha,
                       original_quality_fingerprint=original["fingerprint"],
                       original_quality_token_hash=original_token_hash,
                       original_quality_decoder_calls=original["total_decoder_calls"],
                       retry_source_fingerprint=event_cfg["fingerprint"],
                       retry_source_observer_record=(str(first_record_path.resolve()) if index == 0 else None),
                       retry_session=session, retry_token_hash=retry_token_hash,
                       retry_decoder_calls=retry["total_decoder_calls"],
                       cross_load_token_match=original_token_hash == retry_token_hash,
                       cross_load_call_match=original["total_decoder_calls"] == retry["total_decoder_calls"],
                       cross_load_note="Comparison is descriptive across model loads; observer gate uses only three same-instance diagnostic runs",
                       retry=retry)
        runner._atomic(destination, control)
        print(json.dumps(dict(event="dense_timing_control", id=row["id"],
                              original_quality_unchanged=True,
                              cross_load_token_match=control["cross_load_token_match"],
                              cross_load_call_match=control["cross_load_call_match"],
                              path=str(destination))), flush=True)


def _validate(args: argparse.Namespace) -> list[dict]:
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    if args.ids != identity["selected_ids"]:
        raise ValueError("Smoke IDs/order must exactly match frozen four-ID identity")
    if not args.conditions or len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Conditions must be distinct and nonempty")
    if args.conditions != [name for name in ARM_ORDER if name in args.conditions]:
        raise ValueError("Conditions must be ordered fresh_junyu_T, M1, M3")
    if "M3" in args.conditions and "M1" not in args.conditions and args.conditions != ["M3"]:
        raise ValueError("M3 without M1 is allowed only as an explicit later M3-only run")
    rows = runner._selected([{key: value for key, value in row.items()
                              if key not in runner.GOLD_FIELDS}
                             for row in runner._rows(args.manifest)], args.ids, "smoke")
    return rows


def run(args: argparse.Namespace) -> None:
    import torch
    from dllm.models import create_adapter

    if not torch.cuda.is_available():
        raise RuntimeError("Native smoke batch requires CUDA")
    failure = args.output / "failure.json"
    if failure.exists():
        raise RuntimeError(f"Previous smoke failure requires review: {failure}")
    session = uuid.uuid4().hex
    current_condition = None
    current_id = None
    completed = 0
    try:
        rows = _validate(args)
        configs = {name: _frozen_config(args, name, bool(args.qualify_observer))
                   for name in args.conditions}
        pending = []
        for name in args.conditions:
            for row in rows:
                path = runner._receipt_path(args.output, "smoke", name, 42, row["id"])
                if not _existing_receipt(path, configs[name], row):
                    pending.append((name, row, path))
        if not pending and not args.qualify_observer:
            print(json.dumps(dict(event="all_attempt0_present", conditions=args.conditions)))
            return
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        adapter = create_adapter("diffusion_gemma", str(args.model.resolve()),
                                 device="cuda", precision="bfloat16",
                                 revision=args.revision).load()
        if args.qualify_observer:
            current_condition = "observer_qualification"
            current_id = rows[0]["id"]
            gate_dir = args.output / "observer_qualification" / session
            gate = _qualify(adapter, rows[0], args, gate_dir)
            print(json.dumps(dict(event="observer_qualified", session=session,
                                  gate=str(gate_dir / "gate.json"), passed=gate["passed"])), flush=True)
            if args.dense_timing_controls:
                current_condition = "native_dense_timing_controls"
                current_id = None
                _dense_timing_controls(adapter, rows, args, gate_dir, session)

        for name in args.conditions:
            current_condition = name
            if name == "M3" and "M1" in args.conditions:
                m1 = configs["M1"]
                if any(not _existing_receipt(runner._receipt_path(args.output, "smoke", "M1", 42, row["id"]),
                                             m1, row) for row in rows):
                    raise RuntimeError("M3 gate: all four M1 attempt-0 receipts must complete first")
            for row in rows:
                current_id = row["id"]
                destination = runner._receipt_path(args.output, "smoke", name, 42, row["id"])
                if _existing_receipt(destination, configs[name], row):
                    print(json.dumps(dict(event="reuse_attempt0", condition=name, id=row["id"],
                                          path=str(destination))), flush=True)
                    continue
                result = runner._one(adapter, row, 42, configs[name])
                runner._atomic(destination, result)
                completed += 1
                print(json.dumps(dict(event="attempt0", condition=name, id=row["id"],
                                      request_wall_seconds=result["request_wall_seconds"],
                                      decoder_calls=result["total_decoder_calls"],
                                      path=str(destination))), flush=True)
                if args.stop_after_requests is not None and completed >= args.stop_after_requests:
                    print(json.dumps(dict(event="checkpoint_stop", completed_new_requests=completed)), flush=True)
                    return
    except Exception as error:
        if not failure.exists():
            runner._atomic(failure, dict(schema="numerical_qk_smoke_failure_v1", session=session,
                                         condition=current_condition, id=current_id,
                                         completed_new_requests=completed,
                                         error_type=type(error).__name__, error=str(error)))
        raise


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--conditions", nargs="+", choices=ARM_ORDER, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--torch-library", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=runner.DEFAULT_POLICY)
    parser.add_argument("--policy-name", default="T_s50")
    parser.add_argument("--score-refresh-period", type=int, default=8)
    parser.add_argument("--support", choices=("legacy_junyu_mask", "native_mask"),
                        default="legacy_junyu_mask")
    parser.add_argument("--qualify-observer", action="store_true")
    parser.add_argument("--dense-timing-controls", action="store_true",
                        help="After observer parity, add four native-dense timing retries without changing quality")
    parser.add_argument("--dense-quality-root", type=Path,
                        help="Root of original event-off native-dense attempt-0 quality receipts")
    parser.add_argument("--stop-after-requests", type=int, choices=(1, 2))
    args = parser.parse_args(argv)
    if args.score_refresh_period <= 0:
        parser.error("Score refresh period must be positive")
    if args.dense_timing_controls and (not args.qualify_observer or args.dense_quality_root is None):
        parser.error("--dense-timing-controls requires --qualify-observer and --dense-quality-root")
    return args


if __name__ == "__main__":
    run(parse())
