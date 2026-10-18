"""Independent factorial arithmetic, correlation, and attribution guards."""
import copy
import hashlib
import json

import pytest

from scripts.v31_refresh_diagnostic_report import REQUIRED_LABELS, run, run_headroom, summarize_trials


def trials(indices=(0, 1), seeds=(1, 42), engines=(1, 2)):
    rows = []
    # y = 100 + 4*sparse + 10*fresh + 6*sparse*fresh.
    values = dict(dense_fresh=110, dense_hold=100, sparse_fresh=120, sparse_hold=104,
                  control_before=100, control_after=102)
    for index in indices:
        for seed in seeds:
            for engine in engines:
                for label in REQUIRED_LABELS:
                    rows.append(dict(suite="longbench_v2", index=index, panel_seed=seed,
                                     engine=engine, label=label, score=1,
                                     denoise_forwards=values[label], canvases=10,
                                     output_tokens=200, wall_s=2., decode_s=1.,
                                     finish_reason="stop", output_hash="same",
                                     diagnostic_receipt=dict(exposed=True, prefix_signature=f"state-{index}-{seed}-{engine}",
                                         complete_scope=True, errors=[], added_model_forwards=0, added_sampler_calls=0,
                                         scope=[5, 11, 17, 23, 29])))
    return rows


def test_factorial_simple_main_and_interaction_arithmetic():
    report = summarize_trials(trials(), bootstrap_samples=64)
    contrasts = report["suites"]["longbench_v2"]["contrasts"]
    expected = dict(sparse_minus_dense=7, fresh_minus_hold=13, interaction=6,
                    sparse_minus_dense_fresh=10, sparse_minus_dense_hold=4,
                    fresh_minus_hold_dense=10, fresh_minus_hold_sparse=16,
                    control_after_minus_before=2, dense_fresh_minus_control_mean=9)
    for name, value in expected.items():
        assert contrasts[name]["means"]["denoise_forwards"] == value
        assert contrasts[name]["bootstrap"]["ci95"]["denoise_forwards"] == [value, value]
    assert report["pairing_gate"]["status"] == "passed"
    assert report["coverage"]["complete_groups"] == 8


def test_cluster_bootstrap_keeps_all_seeds_and_engines_together():
    rows = trials()
    for row in rows:
        if row["label"] in ("sparse_fresh", "sparse_hold"):
            # Opposite seed effects cancel within every question, and thus
            # within every legal bootstrap replicate. Row resampling fails.
            row["denoise_forwards"] += 50 if row["panel_seed"] == 1 else -50
    effect = summarize_trials(rows, bootstrap_samples=101)["suites"]["longbench_v2"]["contrasts"]["sparse_minus_dense"]
    assert effect["bootstrap"]["ci95"]["denoise_forwards"] == [7, 7]
    assert effect["by_seed"]["1"]["means"]["denoise_forwards"] == 57
    assert effect["by_seed"]["42"]["means"]["denoise_forwards"] == -43


def test_single_question_does_not_get_fake_precision_from_repeats():
    effect = summarize_trials(trials(indices=(9,)), bootstrap_samples=100)["suites"]["longbench_v2"]["contrasts"]["interaction"]
    assert effect["question_clusters"] == 1
    assert effect["bootstrap"]["samples"] == 0
    assert effect["bootstrap"]["ci95"]["score"] == [None, None]


def test_prefix_mismatch_does_not_remove_unfavorable_outcome():
    rows = trials()
    row = next(row for row in rows if row["label"] == "sparse_fresh")
    row["diagnostic_receipt"]["prefix_signature"] = "different"
    row["score"] = 0
    report = summarize_trials(rows, bootstrap_samples=10)
    assert report["pairing_gate"]["status"] == "failed"
    result = report["suites"]["longbench_v2"]
    assert result["paired_groups"] == 8
    assert result["contrasts"]["sparse_minus_dense_fresh"]["means"]["score"] == -.125
    assert result["contrasts"]["sparse_minus_dense_fresh"]["prefix_matched_subset_descriptive_only"]["paired_groups"] == 7


def test_exposure_and_missing_receipts_are_explicit_not_false_success():
    rows = trials()
    for row in rows:
        if row["index"] == 1:
            row["diagnostic_receipt"]["exposed"] = False
    del rows[0]["diagnostic_receipt"]["prefix_signature"]
    report = summarize_trials(rows, bootstrap_samples=10)
    assert report["pairing_gate"]["status"] == "unverified"
    assert report["exposure_counts"] == {"exposed": 4, "not_exposed": 4}
    effect = report["suites"]["longbench_v2"]["contrasts"]["interaction"]
    assert effect["exposure_subsets"]["not_exposed"]["paired_groups"] == 4


def test_missing_expected_cells_and_labels_remain_in_coverage():
    rows = trials(indices=(0,), seeds=(1,), engines=(1,))
    rows.pop()
    expected = [dict(suite="longbench_v2", index=index, panel_seed=1, engine=1) for index in (0, 1)]
    report = summarize_trials(rows, expected_cells=expected, bootstrap_samples=10)
    assert report["status"] == "partial_coverage"
    assert report["coverage"]["complete_groups"] == 0
    assert len(report["coverage"]["missing"]) == 2
    assert report["suites"]["longbench_v2"]["arms"]["dense_fresh"]["trials"] == 1


def test_duplicates_and_nonfinite_fields_rejected():
    rows = trials()
    with pytest.raises(ValueError, match="duplicate"):
        summarize_trials(rows + [rows[0]], bootstrap_samples=1)
    rows[0]["wall_s"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        summarize_trials(rows, bootstrap_samples=1)


def test_tail_outcome_and_aa_noise_not_trimmed():
    rows = trials()
    for row in rows:
        if row["index"] == 1 and row["label"] == "sparse_fresh":
            row["denoise_forwards"] += 1000
            row["finish_reason"] = "length"
        if row["label"] == "control_after":
            row["output_hash"] = "changed"
    result = summarize_trials(rows, bootstrap_samples=20)["suites"]["longbench_v2"]
    effect = result["contrasts"]["sparse_minus_dense_fresh"]
    assert effect["means"]["denoise_forwards"] == 510
    assert effect["tails"]["denoise_forwards"]["largest_absolute_question_contributions"][0]["index"] == 1
    assert effect["leave_one_question_out"][1]["means"]["denoise_forwards"] == 10
    assert result["arms"]["sparse_fresh"]["finish_reasons"]["length"] == 4
    assert result["control_repeatability"]["output_hash_disagreements"] == 8
    assert result["control_repeatability"]["noise_absolute"]["denoise_forwards"]["max_absolute_delta"] == 2


def test_suite_units_and_headroom_are_not_pooled():
    rows = trials(indices=(0,), seeds=(1,), engines=(1,))
    second = copy.deepcopy(rows)
    for row in second:
        row["suite"] = "ruler128k"
        row["score"] = 100
    headroom = dict(rows[0], label="fixed16")
    report = summarize_trials(rows + second + [headroom], bootstrap_samples=3)
    assert report["suites"]["ruler128k"]["arms"]["dense_fresh"]["means"]["score"] == 100
    assert report["other_arms"]["longbench_v2"]["fixed16"]["trials"] == 1
    assert report["coverage"]["complete_groups"] == 2


def test_control_prefix_is_separate_from_four_arm_causal_gate():
    rows = trials()
    for row in rows:
        if row["label"].startswith("control_"):
            row["diagnostic_receipt"]["prefix_signature"] = None
    report = summarize_trials(rows, bootstrap_samples=3)
    assert report["pairing_gate"]["status"] == "passed"
    assert report["suites"]["longbench_v2"]["control_repeatability"]["control_prefix_counts"] == {"unverified": 8}


def test_bad_intervention_scope_blocks_integrity_without_discarding_outcome():
    rows = trials()
    rows[0]["diagnostic_receipt"]["scope"] = [29]
    report = summarize_trials(rows, bootstrap_samples=3)
    assert report["intervention_integrity"]["status"] == "failed"
    assert report["coverage"]["complete_groups"] == 8


def test_nonfactorial_phases_are_explicitly_excluded():
    rows = trials()
    headroom = dict(rows[0], phase="headroom", label="fixed16")
    qualification = dict(rows[0], phase="qualification", engine=99)
    report = summarize_trials(rows + [headroom, qualification], bootstrap_samples=3)
    assert report["coverage"]["excluded_phase_rows"] == {"headroom": 1, "qualification": 1}
    assert report["coverage"]["complete_groups"] == 8


def test_plan_loader_uses_only_factorial_and_frozen_expected_cells(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    rows = trials(indices=(0,), seeds=(1,), engines=(1,))
    for folder, data in (("factorial", rows), ("qualification", [dict(rows[0], engine=99)])):
        (study / folder).mkdir()
        (study / folder / "scored.json").write_text(json.dumps(data))
    expected = [dict(suite="longbench_v2", index=0, panel_seed=1, engine=engine) for engine in (1, 2)]
    (study / "protocol.json").write_text(json.dumps(dict(expected_cells=expected,
        plans=[dict(phase="factorial", folder="factorial"), dict(phase="qualification", folder="qualification")])) )
    report = run(study, tmp_path / "report", bootstrap_samples=3)
    assert report["status"] == "partial_coverage"
    assert report["coverage"]["total_rows"] == 6
    assert report["coverage"]["expected_groups"] == 2
    assert "qualification/scored.json" not in report["input_sha256"]


def test_immutable_output_bindings_and_private_payload_exclusion(tmp_path):
    source = tmp_path / "scored.json"
    rows = trials()
    rows[0]["prompt"] = "PRIVATE_PROMPT_SENTINEL"
    rows[0]["diagnostic_receipt"]["raw_completion"] = "PRIVATE_RECEIPT_SENTINEL"
    source.write_text(json.dumps(rows))
    output = tmp_path / "report"
    report = run(None, output, scored_paths=[source], bootstrap_samples=10)
    text = (output / "summary.json").read_text()
    assert "PRIVATE_PROMPT_SENTINEL" not in text
    assert "PRIVATE_RECEIPT_SENTINEL" not in text
    assert "scored001:scored.json" in report["input_sha256"]
    assert str(tmp_path) not in text
    assert (output / "analysis_complete.json").exists()
    assert not (output / "final_complete.json").exists()
    with pytest.raises(ValueError, match="immutable"):
        run(None, output, scored_paths=[source], bootstrap_samples=10)


def _referenced_headroom_study(tmp_path, *, old_row_source="old-source"):
    parent, study = tmp_path / "attempt001", tmp_path / "attempt002"
    parent.mkdir()
    study.mkdir()
    old_plan = dict(engine=2, phase="headroom", suite="longbench_v2", family="shared",
                    pass_name="clean", repeat=0, folder="headroom/clean", measured_count=1,
                    request_count=3, cells_sha256="cells", arms_sha256="arms")
    def write(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    old_protocol_sha = write(parent / "protocol.json", dict(source_commit="old-source", plans=[old_plan]))
    base = dict(trials(indices=(0,), seeds=(42,), engines=(2,))[0], phase="headroom", family="shared",
                pass_name="clean", sample_repeat=0, label="dynamic", native_receipt={},
                cuda_graph_captures=0, value_clean_timing=True,
                source_commit=old_row_source, protocol_sha256=old_protocol_sha)
    scored_sha = write(parent / old_plan["folder"] / "scored.json", [base])
    marker_sha = write(parent / old_plan["folder"] / "attempt_complete.json",
                       dict(status="complete", source_commit="old-source", protocol_sha256=old_protocol_sha,
                            files_sha256={"scored.json": scored_sha}))
    reference = dict(attempt=parent.name, folder=old_plan["folder"], protocol_sha256=old_protocol_sha,
                     source_commit="old-source", marker_sha256=marker_sha)
    linked = dict(old_plan, source_reference=reference)
    profile = dict(old_plan, engine=6, pass_name="profile", folder="headroom/profile")
    new_protocol_sha = write(study / "protocol.json", dict(source_commit="new-source", plans=[linked, profile]))
    profile_row = dict(base, engine=6, pass_name="profile", source_commit="new-source",
                       protocol_sha256=new_protocol_sha, value_clean_timing=False,
                       native_receipt={"headroom": {"complete_scope_inventory": True, "spans": {
                           "denoise/global_refresh": {"cuda_inclusive_span_ms": 100},
                           "denoise/discovery": {"cuda_inclusive_span_ms": 80}}}})
    new_scored_sha = write(study / profile["folder"] / "scored.json", [profile_row])
    write(study / profile["folder"] / "attempt_complete.json",
          dict(status="complete", source_commit="new-source", protocol_sha256=new_protocol_sha,
               files_sha256={"scored.json": new_scored_sha}))
    return study, parent


def test_headroom_successor_references_original_source_without_relabeling(tmp_path):
    study, parent = _referenced_headroom_study(tmp_path)
    original = {str(p): p.read_bytes() for p in parent.rglob("*.json")}
    result = run_headroom(study, tmp_path / "report")
    assert result["source_commits"] == ["new-source", "old-source"]
    assert result["mixed_execution_sources"] is True
    assert result["engine_coverage"][0]["source_reference"]["attempt"] == "attempt001"
    assert result["engine_coverage"][0]["source_commit"] == "old-source"
    assert result["engine_coverage"][1]["source_commit"] == "new-source"
    assert result["clean_arms"]["dynamic"]["timing_verification"] == {"passed": 1}
    assert result["audit_arms"]["dynamic"]["refresh_time_proxy"][0]["refresh_span_over_audit_request"] == .05
    assert result["critical_path_headroom"]["request_latency_gain_bound"] is None
    assert "references/attempt001/protocol.json" in result["input_sha256"]
    assert "old-source" in (tmp_path / "report" / "summary.md").read_text()
    assert original == {str(p): p.read_bytes() for p in parent.rglob("*.json")}


@pytest.mark.parametrize("changed", ["protocol.json", "headroom/clean/attempt_complete.json", "headroom/clean/scored.json"])
def test_headroom_successor_rejects_changed_reference_bytes(tmp_path, changed):
    study, parent = _referenced_headroom_study(tmp_path)
    path = parent / changed
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        run_headroom(study, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_headroom_successor_rejects_wrong_row_source_even_with_valid_hashes(tmp_path):
    study, _ = _referenced_headroom_study(tmp_path, old_row_source="wrong-source")
    with pytest.raises(ValueError, match="row source identity"):
        run_headroom(study, tmp_path / "report")
