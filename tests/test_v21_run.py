import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.v13_seed_runs import execution_key
from scripts.v18_protocol import sha
from scripts.v21_run import (stage_entries, strict_v21_warm, validate_arm_config,
                             validate_protocol, validate_resume_events, validate_inputs,
                             read_task_rows, per_canvas_stopping, V20_PROTOCOL)


def panel():
    arms = ["M3_parent", "M3_layout"]
    contracts = {a: dict(kind="v21_method", parent_v20_arm="M3_R2_A8_current_output",
                         scope="ALL_NATIVE_LEGAL", output_score_precision="legacy_bf16_scores",
                         output_layout="head_major" if a == "M3_parent" else "model_major") for a in arms}
    ids = {d: [f"{d}/{i}" for i in range(4)] for d in ("ruler4k", "aime26", "longbench_v2")}
    assignments, schedule = {}, []
    for b in range(24):
        dataset = list(ids)[b // 8]
        qid, seed = ids[dataset][(b % 8) // 2], [101, 202][b % 2]
        host = ["hostA", "hostB"][b % 2]
        uuid = ["gpuA", "gpuB"][b % 2]
        assignments[str(b)] = dict(dataset=dataset, id=qid, seed=seed, host=host, gpu_uuid=uuid)
        order = arms[b % 2:] + arms[:b % 2]
        for role, ordered in (("attempt0", order), ("warm", order[::-1])):
            for arm in ordered:
                schedule.append(dict(index=len(schedule), block=b, dataset=dataset, id=qid, seed=seed,
                                     arm=arm, role=role, repeat=0 if role == "attempt0" else 1,
                                     host=host, gpu_uuid=uuid, cell_id=f"cell-{b}-{arm}"))
    return dict(schema="v21_conditional_panel_v1", status="frozen", execution_ready=True,
                panel_kind="layout_pair", planned_executions=96, protocol_id="v21-test", model_revision="rev",
                arms=arms, arm_contracts=contracts, ids=ids, seeds=[101, 202], schedule=schedule,
                block_assignments=assignments, stages={"first": list(range(4)), "remainder": list(range(4, 24))},
                generation_manifest_sha256={d: "x" * 64 for d in ids})


def numeric_panel():
    p = panel()
    arms = ["D_native", "D_matched", "T_scope", "B_A8_matched", "M1_R1_A8_current_output",
            "M3_R2_A8_current_output", "M3_R3_A8_current_output"]
    p["panel_kind"], p["planned_executions"], p["arms"] = "numeric7", 448, arms
    p["arm_contracts"] = {a: (dict(kind="native") if a == "D_native" else
                              dict(kind="v20_legacy", parent_v20_arm="T_scope", scope="ALL_NATIVE_LEGAL")
                              if a == "T_scope" else
                              dict(kind="v21_control" if a == "D_matched" else "v21_method",
                                   parent_v20_arm=a, scope="ALL_NATIVE_LEGAL",
                                   output_score_precision="fp32_scores_bf16_pv", output_layout="head_major"))
                           for a in arms}
    p["ids"] = {"ruler4k": [f"ruler4k/{i}" for i in range(6)],
                "aime26": [f"aime26/{i}" for i in range(5)],
                "longbench_v2": [f"longbench_v2/{i}" for i in range(5)]}
    p["schedule"], p["block_assignments"] = [], {}
    combos = [(d, q, s) for d, questions in p["ids"].items() for q in questions for s in (101, 202)]
    for b, (dataset, qid, seed) in enumerate(combos):
        host, uuid = ["hostA", "hostB"][b % 2], ["gpuA", "gpuB"][b % 2]
        p["block_assignments"][str(b)] = dict(dataset=dataset, id=qid, seed=seed, host=host, gpu_uuid=uuid)
        order = arms[b % 7:] + arms[:b % 7]
        for role, ordered in (("attempt0", order), ("warm", order[::-1])):
            for arm in ordered:
                p["schedule"].append(dict(index=len(p["schedule"]), block=b, dataset=dataset, id=qid,
                                          seed=seed, arm=arm, role=role, repeat=0 if role == "attempt0" else 1,
                                          host=host, gpu_uuid=uuid, cell_id=f"numeric-{b}-{arm}"))
    p["stages"] = {"first": list(range(6)), "remainder": list(range(6, 32))}
    return p


def diagnostic_panel():
    import json
    old = json.loads(V20_PROTOCOL.read_text())
    ids = old["ids"]["longbench_v2"][:2]
    arms = ["D_native", "D_matched_legacy", "D_matched_new_numeric"]
    contracts = {
        "D_native": dict(kind="native"),
        "D_matched_legacy": dict(kind="v20_legacy", parent_v20_arm="D_matched", scope="ALL_NATIVE_LEGAL"),
        "D_matched_new_numeric": dict(kind="v21_control", parent_v20_arm="D_matched",
                                      scope="ALL_NATIVE_LEGAL", output_score_precision="fp32_scores_bf16_pv",
                                      output_layout="head_major"),
    }
    schedule, assignments = [], {}
    for b, (qid, seed) in enumerate((q, s) for q in ids for s in (101, 202)):
        host, uuid = ["hostA", "hostB"][b % 2], ["gpuA", "gpuB"][b % 2]
        assignments[str(b)] = dict(dataset="longbench_v2", id=qid, seed=seed, host=host, gpu_uuid=uuid)
        for arm in arms[b % 3:] + arms[:b % 3]:
            schedule.append(dict(index=len(schedule), block=b, dataset="longbench_v2", id=qid, seed=seed,
                                 arm=arm, role="attempt0", repeat=0, host=host, gpu_uuid=uuid,
                                 cell_id=f"diagnostic-{b}-{arm}"))
    return dict(schema="v21_conditional_panel_v1", status="frozen", execution_ready=True,
                panel_kind="zero_pruning_diagnostic", planned_executions=12, protocol_id="v21-diagnostic-test",
                model_revision="rev", arms=arms, arm_contracts=contracts, ids={"longbench_v2": ids},
                seeds=[101, 202], schedule=schedule, block_assignments=assignments,
                stages={"all_diagnostic": list(range(4))},
                generation_manifest_sha256={"longbench_v2": "x" * 64},
                v20_protocol_sha256=sha(V20_PROTOCOL.read_bytes()), quality_eligible=False, timing_eligible=False)


class V21RunTests(unittest.TestCase):
    def test_protocol_accepts_complete_same_host_first_reverse_warm(self):
        p = panel()
        validate_protocol(p)
        self.assertEqual(len(stage_entries(p, "hostA", "gpuA", "first")), 8)
        self.assertEqual(len(stage_entries(p, "hostB", "gpuB", "all")), 48)

    def test_numeric_seven_arm_448_schedule(self):
        p = numeric_panel()
        validate_protocol(p)
        self.assertEqual(len(stage_entries(p, "hostA", "gpuA", "all")), 224)

    def test_first_only_cp1_diagnostic_excludes_scoring_and_warm_timing(self):
        p = diagnostic_panel()
        validate_protocol(p)
        self.assertEqual(len(stage_entries(p, "hostA", "gpuA", "all")), 6)
        p["quality_eligible"] = True
        with self.assertRaisesRegex(ValueError, "forbid panel scoring"):
            validate_protocol(p)
        p = diagnostic_panel()
        p["schedule"][0]["role"] = "warm"
        with self.assertRaisesRegex(ValueError, "first/warm role"):
            validate_protocol(p)

    def test_layout_pair_empty_tasks_require_canonical_manifest_and_no_configs(self):
        p = panel()
        p["ids"] = {"ruler4k": [], "aime26": [],
                    "longbench_v2": [f"longbench_v2/{j}" for j in range(12)]}
        for block in range(24):
            qid, seed = p["ids"]["longbench_v2"][block // 2], [101, 202][block % 2]
            assignment = p["block_assignments"][str(block)]
            assignment.update(dataset="longbench_v2", id=qid, seed=seed)
            for row in p["schedule"][block * 4:(block + 1) * 4]:
                row.update(dataset="longbench_v2", id=qid, seed=seed)
        validate_protocol(p)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = root / "manifests"
            manifests.mkdir()
            for dataset in ("ruler4k", "aime26"):
                path = manifests / f"{dataset}_generation_manifest.json"
                path.write_bytes(b"[]\n")
                p["generation_manifest_sha256"][dataset] = sha(path.read_bytes())
                self.assertEqual(read_task_rows(path, dataset, [], p["generation_manifest_sha256"][dataset]), [])
            rows = [dict(id=q, prompt="question", prompt_hash=sha("question"), prompt_tokens=[1, 2],
                         prompt_token_count=2, thinking=True, generation_budget=8192)
                    for q in p["ids"]["longbench_v2"]]
            (manifests / "longbench_v2_generation_manifest.json").write_text(json.dumps(rows) + "\n")
            p["generation_manifest_sha256"]["longbench_v2"] = sha(
                (manifests / "longbench_v2_generation_manifest.json").read_bytes())
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps(p))
            model = root / "model"
            model.mkdir()
            config = root / "config.json"
            config.write_text("{}")
            binding = root / "binding.json"
            binding.write_text(json.dumps(dict(schema="v21_conditional_binding_v1", status="frozen",
                                               panel_protocol_sha256=sha(protocol.read_bytes()),
                                               policy_sha256="a" * 64,
                                               host_models={"hostA": str(model)},
                                               host_configs={"hostA": {"longbench_v2": {
                                                   arm: {"path": str(config), "sha256": sha(config.read_bytes())}
                                                   for arm in p["arms"]}}})))
            with patch("scripts.v21_run.validate_arm_config") as check:
                _, _, loaded, configs = validate_inputs(protocol, binding, manifests, "hostA", "gpuA", stage="all")
            self.assertEqual(len(loaded), 12)
            self.assertEqual(configs["ruler4k"], {})
            self.assertEqual(configs["aime26"], {})
            self.assertEqual(check.call_count, 2)
            empty = manifests / "ruler4k_generation_manifest.json"
            empty.write_bytes(b"[]")
            with patch("scripts.v21_run.validate_arm_config"):
                with self.assertRaisesRegex(ValueError, "byte drift"):
                    validate_inputs(protocol, binding, manifests, "hostA", "gpuA", stage="all")
            with self.assertRaisesRegex(ValueError, "canonical"):
                read_task_rows(empty, "ruler4k", [], sha(empty.read_bytes()))

    def test_protocol_rejects_cross_gpu_and_partial_block(self):
        p = panel()
        p["schedule"][1]["host"] = "hostB"
        with self.assertRaisesRegex(ValueError, "assignment"):
            validate_protocol(p)
        p = panel()
        p["schedule"][3]["arm"] = p["schedule"][2]["arm"]
        with self.assertRaisesRegex(ValueError, "cell identity|reverse warm"):
            validate_protocol(p)

    def test_uncertain_and_partial_resume_refuse_retry(self):
        p = panel()
        entries = stage_entries(p, "hostA", "gpuA", "all")
        identity = dict(protocol_id="v21-test", host="hostA", gpu_uuid="gpuA")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(validate_resume_events([], identity, entries, root), set())
            with self.assertRaisesRegex(ValueError, "terminal"):
                validate_resume_events([dict(event="start", **identity)], identity, entries, root)
            partial = [dict(event="start", **identity),
                       dict(event="run", **entries[0], generation_seed=entries[0]["seed"],
                            execution_key=execution_key(entries[0])),
                       dict(event="worker_end", host="hostA", gpu_uuid="gpuA")]
            with self.assertRaisesRegex(ValueError, "partial"):
                validate_resume_events(partial, identity, entries, root)
            complete = [dict(event="start", **identity)] + [
                dict(event="run", **e, generation_seed=e["seed"], execution_key=execution_key(e))
                for e in entries[:4]] + [dict(event="worker_end", host="hostA", gpu_uuid="gpuA")]
            self.assertEqual(len(validate_resume_events(complete, identity, entries, root)), 4)
            (root / "cells" / entries[4]["cell_id"]).mkdir(parents=True)
            (root / "cells" / entries[4]["cell_id"] / "attempt00.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "orphan"):
                validate_resume_events(complete, identity, entries, root)

    def test_nested_config_identity_without_flattening(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("model")
            source = root / "source.py"
            source.write_text("source")
            parent = {"model": str(model), "manifest_sha256": "m" * 64, "policy_sha256": "p" * 64,
                      "v20_scope": "ALL_NATIVE_LEGAL", "v20_arm": "M3_R2_A8_current_output",
                      "source_hashes": {str(source): sha(source.read_bytes())},
                      "model_metadata_hashes": {"config.json": sha((model / "config.json").read_bytes())}}
            config = {"parent_kind": "v20_method", "parent_config": parent, "condition": "M3",
                      "output_score_precision": "legacy_bf16_scores", "output_layout": "model_major"}
            contract = panel()["arm_contracts"]["M3_layout"]
            import experiments.numerical_qk_reuse as package
            stub = SimpleNamespace(validate_effective=lambda *_: ("ALL_NATIVE_LEGAL", 2))
            with patch.object(package, "v21", stub, create=True):
                validate_arm_config(config, contract, model=str(model), manifest_sha="m" * 64,
                                    policy_sha="p" * 64)
                changed = dict(config, parent_config=dict(parent, manifest_sha256="other"))
                with self.assertRaisesRegex(ValueError, "nested parent"):
                    validate_arm_config(changed, contract, model=str(model), manifest_sha="m" * 64,
                                        policy_sha="p" * 64)

    def test_strict_warm_checks_router_phase_and_native_trajectory(self):
        phase = {"phase": "v21_current_output_decoder", "fresh_decoder_calls": 3,
                 "per_canvas": [{"decoder_calls": 3}], "initial_prefill_end_observed": True}
        first = dict(ok=True, completion_token_hash="t", per_canvas_calls=[3], termination="eos",
                     phase_evidence=phase, router_phase_evidence={"A": 1, "D": 1, "H": 1})
        warm = dict(first, triton_misses=0, triton_disk_entries_added=0, new_shared_objects=[])
        self.assertTrue(strict_v21_warm(first, warm)["accepted"])
        warm = dict(warm, router_phase_evidence={"A": 1, "D": 2, "H": 0})
        self.assertIn("router_phase_evidence_mismatch", strict_v21_warm(first, warm)["reasons"])

    def test_actual_wrapper_shape_satisfies_one_request_fields_without_gpu(self):
        from experiments.numerical_qk_reuse.runner import request_budget, request_thinking
        source = V20_PROTOCOL.parents[2] / "experiments/numerical_qk_reuse/v21.py"
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "_wrap")
        namespace = {"Path": Path, "__file__": str(source), "hashlib": hashlib,
                     "SOURCES": (), "PLUGIN": "v21-test", "REQUEST_ENVELOPE":
                         ("phase", "diagnostic", "timing_events", "thinking", "max_new_tokens"),
                     "_fingerprint": lambda value: hashlib.sha256(json.dumps(value, sort_keys=True,
                                                      separators=(",", ":")).encode()).hexdigest()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        for thinking, budget in ((True, 8192), (False, 128)):
            parent = dict(condition="v20_dense_consumer", phase="v21_test", diagnostic=False,
                          timing_events=False, thinking=thinking, max_new_tokens=budget)
            config = namespace["_wrap"](parent, "v20_control", "legacy_bf16_scores", "head_major")
            row = dict(thinking=thinking, generation_budget=budget)
            self.assertEqual(request_budget(row, config), budget)
            self.assertIs(request_thinking(row, config), thinking)
            self.assertEqual((config["phase"], config["diagnostic"], config["timing_events"]),
                             ("v21_test", False, False))

    def test_native_stopping_flags_do_not_depend_on_timing_events(self):
        receipt = {"per_canvas": [{"native_stop_final_call": True,
                                    "iteration_cap_final_call": False},
                                   {"native_stop_final_call": False,
                                    "iteration_cap_final_call": True}]}
        self.assertEqual(per_canvas_stopping(receipt), [
            {"native_stop": True, "iteration_cap": False},
            {"native_stop": False, "iteration_cap": True}])
        receipt["per_canvas"][0].pop("native_stop_final_call")
        self.assertIsNone(per_canvas_stopping(receipt))


if __name__ == "__main__":
    unittest.main()
