"""Pure-CPU selection/schedule invariants; no model or private gold required."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts import v20_panel as panel
from scripts import v20_run


def row(id_, **meta):
    prompt = "request " + id_
    return dict(id=id_, prompt=prompt, prompt_hash=panel.sha(prompt),
                prompt_tokens=[1, 2, 3], prompt_token_count=3, answer="SECRET", **meta)


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = {d: self.root / (d + ".json") for d in ("ruler4k", "aime26", "longbench_v2")}
        self.data = {
            "ruler4k": [row(f"ruler/{t}/{i}", task=f"task{t:02d}", generation_budget=30)
                         for t in range(13) for i in range(10)],
            "aime26": [row(f"aime26/{i}", generation_budget=8192) for i in range(1, 31)],
            "longbench_v2": [],
        }
        # Rebuild the LB rows without duplicate keyword arguments.
        self.data["longbench_v2"] = []
        for i in range(12):
            x = row(f"longbench_v2/{i}", domain=f"domain{i % 4}",
                    bin="10-15K" if i % 2 == 0 else "15-20K", generation_budget=8192)
            n = 10000 + (i % 2) * 5000
            x["prompt_tokens"] = [1] * n
            x["prompt_token_count"] = n
            self.data["longbench_v2"].append(x)
        for dataset, path in self.paths.items():
            path.write_text(json.dumps(self.data[dataset]))
        self.identity = dict(model_revision=panel.REVISION, scope="ALL_NATIVE_LEGAL", policy_sha256="a" * 64,
                             datasets={d: dict(source_manifest_sha256=panel.sha(p.read_bytes()),
                                               gold_sha256="b" * 64, scorer_sha256="c" * 64,
                                               task_contract_sha256="d" * 64) for d, p in self.paths.items()})
        self.identity_file = self.root / "identity.json"
        self.identity_file.write_text(json.dumps(self.identity))
        self.hosts_file = self.root / "hosts.json"
        self.hosts_file.write_text(json.dumps([dict(host="h1", gpu_uuid="GPU-1"),
                                               dict(host="h2", gpu_uuid="GPU-2")]))

    def freeze(self):
        return panel.freeze(self.paths["ruler4k"], self.paths["aime26"], self.paths["longbench_v2"],
                            self.identity_file, self.hosts_file, self.root / "private", self.root / "out")

    def test_frozen_counts_pairing_and_gold_firewall(self):
        p = self.freeze()
        self.assertEqual({d: len(ids) for d, ids in p["ids"].items()},
                         {"ruler4k": 13, "aime26": 6, "longbench_v2": 6})
        self.assertEqual(p["ids"]["aime26"], list(panel.AIME_IDS))
        self.assertEqual(len(p["schedule"]), 700)
        self.assertEqual(len(p["historical_extension"]["schedule"]), 100)
        self.assertEqual({e["block"] for e in p["schedule"][:84]}, set(range(6)))
        self.assertFalse(any(e["arm"] == panel.HISTORICAL for e in p["schedule"]))
        for block in range(50):
            events = [e for e in p["schedule"] if e["block"] == block]
            self.assertEqual(len(events), 14)
            self.assertEqual([e["arm"] for e in events[7:]], [e["arm"] for e in events[:7]][::-1])
            self.assertEqual(len({(e["host"], e["gpu_uuid"], e["id"], e["seed"]) for e in events}), 1)
            self.assertEqual(p["block_assignments"][str(block)]["scope"], "ALL_NATIVE_LEGAL")
        self.assertEqual(Counter(e["dataset"] for e in p["schedule"][:84]),
                         {"ruler4k": 28, "aime26": 28, "longbench_v2": 28})
        for path in (self.root / "out" / "frozen_protocol.json", *(self.root / "private").glob("*.json")):
            self.assertNotIn("SECRET", path.read_text())
            self.assertNotIn(b"\r\n", path.read_bytes())
        for dataset, digest in p["generation_manifest_sha256"].items():
            self.assertEqual(panel.sha((self.root / "private" / f"{dataset}_generation_manifest.json").read_bytes()), digest)
        self.assertEqual(p["ids"]["ruler4k"], self.freeze()["ids"]["ruler4k"])

    def test_source_drift_fails_closed(self):
        self.data["aime26"][0]["prompt"] = "changed"
        self.paths["aime26"].write_text(json.dumps(self.data["aime26"]))
        with self.assertRaisesRegex(ValueError, "source manifest SHA"):
            self.freeze()

    def test_prompt_and_task_contract_rejected(self):
        self.data["ruler4k"][0]["prompt_hash"] = "f" * 64
        self.paths["ruler4k"].write_text(json.dumps(self.data["ruler4k"]))
        self.identity["datasets"]["ruler4k"]["source_manifest_sha256"] = panel.sha(self.paths["ruler4k"].read_bytes())
        self.identity_file.write_text(json.dumps(self.identity))
        with self.assertRaisesRegex(ValueError, "prompt identity"):
            self.freeze()

    def test_host_and_policy_identity_required(self):
        self.identity["scope"] = None
        self.identity_file.write_text(json.dumps(self.identity))
        with self.assertRaisesRegex(ValueError, "scope"):
            self.freeze()

    def test_worker_stage_partition_and_unbound_method_gate(self):
        p = self.freeze()
        host, uuid = "h1", "GPU-1"
        initial = v20_run.stage_entries(p, host, uuid, "initial")
        remainder = v20_run.stage_entries(p, host, uuid, "remainder")
        historical = v20_run.stage_entries(p, host, uuid, "historical")
        self.assertEqual((len(initial), len(remainder), len(historical)), (42, 308, 50))
        self.assertEqual(len(initial + remainder), 350)
        protocol_path = self.root / "out" / "frozen_protocol.json"
        binding_path = self.root / "binding.json"
        binding_path.write_text(json.dumps({"status": "pending", "panel_protocol_sha256": panel.sha(protocol_path.read_bytes())}))
        with self.assertRaisesRegex(ValueError, "method binding not frozen"):
            v20_run.validate_inputs(protocol_path, binding_path, self.root / "private", host, uuid, stage="initial")

    def test_router_phase_evidence_and_warm_identity(self):
        arm = "M3_R2_A8_current_output"
        counters = dict(attention_calls=16, score_refresh_calls=2,
                        decision_refresh_calls=8, held_decision_calls=8)
        phase = v20_run.router_phase_evidence(arm, counters)
        self.assertEqual((phase["A"], phase["D"], phase["H"]), (2, 6, 8))
        self.assertIsNone(v20_run.router_phase_evidence(arm, dict(counters, held_decision_calls=7)))
        base = dict(ok=True, completion_token_hash="h", per_canvas_calls=[4], termination="eos",
                    phase_evidence=dict(phase="M1_M3_decoder", fresh_decoder_calls=4,
                                        per_canvas=[dict(decoder_calls=4)], initial_prefill_end_observed=True),
                    router_phase_evidence=phase, triton_misses=0, triton_disk_entries_added=0,
                    new_shared_objects=[])
        self.assertTrue(v20_run.strict_v20_warm(base, dict(base))["accepted"])
        changed = dict(base, router_phase_evidence=dict(phase, H=7))
        self.assertIn("router_phase_evidence_mismatch", v20_run.strict_v20_warm(base, changed)["reasons"])


if __name__ == "__main__":
    unittest.main()
