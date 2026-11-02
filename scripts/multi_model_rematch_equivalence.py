"""Audit the actual AIME modules against original modules, without test prompts."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

from experiments.numerical_qk_reuse import multi_model_attention as current


REVISIONS = {"llada22": "224f8f11173bd16e5cf10054a963c74233758cf2",
             "trado": "8c0d14b031636ddcc708df75fa1d57aefa8b0ba4"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def methods(path):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelHarness")
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def dump(node):
    return ast.dump(node, include_attributes=False)


def common_generate(node):
    """Exclude only the unreachable Ultra branch in these two model comparisons."""
    node = copy.deepcopy(node)
    model_if = next(n for n in node.body if isinstance(n, ast.If) and
                    ast.unparse(n.test) == "self.key == 'llada22'")
    trado_if = model_if.orelse[0]
    assert isinstance(trado_if, ast.If) and ast.unparse(trado_if.test) == "self.key == 'trado'"
    trado_if.orelse = [ast.Pass()]
    return dump(node)


def source_checks(old, new):
    before, after = methods(old), methods(new)
    unchanged = sorted(set(before) - {"__init__", "_install_ultra", "generate"})
    for name in unchanged:
        assert dump(before[name]) == dump(after[name]), (name, "model runtime differs")
    assert common_generate(before["generate"]) == common_generate(after["generate"])
    # Preserve every old constructor assignment except the runtime construction;
    # that object's observable numerical behavior is checked directly on CUDA.
    def statements(node):
        return [dump(n) for n in node.body if not (isinstance(n, ast.Assign) and
                any(ast.unparse(t) == "self.runtime" for t in n.targets))]
    old_nodes, new_nodes = statements(before["__init__"]), statements(after["__init__"])
    for stmt in old_nodes:
        assert stmt in new_nodes, "old constructor state changed"
    return {"unchanged_model_methods": unchanged,
            "llada_trado_generation_ast_equal": True,
            "original_constructor_state_preserved": True,
            "excluded_unreachable_model": "ultrallada"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("rerun", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    result = {"status": "fail", "synthetic_only": True, "benchmark_requests": 0,
              "audit_after_generation": True, "source_sha256": sha(__file__), "models": {}}
    try:
        for model, revision in REVISIONS.items():
            old_dir = a.base / "private/execution" / revision / "experiments/numerical_qk_reuse"
            new_dir = Path(current.__file__).parent
            old_config = json.loads((a.base / f"runs/{model}/aime26/value_v2_gaussian32_reuse_r4/shard001/attempt001/config.json").read_text())
            new_config = json.loads((a.rerun / f"runs/{model}/aime26/value_v2_gaussian32_reuse_r4/shard001/attempt001/config.json").read_text())
            for field, filename in (("attention_source_sha256", "multi_model_attention.py"),
                                    ("runtime_sha256", "multi_model_runtime.py")):
                assert old_config[field] == sha(old_dir / filename)
                assert new_config[field] == sha(new_dir / filename)
            dependencies = {}
            for name in ("v27_fa4.py", "v31_value_kernels.py", "v31_value_summary.py", "v31_value_selectors.py"):
                assert sha(old_dir / name) == sha(new_dir / name), name
                dependencies[name] = sha(new_dir / name)
            checks = source_checks(old_dir / "multi_model_runtime.py", new_dir / "multi_model_runtime.py")
            spec = importlib.util.spec_from_file_location(
                "experiments.numerical_qk_reuse.rematch_reference_" + model, old_dir / "multi_model_attention.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cases = []
            torch.manual_seed(6795)
            for nq, nk, h, hk, block in ((4, 197, 8, 2, 4), (32, 513, 4, 1, 32),
                                         (129, 4097, 4, 4, 0)):
                for arm in current.ARMS:
                    for threshold in ((.012489663701836472, .025109133251762627)
                                      if "value_v2" in arm else (.02,)):
                        old, new = module.AttentionRuntime(arm, threshold), current.AttentionRuntime(arm, threshold)
                        for runtime in (old, new):
                            runtime.block(nk - min(nq, 32), nk)
                        for step in range(6):
                            q = torch.randn((1, h, nq, 128), device="cuda", dtype=torch.bfloat16)
                            k = torch.randn((1, hk, nk, 128), device="cuda", dtype=torch.bfloat16)
                            v = torch.randn_like(k)
                            if step == 2:
                                old.invalidate()
                                new.invalidate()
                            for runtime in (old, new):
                                runtime.begin_forward()
                            ref = old.attention(q, k, v, 2, 128 ** -.5, offset=nk-nq, block_size=block)
                            out = new.attention(q, k, v, 2, 128 ** -.5, offset=nk-nq, block_size=block)
                            assert torch.equal(ref, out), (model, arm, nq, nk, step, "output")
                            assert old.counts == new.counts, (model, arm, step, "counts")
                            if "reuse" in arm:
                                assert torch.equal(old.maps[(2, nq, nk)][0], new.maps[(2, nq, nk)][0])
                            cases.append({"arm": arm, "threshold": threshold, "q": nq, "kv": nk,
                                          "step": step, "output_bitwise_equal": True,
                                          "counters_equal": True, "maps_equal": True if "reuse" in arm else None})
            result["models"][model] = {**checks, "dependencies": dependencies,
                  "original_runtime_sha256": old_config["runtime_sha256"],
                  "actual_runtime_sha256": new_config["runtime_sha256"],
                  "original_attention_sha256": old_config["attention_source_sha256"],
                  "actual_attention_sha256": new_config["attention_source_sha256"],
                  "statistics_query_blocks": 1, "cases": cases}
        torch.cuda.synchronize()
        result["status"] = "pass"
    finally:
        result["gpu_worker_seconds"] = time.time() - started
        (a.out / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": sum(len(r["cases"]) for r in result["models"].values()),
                      "gpu_worker_seconds": result["gpu_worker_seconds"]}))


if __name__ == "__main__":
    main()
