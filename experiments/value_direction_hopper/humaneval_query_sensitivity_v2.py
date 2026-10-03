"""Stratified HumanEval audit for value-aware and BLASST routing.

This is a larger follow-up to the 18-question pilot.  It uses one uniform
local/global threshold pair per sparse method, calibrates on twelve disjoint
tasks, and evaluates one hundred seed-42 tasks sampled across six deterministic
function-type strata.  Calibration selects against whole, local, and global
physical sparsity together; every raw completion and trajectory remains
resumable and auditable.
"""
from __future__ import annotations

import argparse, ast, collections, fcntl, gzip, hashlib, json, os, random, re, shutil, subprocess, sys, tempfile, time
import numpy as np
from pathlib import Path
from typing import Any, Mapping

import torch

from . import aime_temporal_sweep as base
from . import aime_query_sensitivity_uniform as uniform
from .query_adaptive import UNIFORM_METHODS
from .experiment import atomic, fingerprint
from fast_dllm_v2.scripts.eval_blasst_mixed_tasks import _execute_humaneval

ROOT_DEFAULT = Path(__file__).resolve().parents[2] / "results" / "humaneval_query_sensitivity_v2"
DATA_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
METHODS = ("dense", "gaussian32", "blasst_aggressive", "C_gate", "T_prior")
SPARSE = METHODS[1:]
TARGET = 0.50
N_TOTAL, N_CAL = 112, 12

# Entry-point taxonomy rather than benchmark labels.  The strata are broad
# enough to cover the benchmark's dominant modes while remaining reproducible
# without a second model or hand-written task annotations.
STRATUM_RULES = (
    ("parsing_format", re.compile(r"parse|decode|encode|roman|binary|base|algebra|file|date|histogram|music|truncate|separate")),
    ("string_text", re.compile(r"string|substring|word|vowel|palindrome|character|letter|case|sentence|spaces|md5|paren|bracket|longest|prefix")),
    ("search_optimization", re.compile(r"search|path|closest|maximum|minimum|largest|smallest|find|arrange|solve|choose|triple|triangle|pile|fill|match|pluck|change")),
    ("collections_sequences", re.compile(r"list|array|sort|sequence|rolling|incr|remove|unique|common|filter|row|column|monotonic|count")),
    ("predicate_logic", re.compile(r"^is_|^has_|check|valid|below|equal|compare|will_it|can_|starts_|correct_|move_|bored|fly|nested|zero")),
)
# The smallest strata still contribute all available examples; the remaining
# 100 evaluation slots are allocated explicitly to avoid a random imbalance.
FINAL_COUNTS = {
    "parsing_format": 8,
    "string_text": 18,
    "search_optimization": 18,
    "collections_sequences": 18,
    "predicate_logic": 12,
    "numeric_logic": 26,
}
CAL_PER_STRATUM = 2


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _download(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fallback = Path(__file__).resolve().parents[2] / "results" / "humaneval_query_sensitivity_v1" / "source_data" / path.name
        if fallback.exists():
            shutil.copy2(fallback, path)
        else:
            import urllib.request
            with urllib.request.urlopen(DATA_URL, timeout=120) as r:
                path.write_bytes(r.read())
    return path


def _prompt(item: Mapping[str, Any]) -> str:
    return ("Complete the following Python function. Return only valid Python code "
            "containing the complete function, without Markdown fences.\n\n" +
            str(item["prompt"]))


def _stratum(item: Mapping[str, Any]) -> str:
    name = str(item["entry_point"]).lower()
    for label, pattern in STRATUM_RULES:
        if pattern.search(name):
            return label
    return "numeric_logic"


def _rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = _download(root / "source_data" / "HumanEval.jsonl.gz")
    with gzip.open(source, "rt", encoding="utf-8") as f:
        all_items = [json.loads(x) for x in f if x.strip()]
    grouped = collections.defaultdict(list)
    for idx, item in enumerate(all_items):
        grouped[_stratum(item)].append(idx)
    rng = random.Random(20261001)
    calibration_indices, evaluation_indices = [], []
    for stratum in FINAL_COUNTS:
        pool = list(grouped[stratum]); rng.shuffle(pool)
        need = CAL_PER_STRATUM + FINAL_COUNTS[stratum]
        if len(pool) < need:
            raise RuntimeError(f"stratum {stratum} has {len(pool)} items, need {need}")
        calibration_indices.extend(pool[:CAL_PER_STRATUM])
        evaluation_indices.extend(pool[CAL_PER_STRATUM:need])
    assert len(calibration_indices) == N_CAL and len(evaluation_indices) == 100
    rows = []
    for pos, idx in enumerate(calibration_indices + evaluation_indices):
        item = all_items[idx]
        rows.append(dict(
            id=f"humaneval/{item['task_id']}", source_id=int(idx), task="humaneval",
            benchmark="humaneval",
            stratum=_stratum(item), entry_point=item["entry_point"],
            prompt=_prompt(item), sparsity_prompt=item["prompt"],
            prompt_hash=hashlib.sha256(item["prompt"].encode()).hexdigest(),
            prompt_tokens=0, generation_budget=512, seed=42,
            label={"canonical_solution": item["canonical_solution"],
                   "test": item["test"], "entry_point": item["entry_point"],
                   "prompt": item["prompt"]},
        ))
    calibration, evaluation = rows[:N_CAL], rows[N_CAL:]
    old_manifest = Path(__file__).resolve().parents[2] / "results" / "humaneval_query_sensitivity_v1" / "manifest.json"
    old_ids = set()
    if old_manifest.exists():
        old = json.loads(old_manifest.read_text())
        old_ids = {x["id"] for x in old.get("calibration", []) + old.get("evaluation", [])}
    overlap = sorted(old_ids & {x["id"] for x in calibration + evaluation})
    return evaluation, calibration, dict(stratum_counts={k: sum(x["stratum"] == k for x in evaluation) for k in FINAL_COUNTS},
                                         calibration_stratum_counts={k: sum(x["stratum"] == k for x in calibration) for k in FINAL_COUNTS},
                                         overlap_with_v1=overlap)


def _score(row: Mapping[str, Any], text: str) -> float:
    result = _execute_humaneval(text, row["label"], timeout=5.0)
    return float(bool(result["passed"])), result


def _policy(local: float, global_: float) -> dict[str, Any]:
    return uniform._uniform_policy(local, global_)


def _initial(method: str) -> dict[str, Any]:
    # Anchors are the selected v1 HumanEval policies.  They only seed the
    # v2 complete-trajectory search; no v1 threshold is reused as a result.
    return {
        "blasst_aggressive": _policy(1.5, 1.45),
        "C_gate": _policy(.95, .55),
        "T_prior": _policy(.75, .35),
        "gaussian32": _policy(-.35, -.50),
    }[method]


def _run_one(adapter, row, method, policy, cfg, projections):
    if method == "dense":
        out = base._run_one(adapter, row, method, None, cfg, projections, diagnostics=True)
    else:
        out = base._run_one(adapter, row, method, policy, cfg, projections, diagnostics=True)
    # Outcomes remain unscored until every threshold has been frozen.
    out.update(stratum=row["stratum"], entry_point=row["entry_point"],
               generation_budget=row["generation_budget"])

    return out


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"accuracy": None, "sparsity": {k: 0.0 for k in ("whole", "local", "global")}}
    m = base._profile(rows)
    m["accuracy"] = sum(r["score"] for r in rows) / len(rows)
    statuses = collections.Counter(r.get("humaneval_execution", {}).get("status", "unknown").split(":")[0] for r in rows)
    by_stratum = {}
    for stratum in FINAL_COUNTS:
        group = [r for r in rows if r.get("stratum") == stratum]
        if group:
            by_stratum[stratum] = {
                "n": len(group), "accuracy": sum(r["score"] for r in group) / len(group),
                "mean_steps": sum(r["steps"] for r in group) / len(group),
                "status_counts": dict(collections.Counter(r.get("humaneval_execution", {}).get("status", "unknown") for r in group)),
            }
    m["execution_status_counts"] = dict(statuses)
    m["mean_prediction_chars"] = sum(len(r.get("prediction", "")) for r in rows) / len(rows)
    m["mean_prediction_chars_pass"] = (sum(len(r.get("prediction", "")) for r in rows if r["score"]) /
                                        max(1, sum(bool(r["score"]) for r in rows)))
    m["mean_prediction_chars_fail"] = (sum(len(r.get("prediction", "")) for r in rows if not r["score"]) /
                                        max(1, sum(not bool(r["score"]) for r in rows)))
    m["by_stratum"] = by_stratum
    return m


def _cache(root, stage, method, key, row, fn):
    path = root / stage / method / key / (hashlib.sha256(row["id"].encode()).hexdigest() + ".json")
    identity = fingerprint([json.loads((root / "configuration.json").read_text())["fingerprint"],
                            method, key, row])
    if path.exists():
        value = json.loads(path.read_text())
        if value.get("identity") != identity or value.get("status") != "complete":
            raise RuntimeError(f"Cached shard identity mismatch: {path}")
        return value
    atomic(root / "progress.json", dict(stage=stage, method=method, key=key, id=row["id"], time=time.time()))
    value = fn(); value["identity"] = identity
    atomic(path, value)
    return value


def _fit(root, adapter, cfg, rows, projections, method, initial, stage, max_points=32):
    """Complete-trajectory fitting; only tile counts determine the threshold."""
    destination = root / "thresholds" / f"{stage}_{method}_s50.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        if old["fingerprint"] != cfg["fingerprint"]:
            raise RuntimeError("Threshold provenance mismatch")
        if old.get("status") == "attained":
            return old
    points = []
    def evaluate(pair):
        pair = [round(float(x), 6) for x in pair]
        for point in points:
            if point["pair"] == pair:
                return point
        policy = _policy(*pair); key = fingerprint(policy)[:16]
        values = [_cache(root, stage, method, key, row,
                        lambda row=row: _run_one(adapter, row, method, policy, cfg, projections))
                  for row in rows]
        metrics = _metrics(values)
        errors = {k: abs(metrics["sparsity"][k] - TARGET) for k in ("whole", "local", "global")}
        point = dict(pair=pair, policy=policy, metrics=metrics, errors=errors,
                     max_error=max(errors.values()), key=key)
        points.append(point)
        atomic(root / "calibration_traces" / f"{stage}_{method}.json", points)
        print(json.dumps(dict(event="threshold_point", stage=stage, method=method,
                              number=len(points), pair=pair, sparsity=metrics["sparsity"],
                              max_error=point["max_error"])), flush=True)
        return point
    def objective(point):
        return (point["max_error"], sum(point["errors"].values()), point["pair"])
    pair = [initial["late"][k]["log_threshold"] for k in ("local", "global")]
    current = evaluate(pair)
    # Aim within 1pp; accept at most 2pp, including actual final-cohort budget.
    slope = .40 if method != "blasst_aggressive" else .17
    jac = np.eye(2) * slope
    radius = .30 if method != "blasst_aggressive" else .55
    failures = 0
    while len(points) < max_points and current["max_error"] > .01:
        x = np.array(current["pair"])
        y = np.array([current["metrics"]["sparsity"][k] for k in ("local", "global")])
        try: delta = np.linalg.solve(jac, TARGET-y)
        except np.linalg.LinAlgError: delta = (TARGET-y)/slope
        delta = np.clip(delta, -radius, radius)
        if np.max(np.abs(delta)) < .001: break
        trial = evaluate(x+delta)
        dx = np.array(trial["pair"])-x
        dy = np.array([trial["metrics"]["sparsity"][k] for k in ("local", "global")])-y
        if dx@dx > 1e-12:
            jac = jac + np.outer(dy-jac@dx, dx)/(dx@dx)
        best = min(points, key=objective)
        if objective(best) < objective(current):
            current = best; failures = 0
        else:
            failures += 1; radius *= .65
        if failures >= 2:
            # Refresh the Jacobian around the best measured point; coupled
            # trajectories can violate the simple monotone diagonal model.
            x = np.array(current["pair"])
            y = np.array([current["metrics"]["sparsity"][k] for k in ("local", "global")])
            h = max(.015, min(.08, radius))
            for axis in range(2):
                if len(points) >= max_points: break
                offset = np.zeros(2); offset[axis] = h
                trial = evaluate(x+offset)
                jac[:,axis] = (np.array([trial["metrics"]["sparsity"][k] for k in ("local", "global")])-y)/h
            current = min(points, key=objective); failures = 0
            if radius < .006: break
    selected = min(points, key=objective)
    result = dict(method=method, stage=stage, target=.5, tolerance=.02,
                  fingerprint=cfg["fingerprint"], policy=selected["policy"],
                  selected_metrics=selected["metrics"], max_error=selected["max_error"],
                  status="attained" if selected["max_error"] <= .02 else "search_incomplete",
                  calibration_ids=[r["id"] for r in rows], candidates=points,
                  threshold_schedule="uniform_all_denoising_steps",
                  accuracy_used_for_selection=False)
    atomic(destination, result)
    if result["status"] != "attained":
        raise RuntimeError(f"{stage}/{method}: search incomplete, best error {selected['max_error']:.4f}")
    return result


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    # Preserve the previously disclosed sample, rather than changing it in
    # response to outcomes. The original preparation contains no generations.
    old_path = Path("/home/exouser/humaneval_query_sensitivity_v2/manifest.json")
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        if old_path.exists():
            atomic(manifest_path, json.loads(old_path.read_text()))
            _download(root / "source_data" / "HumanEval.jsonl.gz")
        else:
            evaluation, calibration, audit = _rows(root)
            atomic(manifest_path, dict(seed=42, total=N_TOTAL, calibration=calibration,
                                      evaluation=evaluation, audit=audit))
    manifest = json.loads(manifest_path.read_text())
    evaluation, calibration = manifest["evaluation"], manifest["calibration"]
    assert len(evaluation) == 100 and len(calibration) == 12
    assert not {r['id'] for r in evaluation} & {r['id'] for r in calibration}
    (root / "PROTOCOL.md").write_text("""# HumanEval v2 protocol

- Seed 42 only; dense, Gaussian32, BLASST, C_gate, T_prior.
- Preserved 12/100 calibration/evaluation manifest prepared October 1.
- Six heuristic function-name strata; these are not official benchmark categories.
- One local/global log-threshold pair for every denoising call; native stopping, acceptance, and temperature schedule unchanged.
- First fit on 12 disjoint calibration prompts using measured whole/local/global tile counts.
- Then audit the actual 100-prompt cohort. If sparsity drifts outside 50 +/- 2 percentage points, adjust the same two thresholds using tile counts alone on that cohort. No correctness or syntax outcome enters selection.
- Such final-cohort budget matching is transductive, not wholly held-out threshold calibration. Preserve the disjoint policies and initial audit point for comparison.
- Freeze all thresholds before scoring. No per-prompt, per-phase, or online threshold changes.
- All raw trajectories, hashes, candidates and completions retained. Instrumented wall times are not clean performance claims.
""")
    return evaluation, calibration, manifest.get("audit", {})


def configuration(root):
    import transformers
    cfg = json.loads(base.RULER_CFG.read_text())
    cfg.update(beta=3., gamma=.5, trajectory_gamma=.5, gate_trajectory_gamma=.65,
               gate_tau=2.5, rank=32, projection_seed=1729, physical_tile=[128,64], canvas=256, max_steps=48)
    paths = [Path(__file__), Path(base.__file__), Path(uniform.__file__),
             Path(__file__).with_name('query_adaptive.py'), Path(__file__).with_name('query_sensitivity_uniform.py'),
             Path(__file__).with_name('integration.py'), Path(__file__).with_name('cuda.py'),
             Path('src/dllm/models/adapters/diffusion_gemma.py'),
             Path('experiments/diffusion_gemma_solattn_blasst_multibench/runner.py'),
             Path('fast_dllm_v2/scripts/eval_blasst_mixed_tasks.py'),
             Path(cfg['library']), Path(cfg['torch_library']), root/'manifest.json']
    cfg.update(source_hashes={str(p.resolve()): _sha(p) for p in paths},
               torch=torch.__version__, transformers=transformers.__version__, seed=42,
               score_blind_budget_fit=True)
    cfg.pop('fingerprint', None); cfg['fingerprint'] = fingerprint(cfg)
    path=root/'configuration.json'
    if path.exists() and json.loads(path.read_text()) != cfg:
        raise RuntimeError('Source/configuration changed: use a new root')
    if not path.exists(): atomic(path,cfg)
    return cfg


def run(root):
    evaluation, calibration, audit = prepare(root)
    lock = (root/'worker.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    cfg=configuration(root)
    # Calibration and budget fitting never execute or use answer tests.
    base.aime_score=lambda row,text: 0.
    adapter=base._adapter(cfg); projections=base.Projections()
    disjoint={m:_fit(root,adapter,cfg,calibration,projections,m,_initial(m),'calibration') for m in SPARSE}
    atomic(root/'calibration_complete.json', dict(passed=True, thresholds=disjoint))
    print(json.dumps(dict(event='calibration_complete', policies={m:d['policy'] for m,d in disjoint.items()})),flush=True)
    thresholds={}
    for method in SPARSE:
        thresholds[method]=_fit(root,adapter,cfg,evaluation,projections,method,disjoint[method]['policy'],'budget_match',max_points=20)
        atomic(root/'thresholds'/f'{method}_s50.json',thresholds[method])
    atomic(root/'thresholds_frozen.json',dict(passed=True,thresholds=thresholds,accuracy_used=False))
    all_rows={}
    for method in METHODS:
        policy=None if method=='dense' else thresholds[method]['policy']
        key='dense' if policy is None else fingerprint(policy)[:16]
        stage='evaluation' if policy is None else 'budget_match'
        values=[]
        for row in evaluation:
            result=dict(_cache(root,stage,method,key,row,lambda row=row:_run_one(adapter,row,method,policy,cfg,projections)))
            result['score'],result['humaneval_execution']=_score(row,result['prediction'])
            values.append(result)
        all_rows[method]=values
        atomic(root/f'{method}_evaluation.json',values)
        print(json.dumps(dict(event='method_complete',method=method,metrics=_metrics(values))),flush=True)
    report=dict(protocol=dict(seed=42,calibration=12,evaluation=100,target=.5,tolerance=.02,
                             threshold_fit='disjoint first, then final-cohort tile-count-only budget matching',strata=audit),
                methods={m:_metrics(v) for m,v in all_rows.items()},thresholds=thresholds,
                fingerprint=cfg['fingerprint'])
    atomic(root/'report.json',report)
    lines=['# HumanEval v2', '', 'Seed 42, 100 fixed stratified prompts; thresholds selected using tile counts only.',
           'Final-cohort budget matching is transductive; disjoint calibration is archived separately.', '',
           '| Method | Pass@1 | Overall | Local | Global | Calls/canvas | Syntax failures | Capped canvases |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,r in report['methods'].items():
        sp=r['sparsity']; errors=r['execution_status_counts'].get('syntax_error',0)
        lines.append(f"| {m} | {r['accuracy']:.1%} | {sp['whole']:.2%} | {sp['local']:.2%} | {sp['global']:.2%} | {r['mean_canvas_steps']:.2f} | {errors} | {r['cap_canvases']} |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    atomic(root/'final_complete.json',dict(passed=True,n=100,methods=METHODS,fingerprint=cfg['fingerprint']))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=("prepare", "run")); ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    a = ap.parse_args()
    if a.stage == "prepare": prepare(a.root)
    else: run(a.root)


if __name__ == "__main__": main()
