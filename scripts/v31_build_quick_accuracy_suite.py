"""Build private, deterministic quick-evaluation manifests for auxiliary suites."""
from __future__ import annotations
import argparse, gzip, hashlib, json, random, shutil, zipfile
from pathlib import Path

SEED = 20261011
ROOT = Path(__file__).resolve().parents[1]

def sha(x): return hashlib.sha256(x.encode()).hexdigest()
def load_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def sample_sorted(rows, key, n):
    rows = sorted(rows, key=lambda x: str(x.get(key, "")))
    rng = random.Random(SEED + n)
    return sorted(rng.sample(rows, min(n, len(rows))), key=lambda x: str(x.get(key, "")))
def parse_answers(x):
    if isinstance(x, list): return x
    try: return json.loads(x.replace("'", '"'))
    except Exception: return [x]

def gsm_prompt(examples, question):
    s = ""
    for e in examples:
        s += "Question: " + e["question"] + "\nLet's think step by step\nAnswer: " + e["answer"] + "\n\n"
    return s + "Question: " + question + "\nLet's think step by step\nAnswer:"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--tokenizer", type=Path, required=True); ap.add_argument("--sample", type=int, default=20); ap.add_argument("--longbench-sample", type=int, default=None)
    for name in ('humaneval', 'mbpp', 'gsm8k', 'math500', 'longbench-zip'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args(); out = a.out.resolve(); raw = out / "raw"; out.mkdir(parents=True, exist_ok=True); raw.mkdir(exist_ok=True)
    longbench_sample = a.longbench_sample if a.longbench_sample is not None else a.sample
    from dllm.models import create_adapter
    adapter = create_adapter("diffusion_gemma", str(a.tokenizer), device="cpu", precision="float32").load_tokenizer()
    all_cells=[]; build_sources={}
    def write_dataset(name, rows, gold, budgets, max_len):
        manifest=[]; rowinfo=[]
        for i,(prompt,g) in enumerate(zip(rows,gold)):
            toks = adapter.encode_prompt(prompt, {"thinking": True})
            rec={"id":f"{name}_{i:04d}","source_id":g.get("source_id",f"{name}_{i:04d}"),"index":i,"benchmark":name,"task":name,"thinking":True,"prompt":prompt,"prompt_hash":sha(prompt),"prompt_tokens":toks,"prompt_token_count":len(toks),"generation_budget":budgets}
            manifest.append(rec); rowinfo.append({"id":rec["id"],"index":i,"prompt_sha256":sha(json.dumps(toks,separators=(",",":"))),"gold":g})
            all_cells.append({"dataset":name,"id":rec["id"],"index":i,"prompt_tokens":toks,"seed":1})
        (out/(name+"_generation_manifest.json")).write_text(json.dumps(manifest,separators=(",",":")))
        (out/(name+"_rowinfo.json")).write_text(json.dumps({"schema":"v31_rowinfo_v2","manifest_sha256":sha(json.dumps(manifest,separators=(",",":"))),"max_model_len":max_len,"rows":rowinfo},separators=(",",":")))
        (out/(name+"_gold.json")).write_text(json.dumps(rowinfo,separators=(",",":")))
    # HumanEval
    he=load_jsonl(a.humaneval); he=sample_sorted(he,"task_id",a.sample)
    hp="Complete the Python function below. Return the complete Python solution in a single fenced python code block, without a test harness.\n\n"
    write_dataset("humaneval",[hp+x["prompt"] for x in he],[{"source_id":x["task_id"],"task_id":x["task_id"],"test":x["test"],"entry_point":x["entry_point"]} for x in he],8192,16384)
    # MBPP
    mb=load_jsonl(a.mbpp); mb=sample_sorted(mb,"task_id",a.sample)
    write_dataset("mbpp",["Solve this Python programming problem. Return only a complete Python solution in a fenced python code block.\n\n"+x["text"] for x in mb],[{"source_id":str(x["task_id"]),"task_id":x["task_id"],"text":x["text"],"test_setup_code":x.get("test_setup_code",""),"test_list":x.get("test_list",[]),"challenge_test_list":x.get("challenge_test_list",[])} for x in mb],4096,16384)
    # GSM8K, fixed four examples from the repository yaml
    gs=load_jsonl(a.gsm8k); gs=sample_sorted(gs,"id",a.sample)
    yaml=Path("third_party/dinfer/evaluations/tasks/gsm8k/gsm8k-llada-moe.yaml").read_text()
    import re
    qs=re.findall(r"question:\s*\|\n((?:\s{4}.*\n)+)\s*answer:\s*\|\n((?:\s{4}.*\n)+)",yaml)
    examples=[]
    for q,ans in qs[:4]: examples.append({"question":"\n".join(z[4:] for z in q.rstrip().splitlines()),"answer":"\n".join(z[4:] for z in ans.rstrip().splitlines())})
    if len(examples)<4: examples=[{"question":"What is 2+2?","answer":"4"}]*4
    write_dataset("gsm8k_4shot",[gsm_prompt(examples,x["prompt"]) for x in gs],[{"source_id":str(x["id"]),"id":x["id"],"answer":x["answer"]} for x in gs],4096,16384)
    # MATH500
    ma=load_jsonl(a.math500); ma=sample_sorted(ma,"unique_id",a.sample)
    write_dataset("math500",["Solve the following math problem. Show your reasoning and put the final answer in \\boxed{}.\n\n"+x["problem"] for x in ma],[{"source_id":x["unique_id"],"unique_id":x["unique_id"],"answer":x["answer"]} for x in ma],4096,16384)
    # LongBench v1 requested tasks
    with zipfile.ZipFile(a.longbench_zip) as z:
        for task,file in [("hotpotqa","data/hotpotqa.jsonl"),("triviaqa","data/triviaqa.jsonl"),("narrativeqa","data/narrativeqa.jsonl"),("qasper","data/qasper.jsonl"),("multifieldqa","data/multifieldqa_en.jsonl")]:
            rows=[json.loads(x) for x in z.read(file).decode().splitlines() if x.strip()]; rows=sample_sorted(rows,"_id",longbench_sample); name="longbench_"+task
            write_dataset(name,[x["input"] for x in rows],[{"source_id":x.get("_id"),"answers":parse_answers(x.get("answers",[])),"metric_dataset":task,"all_classes":x.get("all_classes")} for x in rows],4096,32768)
    (out/"cells_all.json").write_text(json.dumps(all_cells,separators=(",",":")))
    short=[x for x in all_cells if x["dataset"] in {"humaneval","mbpp","gsm8k_4shot","math500"}]; long=[x for x in all_cells if x["dataset"].startswith("longbench_")]
    (out/"cells_short.json").write_text(json.dumps(short,separators=(",",":"))); (out/"cells_longbench.json").write_text(json.dumps(long,separators=(",",":")))
    for p in [a.humaneval, a.mbpp, a.gsm8k, a.math500, a.longbench_zip]:
        build_sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/"build.json").write_text(json.dumps({"schema":"v31_quick_accuracy_pool_v1","seed":SEED,"sample_per_task":a.sample,"longbench_sample_per_task":longbench_sample,"source_sha256":build_sources,"datasets":sorted({x["dataset"] for x in all_cells})},indent=2)+"\n")
    print(json.dumps({"out":str(out),"cells":len(all_cells),"short":len(short),"longbench":len(long)}))
if __name__ == "__main__": main()
