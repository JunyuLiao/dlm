"""Score private quick-suite completions and report sanitized accuracy/trajectory aggregates."""
from __future__ import annotations
import argparse, collections, json, math, re, subprocess, tempfile, os, statistics
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference/LongBench/LongBench"))
from metrics import qa_f1_score as official_qa_f1

def code_extract(s):
    m=re.search(r"```(?:python|py)?\s*(.*?)```",s or "",re.S|re.I); return (m.group(1) if m else s).strip()
def last_num(s):
    vals=re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?",s or "")
    return vals[-1] if vals else None
def norm(s): return re.sub(r"[^a-z0-9]","",str(s).lower())
def f1(pred, gold):
    return float(official_qa_f1(pred, gold))
def extract_answer(s):
    m=re.findall(r"\\boxed\{([^{}]*)\}",s or "")
    if m:return m[-1]
    return last_num(s)
def finish_text(s): return op.load_final_response()[0](s, True)
def run_mbpp(code, g):
    code=code_extract(code); tests=g.get("test_list",[])+g.get("challenge_test_list",[])
    src=code+"\n"+g.get("test_setup_code","")+"\n"+"\n".join(tests)
    try:
        p=subprocess.run([sys.executable,"-c",src],capture_output=True,text=True,timeout=5)
        return p.returncode==0
    except Exception:return False

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--pool",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    final,_=op.load_final_response(); rows={}
    for ds in [p.name.replace("_rowinfo.json","") for p in a.pool.glob("*_rowinfo.json")]:
        x=json.loads((a.pool/(ds+"_rowinfo.json")).read_text()); rows[ds]={r["id"]:r["gold"] for r in x["rows"]}
        manifest={r["id"]:r for r in json.loads((a.pool/(ds+"_generation_manifest.json")).read_text())}
        for rid,g in rows[ds].items():
            if rid in manifest: g.setdefault("prompt", manifest[rid].get("prompt", ""))
    allarms=collections.defaultdict(list)
    for p in a.root.glob("**/private/*.private.jsonl"):
        if not (p.parent.parent / "attempt_complete.json").exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                r=json.loads(line); r["method"] = p.parts[-4] if len(p.parts) >= 4 else r["arm"]; allarms[r["method"]].append(r)
    output={"schema":"v31_quick_accuracy_summary_v1","arms":{}}
    for arm,recs in sorted(allarms.items()):
        byds=collections.defaultdict(list); traj=[]
        for r in recs:
            ds=r["dataset"]; gold=rows.get(ds,{}).get(r["id"],{})
            text=final(r.get("completion",""),True); ok=False; score=0.0
            if ds=="humaneval":
                try:
                    import v27_humaneval as he
                    ok=he.run_test(he.program_for(gold,text if text else r.get("completion","")))[0]
                except Exception: ok=False
                score=float(ok)
            elif ds=="mbpp": ok=run_mbpp(text,gold); score=float(ok)
            elif ds=="gsm8k_4shot":
                pred=last_num(text); target=last_num(gold.get("answer","")); ok=pred is not None and target is not None and abs(float(pred)-float(target))<1e-6; score=float(ok)
            elif ds=="math500":
                pred=extract_answer(text); ok=pred is not None and norm(pred)==norm(gold.get("answer","")); score=float(ok)
            elif ds.startswith("longbench_"):
                pred=text; score=max((f1(pred,x) for x in gold.get("answers",[])),default=0.0); ok=score>=0.5
            byds[ds].append(score); traj.append({k:r.get(k) for k in ("denoise_forwards","canvases","wall_s","output_tokens")})
        dsout={d:{"n":len(v),"mean":100*statistics.mean(v),"correct":sum(x>=0.999999 for x in v)} for d,v in sorted(byds.items())}
        long=[x["mean"] for d,x in dsout.items() if d.startswith("longbench_")]
        dsout["longbench_average"]={"n":sum(x["n"] for d,x in dsout.items() if d.startswith("longbench_")),"mean":statistics.mean(long) if long else None}
        nums={k:[x[k] for x in traj if isinstance(x.get(k),(int,float))] for k in ("denoise_forwards","canvases","wall_s","output_tokens")}
        output["arms"][arm]={"cells":len(recs),"datasets":dsout,"trajectory":{"denoising_steps_per_problem":statistics.mean(nums["denoise_forwards"]) if nums["denoise_forwards"] else None,"canvases_per_problem":statistics.mean(nums["canvases"]) if nums["canvases"] else None,"wall_s_mean":statistics.mean(nums["wall_s"]) if nums["wall_s"] else None,"output_tokens_mean":statistics.mean(nums["output_tokens"]) if nums["output_tokens"] else None}}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(output,indent=2)+"\n"); print(json.dumps({"arms":sorted(output["arms"]),"out":str(a.out)}))
if __name__=="__main__": main()
