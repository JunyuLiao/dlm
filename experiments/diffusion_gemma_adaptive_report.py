"""Regenerate dense-paired reports for native adaptive bundles."""
import argparse, csv, gzip, json
from pathlib import Path


def records(d):
    src = d.get("records_source")
    if src:
        return json.loads(gzip.decompress(Path(src["path"]).read_bytes()))
    return d.get("records", [])


def one(d, dense):
    rs = records(d); e=s=ge=gs=le=ls=mass=rows=0
    for r in rs:
        e += float(r.get("eligible", 0)); s += float(r.get("skipped", 0))
        mass += float(r.get("mass_sum", 0)); rows += float(r.get("rows", 0))
        if r.get("attention_type") == "global": ge += float(r.get("eligible", 0)); gs += float(r.get("skipped", 0))
        if r.get("attention_type") == "local": le += float(r.get("eligible", 0)); ls += float(r.get("skipped", 0))
    a=d.get("completion_tokens") or []; b=dense.get("completion_tokens") or []
    n=max(len(a), len(b)); agree=sum(x == y for x, y in zip(a, b))/max(1, n)
    return dict(id=d["id"], benchmark=d.get("benchmark", d["id"].split("/",1)[0]),
        score=float(d.get("score") or 0), dense_score=float(dense.get("score") or 0),
        delta_pp=100*(float(d.get("score") or 0)-float(dense.get("score") or 0)),
        eligible=e, skipped=s, actual_sparsity=s/max(1,e), global_eligible=ge,
        global_skipped=gs, global_sparsity=gs/max(1,ge), local_eligible=le,
        local_skipped=ls, local_sparsity=ls/max(1,le), retained_mass=mass/max(1,rows),
        token_agreement=agree, sequence_exact=int(a == b), compared_tokens=n)


def run(root, source):
    root=Path(root); source=Path(source)
    dense={}
    # Predecessor bundles use the same hashed shard paths and prompts.
    for f in (source/"dense/dense/shards").glob("*.json"):
        d=json.loads(f.read_text()); dense[d["id"]]=d
    all_rows=[]
    for cond in sorted((root/"final").glob("adaptive_analytic_s*/shards")):
        label=cond.parent.name; out=[]
        for f in cond.glob("*.json"):
            d=json.loads(f.read_text());
            if d["id"] not in dense: raise ValueError(f"missing dense shard for {d['id']}")
            x=one(d,dense[d["id"]]); x.update(condition=label,target=float(d.get("target", label.rsplit("s",1)[-1])/100)); out.append(x); all_rows.append(x)
        if not out: continue
    write_json(root/"comparison.json", all_rows)
    fields=list(all_rows[0]) if all_rows else []
    with (root/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    summaries=[]
    for label in sorted({x["condition"] for x in all_rows}):
        g=[x for x in all_rows if x["condition"]==label]
        def w(k): return sum(x[k]*x["eligible"] for x in g)/max(1,sum(x["eligible"] for x in g))
        ge=sum(x["global_eligible"] for x in g); le=sum(x["local_eligible"] for x in g)
        summaries.append(dict(condition=label, benchmark=sorted({x["benchmark"] for x in g}), examples=len(g),
            accuracy=sum(x["score"] for x in g)/len(g), dense_accuracy=sum(x["dense_score"] for x in g)/len(g),
            delta_pp=sum(x["delta_pp"] for x in g)/len(g), actual_sparsity=w("actual_sparsity"),
            global_sparsity=sum(x["global_skipped"] for x in g)/max(1,ge),
            local_sparsity=sum(x["local_skipped"] for x in g)/max(1,le),
            retained_mass=w("retained_mass"), token_agreement=sum(x["token_agreement"] for x in g)/len(g),
            sequence_exact=sum(x["sequence_exact"] for x in g)/len(g)))
    write_json(root/"comparison_summary.json", summaries)
    lines=["# Dense-paired native adaptive report", "", "All reported sparsity values use aggregate eligible physical tiles; retained mass is the dense probability mass left after the adaptive mask.", "", "| condition | examples | accuracy | dense | delta (pp) | actual sparsity | global | local | mass | token agreement | exact sequence |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in summaries:
        lines.append(f"| {x['condition']} | {x['examples']} | {100*x['accuracy']:.2f}% | {100*x['dense_accuracy']:.2f}% | {x['delta_pp']:+.2f} | {100*x['actual_sparsity']:.2f}% | {100*x['global_sparsity']:.2f}% | {100*x['local_sparsity']:.2f}% | {100*x['retained_mass']:.2f}% | {100*x['token_agreement']:.2f}% | {100*x['sequence_exact']:.2f}% |")
    (root/"comparison_report.md").write_text("\n".join(lines)+"\n")
    return summaries


def write_json(path,obj):
    path=Path(path); path.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--root",required=True); p.add_argument("--source",required=True); a=p.parse_args(); print(json.dumps(run(a.root,a.source),indent=2))
