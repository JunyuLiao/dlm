"""Score frozen MATH500 panels with bound prompts and symbolic equivalence.

Uses the standard final-response parser and math-verify's expression/LaTeX
metric. Unclosed thinking has no final answer and is wrong. Outputs are numeric
aggregates, anonymous cell keys, and bindings; no completions or answers.
"""
from __future__ import annotations
import collections
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import v31_official as op


def make_verifier():
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    return math_metric(gold_extraction_target=(LatexExtractionConfig(),),
                       pred_extraction_target=(ExprExtractionConfig(),LatexExtractionConfig()))


def judge(verifier, expected, final):
    if not final.strip():return False,False,False
    expected=expected.strip()
    if expected.startswith('\\(') and expected.endswith('\\)'):expected=expected[2:-2].strip()
    if expected.startswith('$') and expected.endswith('$'):expected=expected[1:-1].strip()
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            score,extracted=verifier([f'\\boxed{{{expected}}}'],[final])
        parsed=bool(extracted is not None and extracted[1])
        return bool(score),parsed,False
    except Exception:
        return False,False,True


def main():
    args=op.scorer_cli(__doc__,('out_prefix','manifest_dir'))
    records,coverage=op.planned_records(args)
    pools=op.Pools(args.manifest_dir,('thinking','gold'))
    final,frsource=op.load_final_response();verifier=make_verifier()
    official={};binding={};diag={}
    for name,k,r in records:
        row,b,verified=op.bind(r,pools,args.legacy_unbound)
        final_text=final(r['completion'],row['thinking'])
        ok,parsed,error=judge(verifier,row['gold']['answer'],final_text)
        op.put(official,name,k,ok);op.put(binding,name,k,b)
        d=diag.setdefault(name,collections.Counter());d['cells']+=1;d['verified']+=verified
        d[op.finish_class(r['finish_reason'])]+=1;d['no_final_response']+=not bool(final_text)
        d['unparsed']+=not parsed;d['verifier_errors']+=error
    summary=dict(metric='MATH500 symbolic answer equivalence (math-verify)',coverage=coverage,
        answer_text=frsource,math_verify_version=importlib.metadata.version('math-verify'),
        arms={name:dict(correct=sum(x.values()),accuracy=sum(x.values())/len(x),**dict(diag[name]))
              for name,x in official.items()})
    op.write_outputs(args.out_prefix,official,{},summary,binding)
    for name,x in summary['arms'].items():print(name,x['correct'], '/',x['cells'])

if __name__=='__main__':main()
