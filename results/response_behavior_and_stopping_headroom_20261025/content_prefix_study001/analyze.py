"""Blinded supplemental descriptive prefix extraction; never an early-stop rule."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

BASE=Path(__file__).resolve().parents[1]
R=BASE/'track_r/attempt001';REPO=BASE.parents[1]
sys.path.insert(0,str(R/'code'));sys.path.insert(0,str(REPO))
from scoring import Scorer


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prefix_metrics(text,row,scorer):
    # Character checkpoints, not emission times/canvases. An extractable prefix
    # may be wrong or syntactically valid before the eventual complete answer.
    points={len(text)}
    points.update(m.end() for m in re.finditer(r'\n|\}|```',text))
    first=None
    for point in sorted(points):
        prefix=text[:point]
        if row['benchmark']=='aime26':
            value=scorer.numeric(scorer.final(prefix,True),'0')
            available=value['extracted'] is not None
        else:
            problem=scorer.gold[row['source_id']]
            available=scorer_module_he_extract(prefix,problem['entry_point'])
        if available:first=point;break
    return dict(first_officially_extractable_checkpoint_character=first,
        checkpoint_character_fraction=first/max(1,len(text)) if first is not None else None,
        full_output_characters=len(text),checkpoints=len(points),
        endpoint='first official-parser candidate among newline/brace/fence/full-output character checkpoints',
        correctness_of_prefix_not_tested=True,native_stopping_time_not_measured_by_this_endpoint=True)


def scorer_module_he_extract(text,entry):
    from scripts import v27_humaneval as he
    code,_=he.extract_code(text,entry)
    return code is not None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['original','panel'],required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parent;protocol=json.loads((root/'protocol.json').read_text())
    assert sha(Path(__file__))==protocol['analysis_source_sha256']
    if args.scope=='panel':assert (R/'readout_frozen.json').exists()
    out=root/(args.scope+'001');out.mkdir(exist_ok=False);scorer=Scorer(R);start=time.time();rows=[];inputs={}
    if args.scope=='original':
        old=REPO/'results/denoising_e2e_research_20261024/attempt003'
        panel=json.loads((old/'e2e_protocol.json').read_text())['panel']['short']
        locations=[(arm,identity,old/f'private/generations/panel/short/{arm}/clean{i:03d}.json') for arm in protocol['arms'] for i,identity in enumerate(panel)]
    else:
        panel=json.loads((R/'protocol.json').read_text())['panel']
        locations=[(arm,identity,R/f'private/generations/panel/{benchmark}/{arm}/clean{i:03d}.json') for benchmark,cells in panel.items() for arm in protocol['arms'] for i,identity in enumerate(cells)]
    for arm,identity,path in locations:
        private=json.loads(path.read_text());row=scorer.rows[identity['id']]
        rows.append(dict(arm=arm,**identity,benchmark=row['benchmark'],stratum=row['stratum'],**prefix_metrics(private['completion'],row,scorer)))
        inputs[arm+'|'+identity['id']+'|'+str(identity['seed'])]=sha(path)
    expected=200 if args.scope=='original' else 700;assert len(rows)==expected
    with (out/'summary.json').open('x') as f:json.dump(dict(passed=True,rows=rows,private_input_hashes_by_identity=inputs,allocated_CPU_seconds=time.time()-start,
        protocol_sha256=sha(root/'protocol.json'),source_sha256=sha(Path(__file__)),new_model_forwards=0,
        interpretation='Supplemental descriptive character-checkpoint candidates only; no prefix gold scoring, no native early termination, no causal content explanation.'),f,indent=2)
    print(json.dumps(dict(scope=args.scope,outputs=len(rows),CPU_seconds=time.time()-start)))


if __name__=='__main__':main()
