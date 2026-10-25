"""Official private-output scoring and descriptive rubric, outside clean timing."""
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(REPO))
from scripts import v31_official as official
from scripts import v27_humaneval as he
from text_metrics import metrics


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class Scorer:
    def __init__(self,root):
        self.root=root
        self.rows={x['id']:x for x in json.loads((root/'private/manifest.json').read_text())}
        paths=json.loads((root/'private/runtime_paths.json').read_text())
        raw=json.loads(Path(paths['gold_path']).read_text())
        self.gold={v['task_id']:v for v in raw.values()}
        self.final,self.final_source=official.load_final_response()
        self.numeric,self.numeric_source=official.aime_numeric_score()
        he.sandbox_preflight()

    def score(self,record,path):
        raw=json.loads(path.read_text());assert raw['output_sha256']==record['output_sha256']
        row=self.rows[record['id']];text=raw['completion'];answer=self.final(text,row['thinking'])
        detail={}
        if row['benchmark']=='aime26':
            v=self.numeric(answer,row['expected']);correct=v['correct'];detail['parsed']=v['extracted'] is not None
        else:
            problem=self.gold[row['source_id']];code,reason=he.extract_code(text,problem['entry_point'])
            correct,outcome=he.run_test(he.program_for(problem,code)) if code is not None else (False,'not_run')
            detail.update(extraction=reason,outcome=outcome,parsed=code is not None)
        return dict(id=record['id'],seed=record['seed'],arm=record['arm'],benchmark=row['benchmark'],correct=bool(correct),
            no_final_response=not answer,quality_detail=detail,text_metrics=metrics(text),
            private_generation_sha256=sha(path),output_sha256=record['output_sha256'],stratum=row['stratum'],A1_exposed=row['A1_exposed'])


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',required=True)
    p.add_argument('--benchmark',required=True);p.add_argument('--arm',required=True);a=p.parse_args()
    leaf=a.root/'runs'/a.phase/a.benchmark/a.arm
    assert json.loads((leaf/'final_complete.json').read_text())['passed']
    scorer=Scorer(a.root);rows=[];inputs={}
    for path in sorted(leaf.glob('clean???.json')):
        record=json.loads(path.read_text());private=a.root/'private/generations'/a.phase/a.benchmark/a.arm/path.name
        rows.append(scorer.score(record,private));inputs[str(path.relative_to(a.root))]=sha(path)
    with (leaf/'official_scores.json').open('x') as f:json.dump(dict(outcomes=rows,input_hashes=inputs,source_sha256=sha(Path(__file__))),f,indent=2)
    print(json.dumps(dict(arm=a.arm,phase=a.phase,benchmark=a.benchmark,scored=len(rows),correct=sum(x['correct'] for x in rows))))


if __name__=='__main__':main()
