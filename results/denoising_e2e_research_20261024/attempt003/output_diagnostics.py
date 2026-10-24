"""Private-text repetition diagnostics; publish only IDs and scalar rates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,'/home/exouser/ljy/dlm')
from scripts import v31_official


def summarize(text):
    words=re.findall(r'\S+',text)
    ngrams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    counts=Counter(ngrams)
    return dict(characters=len(text),whitespace_words=len(words),
        repeated_4gram_fraction=(1-len(counts)/len(ngrams)) if ngrams else 0.,
        maximum_4gram_repetitions=max(counts.values(),default=0))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();out=root/'output_diagnostics001';out.mkdir(exist_ok=False)
    final,source=v31_official.load_final_response();values=[];hashes={}
    for record in (root/'runs/panel').glob('*/*/clean[0-9][0-9][0-9].json'):
        r=json.loads(record.read_text());suite,arm=record.parent.parent.name,r['arm']
        p=root/'private/generations/panel'/suite/arm/record.name
        private=json.loads(p.read_text());raw=private['completion'];answer_text=final(raw,True)
        values.append(dict(id=r['id'],seed=r['seed'],arm=arm,benchmark=r['benchmark'],
            full_output=summarize(raw),final_region=summarize(answer_text),no_final_response=not answer_text,
            T=r['T'],finish_reason=r['finish_reason'],generation_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        hashes[str(record.relative_to(root))]=hashlib.sha256(record.read_bytes()).hexdigest()
    result=dict(rows=values,definition='Whitespace4gram repetition is a descriptive degeneracy signal, not a correctness metric or tokenizer-level repetition rate.',
        final_response_source=source,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'input_hashes.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(records=len(values),private_text_published=False)))


if __name__=='__main__':main()
