"""Quantify exact-mass nonempty rescue in the initial proxy/ranking screen."""
import json
from collections import defaultdict
from .protocol import ROOT,prepare,sha,fingerprint
from .run import shard_path
from .report import csv_write
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write


def audit(root=ROOT):
    setup=prepare(root);groups=defaultdict(lambda:dict(rows=0.,rescued_rows=0.));sources={}
    for row in setup['calibration']:
        path=shard_path(root,'screen','dense',row['id']);out=json.loads(path.read_text())
        assert out['fingerprint']==fingerprint(root)
        sources[str(path)]=sha(path.read_bytes())
        for r in out['records']:
            if not r['probe'].startswith(('sol/','oracle/qk/')):continue
            for kind in ('overall',r['attention_type']):
                target=groups[row['benchmark'],r['probe'],kind]
                target['rows']+=r['rows'];target['rescued_rows']+=r['rescued_rows']
    rows=[dict(benchmark=b,probe=p,attention_type=k,**v,rescued_row_rate=v['rescued_rows']/max(v['rows'],1)) for (b,p,k),v in sorted(groups.items())]
    result=dict(fingerprint=fingerprint(root),heldout_used=False,rows=rows,sources=sources,
        definition='Rows empty after primary proxy/ranking selection, before generic exact-attention-mass rescue; counted valid-row weighted',
        concern='An exact-mass fallback can contaminate a nominally proxy-only or QK-ranking-only criterion; initial screening results must disclose it')
    _write(root/'initial_ranking_fallback_audit.json',result);csv_write(root/'initial_ranking_fallback_audit.csv',rows)
    print(json.dumps([r for r in rows if r['attention_type']=='overall' and (r['probe'].endswith('s90') or r['probe'].endswith('s50'))],indent=2))


if __name__=='__main__':audit()
