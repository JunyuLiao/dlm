"""Outcome-blind cohort and exposure identity; original prompts remain private."""
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(REPO))
from experiments.diffusion_gemma_sparse_attention_research.dynamics_screen import AIME_SOURCE,HUMANEVAL_SOURCE


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];prior=REPO/'results/denoising_e2e_research_20261024/attempt003'
    previous=json.loads((prior/'private/tokenized_manifest.json').read_text())
    old={r['id']:r for r in previous};panel=json.loads((prior/'e2e_protocol.json').read_text())['panel']['short']
    legacy=list(dict.fromkeys(x['id'] for x in panel));legacyhe=[x for x in legacy if x.startswith('humaneval/')]
    a1=json.loads((REPO/'results/denoising_mechanism_stage_a_20261025/attempt002/manifest_identity.json').read_text())
    exposed_a1={x['id'] for x in a1['identities']}
    aime=json.loads(AIME_SOURCE.read_text());human=json.loads(HUMANEVAL_SOURCE.read_text())
    assert len(aime)==len({str(x['source_id']) for x in aime})==30
    assert len(human)==len({x['source_id'] for x in human})==100
    pools={}
    for benchmark,rows in [('aime26',aime),('humaneval',human)]:
        for row in rows:
            item=dict(row,id=benchmark+'/'+str(row['source_id']))
            assert item['id'] not in pools;pools[item['id']]=item
    newhe=sorted([x for x in pools if x.startswith('humaneval/') and x not in old and x not in exposed_a1],
        key=lambda x:hashlib.sha256(('response-replication-20261025|'+x).encode()).digest())[:30]
    assert len(newhe)==30
    ids=[x for x in legacy if x.startswith('aime26/')]+sorted([x for x in pools if x.startswith('aime26/') and x not in old],key=lambda x:int(x.split('/')[-1]))+legacyhe+newhe
    assert len(ids)==len(set(ids))==70
    rows=[];public=[]
    for identity in ids:
        r=pools[identity]
        if identity in old:
            assert all(r[k]==old[identity][k] for k in ['prompt','prompt_tokens','thinking','generation_budget','prompt_hash'])
        r.update(stratum='legacy_response' if identity in old else 'new_to_response',A1_exposed=identity in exposed_a1,
            historical_benchmark_exposure='Present in inherited V31 value-matrix source pool; individual prior native treatment exposure not certified absent',
            primary_confirmation=identity not in old)
        rows.append(r)
        public.append({k:r[k] for k in ['id','source_id','benchmark','stratum','A1_exposed','historical_benchmark_exposure','primary_confirmation','thinking','generation_budget','prompt_hash']} |
            dict(seeds=[42,43],prompt_token_count=len(r['prompt_tokens']),prompt_tokenization_sha256=hashlib.sha256(json.dumps(r['prompt_tokens']).encode()).hexdigest()))
    # Explicit legacy gate includes the previously observed coding regression.
    gate=[dict(id='aime26/18',seed=42),dict(id='aime26/9',seed=43),dict(id='humaneval/HumanEval/71',seed=43),dict(id='humaneval/HumanEval/124',seed=42)]
    full={b:[dict(id=r['id'],seed=s) for s in [42,43] for r in rows if r['benchmark']==b] for b in ['aime26','humaneval']}
    (root/'private/manifest.json').write_text(json.dumps(rows))
    identity=dict(identities=public,source_pool_counts={'aime26':30,'humaneval':100},selected_counts={'aime26':30,'humaneval':40},
        source_pool_sha256={'aime26':sha(AIME_SOURCE),'humaneval':sha(HUMANEVAL_SOURCE)},legacy_reproduction=gate,panel=full,
        legacy_order=legacy,selection='All unique AIME; original10HE plus30 hash-ordered additions excluding original Response and A1 identities; no outcome filtering.',
        exposure_registry_scope='Original Response25,A1 twelve,and inherited V31 matrix pools; absence from other historical studies is not certified.',
        private_manifest_sha256=sha(root/'private/manifest.json'))
    with (root/'manifest_identity.json').open('x') as f:json.dump(identity,f,indent=2,sort_keys=True)
    with (root/'exposure_registry.json').open('x') as f:json.dump(dict(identities=public,known_studies=['denoising_e2e_research_20261024/attempt003','denoising_mechanism_stage_a_20261025/attempt002','v31_value_matrix_20261011'],no_universal_nonexposure_claim=True),f,indent=2)
    print(json.dumps(dict(questions=len(ids),AIME=30,HE=40,primary_AIME=sum(x['primary_confirmation'] for x in public if x['benchmark']=='aime26'),primary_HE=30)))


if __name__=='__main__':main()
