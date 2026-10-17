"""Independent input-fingerprint/RNG/paired-settings check, CPU only."""
import argparse
import json
from pathlib import Path
from scripts.v31_value_pause_analysis import read, lines, sha, require, CURRENT, OLD
from scripts.v31_vllm_paired_bench import request_seed
from scripts import v31_official as official


def run(output):
    require(not output.exists(),'preserve integrity report')
    result={}
    for name,study in [('current',CURRENT),('old',OLD)]:
        protocol=read(study/'protocol.json');inputs={}
        for rel,digest in protocol['input_fingerprints'].items():
            if rel.startswith(('private/pools/longbench_v2/','private/schedules/longbench_v2/')):
                require(sha(study/rel)==digest,'input fingerprint mismatch: '+rel)
                inputs[rel]=digest
        marker_glob='runs/*/*/shard*/attempt*/attempt_complete.json' if name=='current' else 'shared_runs/longbench_v2/shard*/repeat*/attempt*/attempt_complete.json'
        bindings={};trials=0
        for marker in sorted(study.glob(marker_glob)):
            for p in sorted((marker.parent/'exports').glob('*/records.jsonl')):
                for r in lines(p):
                    cell=dict(dataset=r['dataset'],index=r['index'],seed=r['panel_seed'])
                    require(r['rng_seed']==request_seed(31,cell,r['repeat']),'RNG derivation')
                    key=(r['dataset'],r['index'],r['panel_seed'],r['repeat'])
                    settings={f:r[f] for f in official.COMPARE_FIELDS}
                    if key in bindings:require(settings==bindings[key],'paired settings differ')
                    else:bindings[key]=settings
                    trials+=1
        result[name]=dict(trials=trials,cells=len(bindings),input_sha256=inputs,protocol_sha256=sha(study/'protocol.json'),
                          derived_rng_and_paired_settings_match=True)
    require(result['current']['trials']==928 and result['old']['trials']==768,'coverage')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(status='passed',source_sha256=sha(__file__),studies=result),indent=2,sort_keys=True)+'\n')
    print({k:v['trials'] for k,v in result.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
