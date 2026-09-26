"""v15 setup/compatibility executions (<=10, never scored, never pooled with the panel).

One eligible NON-panel item (the excluded v14 cost-profile item 66f39aa7821e116aacb2da76, 17.6K tokens),
seed 42 (not a panel seed), attempt 0 only, arms D_native / T_G_original / CVM_T copied from the
frozen v15 protocol. Purpose: complete long-context request path, memory headroom, request time.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import v15_longbench_task as task
from scripts.v10_request_runs import arm_config
from scripts.v13_seed_runs import arm_config_hash, plan_schedule

ITEM = '66f39aa7821e116aacb2da76'
ARMS = ('D_native', 'T_G_original', 'CVM_T')


def main(private, out):
    from dllm.models import create_adapter
    main_protocol = json.loads((Path(out) / 'frozen_protocol.json').read_text())
    item = next(r for r in json.loads(task.LB_DATA.read_text()) if r['_id'] == ITEM)
    adapter = create_adapter('diffusion_gemma', main_protocol['model_path'], device='cpu', precision='float32',
                             revision=main_protocol['model_revision']).load_tokenizer()
    r = task.render(adapter, item)
    row = dict(id=f'longbench_v2/{ITEM}', source_id=ITEM, benchmark='longbench_v2', prompt=r['prompt'], prompt_hash=r['prompt_hash'],
               prompt_tokens=r['prompt_tokens'], prompt_tokens_n=r['prompt_tokens_n'], thinking=True, generation_budget=8192)
    Path(private).mkdir(parents=True, exist_ok=True)
    manifest = Path(private) / 'smoke_manifest.json'
    manifest.write_text(json.dumps([row]) + '\n')
    arms = {k: main_protocol['arms'][k] for k in ARMS}
    pid = 'v15_lbv2_smoke'
    ns = SimpleNamespace(phase=pid, ids=[row['id']], manifest=manifest, policy=Path(main_protocol['policy_file']),
                         model=Path(main_protocol['model_path']), revision=main_protocol['model_revision'], seeds=[42])
    hashes = {k: arm_config_hash(arm_config(ns, a)) for k, a in arms.items()}
    if any(hashes[k] != main_protocol['arm_hashes'][k] for k in ARMS):
        raise RuntimeError('smoke arm identity differs from the frozen panel arms')
    proto = dict(schema='v15_smoke_protocol_v1', protocol_id=pid, ids=[row['id']], development_ids=[row['id']], seeds=[42],
                 model_path=main_protocol['model_path'], model_revision=main_protocol['model_revision'],
                 policy_file=main_protocol['policy_file'], arms=arms, arm_hashes=hashes,
                 schedule=plan_schedule(pid, main_protocol['model_revision'], hashes, [row['id']], [42], 1, warm_repeats=0),
                 prompt_tokens_n=row['prompt_tokens_n'], purpose=__doc__)
    (Path(out) / 'smoke_protocol.json').write_text(json.dumps(proto, indent=2, sort_keys=True) + '\n')
    print(pid, len(proto['schedule']), row['prompt_tokens_n'])


if __name__ == '__main__':
    main(*sys.argv[1:])
