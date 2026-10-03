"""Bounded six-probe support-key regroup: CPU proxy, never a speed claim.

Cooperation candidate inspired by Haowei's support-set grouping. The independent
policy samples six fixed evenly spaced prefix tiles, stably groups rows within
each head/Q128 block, and accepts a block only if full support unions improve
tile count or alias2 maximum without worsening either. No quality labels, QK,
current tokens, or offline search enter the key. All row needs remain covered.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scripts.v28_regroup_screen import validate_need, natural_order, group_support, load_need, cost_from_counts


def probe_tiles(prefix_tiles):
    if type(prefix_tiles) is not int or prefix_tiles < 1:
        raise ValueError("Positive prefix tile count required")
    # Duplicates for short prefixes are intentional and deterministic.
    return tuple(i * (prefix_tiles - 1) // 5 for i in range(6))


def support_key_order(need):
    need = validate_need(need)
    h, q, pt = need.shape
    key = np.zeros((h, q), dtype=np.int32)
    for bit, tile in enumerate(probe_tiles(pt)):
        key |= need[:, :, tile].astype(np.int32) << bit
    # Include original offset to make every key unique and order deterministic.
    key = key.reshape(h, q // 128, 128) * 128 + np.arange(128)
    return (np.argsort(key, axis=-1) + 128 * np.arange(q // 128)[None, :, None]).reshape(h, q)


def screen_need(need):
    need = validate_need(need)
    h, q, pt = need.shape
    start = time.perf_counter()
    natural = natural_order(need)
    candidate = support_key_order(need)
    natural_support = group_support(need, natural)
    candidate_support = group_support(need, candidate)
    a = natural_support.sum(-1).reshape(h, q // 128, 2)
    b = candidate_support.sum(-1).reshape(h, q // 128, 2)
    old_total, new_total = a.sum(-1), b.sum(-1)
    old_max, new_max = ((a + 1) // 2).max(-1), ((b + 1) // 2).max(-1)
    accept = (new_total <= old_total) & (new_max <= old_max) & ((new_total < old_total) | (new_max < old_max))
    order = np.where(np.repeat(accept,128,axis=1), candidate, natural)
    support = np.where(np.repeat(accept,2,axis=1)[:,:,None], candidate_support, natural_support)
    # No approximate support is ever consumed. Prefix-only proxy; original row
    # output restoration and all canvas/boundary work are still required.
    seconds = time.perf_counter() - start
    return dict(order=order, support=support, construction_cpu_seconds=seconds,
                accepted_blocks=int(accept.sum()), blocks=accept.size,
                moved_rows=int(np.count_nonzero(order != natural)), total_rows=h*q,
                natural_cost=cost_from_counts(a.reshape(h, q//64)),
                candidate_cost=cost_from_counts(support.sum(-1)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("private_need_dir",type=Path);p.add_argument("new_report",type=Path)
    args=p.parse_args(argv)
    if args.new_report.exists(): raise FileExistsError("Use a new report")
    paths=sorted(args.private_need_dir.glob("*.npz"))
    if not paths: raise ValueError("No snapshots")
    rows=[]
    for index,path in enumerate(paths):
        need=load_need(path);r=screen_need(need)
        rows.append(dict(snapshot_index=index,prefix_tokens=need.shape[-1]*64,
                         **{k:v for k,v in r.items() if k not in ("order","support")}))
    report=dict(schema="v30_regroup_key6_cpu_screen_v1",probe_rule="floor(i*(PT-1)/5), i=0..5; lower-index probe is low bit",
                peer_inspiration_commit="6f7279c1625f7fa53bacb233b96fb69a385df8c1",
                scope="CPU support/load proxy only; no CUDA or E2E execution",gpu_reserved_seconds=0,
                selection="All snapshots in the pre-existing held component inventory; development evidence, not independent confirmation",
                limitations=["Historical prefix-only needs; no observed epoch lifetime", "CPU build timing includes natural/candidate support and gate, excludes data loading",
                             "No GPU construction, permutation, inverse, tail, selector, list, launch or consumer timing",
                             "Coverage is preserved; regroup changes per-row union supersets and requires output/quality qualification"],rows=rows)
    with args.new_report.open("x") as f: json.dump(report,f,indent=2);f.write("\n")
    print(json.dumps({"snapshots":len(rows),"accepted_blocks":sum(r['accepted_blocks'] for r in rows),"gpu_started":False}))


if __name__ == "__main__":main()
