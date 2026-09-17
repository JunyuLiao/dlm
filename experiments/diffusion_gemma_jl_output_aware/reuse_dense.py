"""Populate compatible development aliases without rerunning dense inference.

Safe alongside the calibration worker: only writes absent, not-yet-consumed
development dense caches; never edits the source or a completed destination.
"""
from .protocol import ROOT, prepare, execution, sha
from .runner import cached, write_output
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_value_aware.protocol import frozen_write


def reuse(root=ROOT):
    setup,contract=prepare(root),execution(root)
    final={r['id']:r for r in setup['final']}
    aliases=[]
    for row in [r for r in setup['calibration'] if r['benchmark']=='aime26'][:2]:
        if any(row[k]!=final[row['id']][k] for k in ('prompt','prompt_tokens','prompt_hash','seed','generation_budget')):
            raise ValueError('Development dense reuse is not identical to final dense')
        source=shard_path(root,'dense','dense',row['id'])
        dest=shard_path(root,'development','dense',row['id'])
        out=cached(None,root,row,'dense','dense','dense',{},None,contract)
        if not dest.exists():
            copied=dict(out,imported_source=dict(path=str(source),sha256=sha(source.read_bytes()),original_fingerprint=contract['fingerprint']))
            write_output(dest,copied)
        cached(None,root,row,'development','dense','dense',{},None,contract)
        aliases.append(dict(id=row['id'],source=str(source),source_sha256=sha(source.read_bytes()),
            destination=str(dest),destination_sha256=sha(dest.read_bytes()),inference_performed=False))
    frozen_write(root/'development_dense_aliases.json',aliases)
    return aliases


if __name__=='__main__':
    import json
    print(json.dumps(reuse()))
