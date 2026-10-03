from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from experiments.value_direction_hopper.sparsity_steps import StoppingProbe, routing_counts, conditions
from experiments.value_direction_hopper.sparsity_steps_report import bootstrap


def test_force_step_keeps_stopper_observation():
    record={}
    class Stopper:
        calls=0
        def __call__(self):
            self.calls+=1
            return torch.tensor([True])
    inner=Stopper()
    assert not StoppingProbe(inner,record,True)().item()
    assert record['would_stop'] and inner.calls==1
    assert StoppingProbe(inner,record,False)().item()


def test_counts_are_tile_weighted_and_split_correctly():
    counts=routing_counts([dict(attention_type='local',eligible=10,skipped=9),
                           dict(attention_type='global',eligible=90,skipped=9)])
    assert counts['whole']==dict(eligible=100,skipped=18)
    assert counts['local']==dict(eligible=10,skipped=9)
    assert counts['global']==dict(eligible=90,skipped=9)
    with pytest.raises(ValueError):routing_counts([dict(attention_type='local',eligible=1,skipped=2)])


def test_conditions_and_task_bootstrap():
    c=conditions()
    assert len(c['adaptive'])==12 and len(c['fixed4'])==6
    assert 'gaussian32_s40' not in c['fixed4']
    assert bootstrap([1,1,3,3],['a','a','b','b'])==[2.,2.]
    assert bootstrap([1,2,3,7],['a','a','b','b'])==bootstrap([1,2,3,7],['a','a','b','b'])


def test_provenance_accepts_migration_but_rejects_unrelated_sources(tmp_path):
    from experiments.value_direction_hopper.cuda import source_digest
    actual=tmp_path/'actual.so';actual.touch()
    alias=tmp_path/'alias.so';alias.symlink_to(actual)
    assert source_digest({str(alias):'original-hash'},actual)=='original-hash'
    assert source_digest({str(tmp_path/'unrelated.so'):'original-hash'},actual) is None
    with pytest.raises(RuntimeError):
        source_digest({str(alias):'a',str(actual):'b'},actual)
