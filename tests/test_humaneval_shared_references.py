from copy import deepcopy
import json
from pathlib import Path
import pytest
from scripts import v31_humaneval_cfgdet_study as campaign


def protocol(k):
    return dict(prefix_budget_tokens=k,observing_fa4_no_overlap=True,runtime={'gpu':'same'},native={'max_new_tokens':8192},
        manifests={'source':'same'},humaneval_source_sha256='data',model_inventory_sha256='model',
        cohort=[0,23,129,106,145,116],seed=42,extra_seed=43,primary_shard_count=1,
        suites={'thinking_on':dict(mage_k=k,temperature=[.8,.4],chunk=4096)})


def test_controls_require_identical_source_runtime_data_and_sampler():
    a,b=protocol(128),protocol(256);source={'execution':{'one':'hash'},'commit':'same'}
    assert campaign.reference_compatible(a,b,source,deepcopy(source))
    for key in ('runtime','native','manifests','humaneval_source_sha256','model_inventory_sha256','cohort','seed','extra_seed','primary_shard_count'):
        changed=deepcopy(b);changed[key]='different'
        assert not campaign.reference_compatible(a,changed,source,source)
    assert not campaign.reference_compatible(a,b,source,{'commit':'different'})
    b['suites']['thinking_on']['temperature']=[.7,.4]
    assert not campaign.reference_compatible(a,b,source,source)
    a,b=protocol(128),protocol(256);del a['runtime'];del b['runtime']
    assert not campaign.reference_compatible(a,b,source,source)


def test_only_retention_budget_may_differ_in_suite_settings():
    a,b=protocol(128),protocol(256);s={'source':'same'}
    b['suites']['thinking_on']['chunk']=2048
    assert not campaign.reference_compatible(a,b,s,s)
    b=protocol(256);b['observing_fa4_no_overlap']=False
    assert not campaign.reference_compatible(a,b,s,s)


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_shared_leaf_must_be_in_its_exact_control_owner_location(tmp_path,monkeypatch):
    own,owner=tmp_path/'k128',tmp_path/'k256';write(own/'protocol.json',{'own':128});write(owner/'protocol.json',{'own':256})
    family=campaign.REFERENCES[0];relative=Path('runs/thinking_on/audit')/family/'primary_shard001/attempt001'
    leaf=owner/relative;leaf.mkdir(parents=True);(leaf/'records.jsonl').write_text('{}\n')
    done=dict(protocol_sha256=campaign.sha(owner/'protocol.json'),files_sha256={'records.jsonl':campaign.sha(leaf/'records.jsonl')})
    write(leaf/'attempt_complete.json',done)
    target=(own/relative).parents[1];target.parent.mkdir(parents=True);target.symlink_to((owner/relative).parents[1],target_is_directory=True)
    monkeypatch.setattr(campaign,'reference_owner',lambda *a:owner)
    assert campaign.verify_leaf(own,own/relative)==done
    done['protocol_sha256']='wrong';write(leaf/'attempt_complete.json',done)
    with pytest.raises(RuntimeError,match='protocol'):campaign.verify_leaf(own,own/relative)
    done['protocol_sha256']=campaign.sha(owner/'protocol.json');write(leaf/'attempt_complete.json',done)
    (leaf/'records.jsonl').write_text('mutated\n')
    with pytest.raises(RuntimeError,match='artifact'):campaign.verify_leaf(own,own/relative)


def test_reference_links_never_include_a_selector_family(tmp_path,monkeypatch):
    own,owner=tmp_path/'k128',tmp_path/'k256'
    write(owner/'independent_verification.json',dict(status='passed'))
    write(owner/'independent_rescoring.json',dict(mismatches=0,qualification_mismatches=0))
    for stage in ('qualification','runs'):
        for mode in campaign.MODES:
            for part in ('audit','clean'):
                for family in campaign.REFERENCES:
                    leaf=owner/stage/mode/part/family/'primary_shard001/attempt001'
                    write(leaf/'attempt_complete.json',{})
                    write(leaf/f'exports/{family}/records.jsonl',{})
    checked=[];monkeypatch.setattr(campaign,'reference_owner',lambda *a:owner)
    monkeypatch.setattr(campaign,'verify_leaf',lambda root,leaf:checked.append((root,leaf)))
    campaign.link_references(own,{})
    assert len(checked)==24
    assert (own/'runs/thinking_on/clean'/campaign.REFERENCES[0]).is_symlink()
    assert not (own/'runs/thinking_on/clean/shared').exists()
    assert all(owner==root and any(family in leaf.parts for family in campaign.REFERENCES) for root,leaf in checked)
    selected=campaign.selected_paths(own,'thinking_on','audit')
    assert all(len(selected[family])==1 for family in campaign.REFERENCES)
    assert not selected['control_before']


def test_allocation_chain_keeps_total_and_rejects_ledger_drift(tmp_path,monkeypatch):
    inv=tmp_path/'investigation';inv.mkdir();(tmp_path/'spec.md').write_text('unchanged global ceiling')
    write(inv/'protocol.json',{'frozen':True})
    caps={'diagnosis':10800,'k256':16200,'k128':16200}
    first=dict(spec_file='spec.md',spec_sha256=campaign.sha(tmp_path/'spec.md'),
        prior_protocol_sha256=campaign.sha(inv/'protocol.json'),caps_s=caps)
    write(inv/'allocation_addendum.json',first);prior=inv/'allocation_addendum.json'
    for number,next_caps in ((2,{'diagnosis':10800,'k256':19800,'k128':12600}),
                             (3,{'diagnosis':10478,'k256':19400,'k128':0,'evaluation':13322})):
        path=inv/f'allocation_addendum{number:03d}.json'
        write(path,dict(prior_allocation_sha256=campaign.sha(prior),prior_caps_s=caps,
            spec_file='spec.md',spec_sha256=campaign.sha(tmp_path/'spec.md'),total_cap_s=43200,caps_s=next_caps))
        prior,caps=path,next_caps
    write(inv/'gpu_budget.json',dict(caps_s=caps,allocation_addendum_sha256=campaign.sha(prior)))
    monkeypatch.setattr(campaign,'ROOT',tmp_path);monkeypatch.setattr(campaign,'INVESTIGATION',inv)
    assert campaign.phase_caps()==caps and campaign.allocation_file()==prior
    bad=dict(caps);bad['evaluation']+=1
    write(inv/'gpu_budget.json',dict(caps_s=bad,allocation_addendum_sha256=campaign.sha(prior)))
    with pytest.raises(RuntimeError,match='ledger'):campaign.phase_caps()


def test_owner_verification_preserves_requesting_budget_environment(tmp_path,monkeypatch):
    own,owner=tmp_path/'k128',tmp_path/'k256';a,b=protocol(128),protocol(256);source={'execution':'identical'}
    write(owner/'protocol.json',b);write(owner/'receipts/source_fingerprints.json',source)
    write(own/'receipts/source_fingerprints.json',source)
    a['shared_references']=dict(study='k256',families=list(campaign.REFERENCES),protocol_sha256=campaign.sha(owner/'protocol.json'),
        source_fingerprints_sha256=campaign.sha(owner/'receipts/source_fingerprints.json'))
    monkeypatch.setattr(campaign,'ROOT',tmp_path);monkeypatch.setattr(campaign.base,'SUITES',{})
    monkeypatch.setattr(campaign,'verify',lambda path:campaign.base.SUITES.update(b['suites']))
    assert campaign.reference_owner(own,a)==owner
    assert campaign.base.SUITES['thinking_on']['mage_k']==128


def test_total_limit_revision_requires_explicit_bounded_task_scope(tmp_path,monkeypatch):
    inv=tmp_path/'investigation';inv.mkdir();(tmp_path/'spec.md').write_text('self-imposed estimate revised before generation')
    write(inv/'protocol.json',{'frozen':True});caps={'diagnosis':10800,'k256':16200,'k128':16200}
    write(inv/'allocation_addendum.json',dict(spec_file='spec.md',spec_sha256=campaign.sha(tmp_path/'spec.md'),
        prior_protocol_sha256=campaign.sha(inv/'protocol.json'),caps_s=caps))
    changed=dict(prior_allocation_sha256=campaign.sha(inv/'allocation_addendum.json'),prior_caps_s=caps,
        spec_file='spec.md',spec_sha256=campaign.sha(tmp_path/'spec.md'),prior_total_cap_s=43200,total_cap_s=65878,
        caps_s={'diagnosis':10478,'k256':19400,'k128':0,'evaluation':36000})
    path=inv/'allocation_addendum002.json';write(path,changed)
    write(inv/'gpu_budget.json',dict(caps_s=changed['caps_s'],allocation_addendum_sha256=campaign.sha(path)))
    monkeypatch.setattr(campaign,'ROOT',tmp_path);monkeypatch.setattr(campaign,'INVESTIGATION',inv)
    with pytest.raises(RuntimeError,match='binding'):campaign.phase_caps()
    changed.update(self_imposed_limit_revision=True,scope='humaneval_corrected_k256_k128');write(path,changed)
    write(inv/'gpu_budget.json',dict(caps_s=changed['caps_s'],allocation_addendum_sha256=campaign.sha(path)))
    assert campaign.phase_caps()['evaluation']==36000
