"""Read-only hashes and canonical completion checks, writing one new closeout leaf."""
from pathlib import Path
import json,hashlib,subprocess,datetime
S=Path(__file__).resolve().parent;REPO=S.parents[1];OUT=S/'closeout001';CACHE={}
def sha(p):
    p=p.resolve();key=(str(p),p.stat().st_size,p.stat().st_mtime_ns)
    if key not in CACHE:
        h=hashlib.sha256()
        with p.open('rb') as f:
            while x:=f.read(8*1024*1024):h.update(x)
        CACHE[key]=h.hexdigest()
    return CACHE[key]
def load(p):return json.loads(p.read_text())
def main():
    OUT.mkdir(exist_ok=False);checks={};source_count=0
    for attempt in ['attempt001','attempt002','attempt003','attempt004']:
        root=S/attempt;p=load(root/'protocol.json')
        for name,h in {**p['sources'],**p['scorer_sources']}.items():assert sha(REPO/name)==h,name;source_count+=1
        checks[attempt+'_source_pins']=True
    checks['live_branch']=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='denoising_trajectory_research';assert checks['live_branch']
    assert sha(REPO/'results/native_support_discovery_20261026/experiment/attempt006/protocol.json')=='10398a912c355dabf3d6511475a7bdf0fb3c3434c0e7055e439a724fd5bc0d70'
    old=REPO/'results/native_support_discovery_20261026/experiment/attempt006/code';old_checks={}
    for p in old.glob('*.py'):
        blob=subprocess.check_output(['git','show','c3f3fcdf1cbd1c9f116ef66d992d5aa574173984:'+str(p.relative_to(REPO))],cwd=REPO);old_checks[p.name]=hashlib.sha256(blob).hexdigest()==sha(p);assert old_checks[p.name]
    count=0
    maps=[(S/'ranking_audit/attempt001/input_hashes.json',REPO),(S/'ranking_audit/selection_pool002/input_hashes.json',REPO),(S/'ranking_audit/pool_eligibility001/input_hashes.json',REPO),(S/'prior_policy_quality001/input_hashes.json',REPO),(S/'attempt002/numerical_qualification002/input_hashes.json',S/'attempt002'),(S/'attempt002/analysis001/input_hashes.json',S/'attempt002'),(S/'attempt002/analysis003/input_hashes.json',S/'attempt002'),(S/'physical_alias_audit001/input_hashes.json',S),(S/'applied_numerical_audit001/input_hashes.json',S),(S/'common_state_operators001/input_hashes.json',S/'attempt002'),(S/'official_rescore001/input_hashes.json',S),(S/'report001/input_hashes.json',S)]
    for manifest,base in maps:
        for name,h in load(manifest).items():assert sha(base/name)==h,(str(manifest),name);count+=1
    for name,h in load(S/'attempt002/reference_binding.json')['input_hashes'].items():assert sha(S/name)==h;count+=1
    checks['all_consumed_data_hashes']=True
    for n in ['attempt001/controls_complete.json','attempt002/runs/treatments/final_complete.json','attempt004/runs/qualification/final_complete.json','physical_alias_audit001/final_complete.json','applied_numerical_audit001/final_complete.json','common_state_operators001/final_complete.json','official_rescore001/final_complete.json','attempt002/analysis001/final_complete.json','attempt002/analysis003/final_complete.json','report001/final_complete.json']:
        assert load(S/n)['passed'];checks[n]=True
    a=load(S/'attempt002/analysis001/method_decisions.json');b=load(S/'attempt002/analysis003/method_decisions.json');assert len(a)==len(b)==32 and all(not x['advance'] for x in a+b);checks['original_and_repaired_gates_both_no_go']=True
    assert sha(S/'attempt002/branch_manifest.json')==load(S/'attempt002/branch_binding.json')['branch_manifest_sha256'];checks['pre_outcome_choices_unchanged']=True
    protected=load(S/'initial_state.json')['protected'];preservation={p:sha(REPO/p)==h for p,h in protected.items()};assert all(preservation.values());checks['initial_unrelated_edits_unchanged_before_documentation']=True
    paths=load(S/'private/runtime_paths.json');model=[]
    for r in load(S/'attempt004/model_fingerprints.json')['files']:
        actual=sha(Path(paths['model_path'])/r['file']);assert actual==r['sha256'];model.append(dict(file=r['file'],sha256=actual,verified=True))
    package=Path(paths['python']).parents[1]/'lib/python3.12/site-packages/vllm'
    for name,h in load(S/'attempt004/protocol.json')['runtime_source_pins'].items():assert sha(package/name)==h
    assert sha(Path(paths['gold_path']))==load(S/'attempt004/environment_qualification.json')['official_gold_sha256'];checks['model_runtime_gold']=True
    hardware=load(S/'hardware_final001.json');assert hardware['all_study_GPU_workers_finished'] and not hardware['compute_pids'];checks['all_GPU_workers_finished']=True
    report=dict(passed=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,source_pin_checks=source_count,consumed_data_hash_checks=count,old_native_sources_unchanged=old_checks,protected_initial_edits=preservation,model_files=model,scope='Complete bounded development and qualification; zero promoted methods; no clean-online/V31 rerun',no_push=True)
    with (OUT/'verification.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True)
    print(json.dumps(dict(passed=True,source_pin_checks=source_count,data_hash_checks=count,model_files=len(model))))
if __name__=='__main__':main()
