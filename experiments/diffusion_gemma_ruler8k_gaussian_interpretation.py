"""Post-audit interpretation; does not change frozen reports, policies or runs."""
import argparse
from pathlib import Path

from experiments import diffusion_gemma_ruler8k_gaussian_sweep as study

base = study.base


def comparison(rows, candidate, reference):
    for r in rows:
        if (r['candidate'],r['reference']) == (candidate,reference): return r
        if (r['candidate'],r['reference']) == (reference,candidate):
            return dict(r,candidate=candidate,reference=reference,delta=-r['delta'],
                paired_ci95=[-r['paired_ci95'][1],-r['paired_ci95'][0]])
    raise ValueError('Missing paired comparison')


def build(root=study.ROOT):
    audit_path=root/'audit.json';audit=base.read(audit_path)
    verification=base.read(root/'regeneration_verification.json')
    if not audit['complete'] or audit['completed']!=2470 or not verification['passed'] or verification['audit_sha256']!=base.sha(audit_path.read_bytes()):
        raise ValueError('Complete independently regenerated sweep required')
    sources={str(audit_path):base.sha(audit_path.read_bytes()),
        str(root/'regeneration_verification.json'):base.sha((root/'regeneration_verification.json').read_bytes()),
        str(Path(__file__)):base.sha(Path(__file__).read_bytes())}
    names=('summary','comparisons','common_support','shared_operator','non_vt_sensitivity','per_sample')
    sources.update({str(root/(n+'.json')):audit['artifacts'][n+'.json'] for n in names})
    base.evidence.check_sources(sources)
    loaded={n:base.read(root/(n+'.json')) for n in names}
    summary={r['condition']:r for r in loaded['summary']}
    comparisons={name:comparison(loaded['comparisons'],*pair) for name,pair in dict(
        r32_vs_r2=('jl_gaussian_r32_s75','jl_gaussian_r2_s75'),
        r32_vs_full=('jl_gaussian_r32_s75','full_centered_s75'),
        r32_vs_mass=('jl_gaussian_r32_s75','mass_s75'),
        r8_vs_r16=('jl_gaussian_r8_s75','jl_gaussian_r16_s75')).items()}
    recovered=(summary['jl_gaussian_r32_s75']['accuracy']-summary['jl_gaussian_r2_s75']['accuracy'])/(summary['full_centered_s75']['accuracy']-summary['jl_gaussian_r2_s75']['accuracy'])
    vt=[r for r in loaded['per_sample'] if r['condition']=='dense' and r['task']=='vt']
    if len(vt)!=10 or any(len(r['completion_tokens'])!=30 or r['accuracy']!=0 for r in vt):
        raise ValueError('Dense VT budget caveat differs from audited data')
    common=[r for r in loaded['common_support'] if r['axis']=='overall' and r['target']==.75]
    result=dict(complete=True, conditions=19, final_outputs=2470, new_final_outputs=1300,
        reused_final_outputs=1170, comparisons=comparisons, recovered_observed_r2_full_score_gap=recovered,
        common_support_75_primary=[{k:r[k] for k in ('rank','seed','mean_norm_ratio','rms_relative_norm_error',
            'factor_two_underestimate_rate','dangerous_underestimate_rate','tile_disagreement_rate')} for r in common if r['seed']==1729],
        diagnostic_rms_error_seed_ranges={rank:[min(r['rms_relative_norm_error'] for r in common if r['rank']==rank),
            max(r['rms_relative_norm_error'] for r in common if r['rank']==rank)] for rank in study.DIAGNOSTIC_RANKS},
        conclusion='Projection loss is strongly supported as a major contributor to the Gaussian2 collapse at75%, but not established as the sole cause; no monotonic downstream rank guarantee.',
        limitations=['One end-to-end generation/projection seed; other seeds are diagnostic only.',
            'Different fixed, non-nested matrices and independently calibrated thresholds across ranks.',
            'Shared-state diagnostics cover390 calibration snapshots at step0, not later adaptive sparse states.',
            'Full-dimensional immediate centered routing itself loses substantial accuracy at75%.',
            'Previously examined examples; unadjusted paired intervals; VT output-budget artifact.'])
    base._write(root/'interpretation.json',result)
    pct=lambda v:f'{100*v:.2f}'
    lines=['# Does projection information loss explain the RULER8K accuracy drop?', '',
        '**The results strongly support it as a major contributor at75% sparsity. Gaussian32 recovers most of the Gaussian2–full gap. '
        'It is not the only limitation, and downstream accuracy is not monotonic in rank.**', '',
        'Complete, independently audited experiment:130 matched questions (10 each of13 tasks),19 conditions,2470 final outputs. '
        'Added Gaussian1/4/8/16/32 at50% and75%; reused dense, BLASST, mass-only, full-dimensional and Gaussian2 unchanged. '
        'All41 CPU/CUDA tests and the two-input real-model smoke passed; no calibration/generation failures or audit violations.', '',
        '## Main results', '',
        'Official RULER equal-task score, including fractional multi-answer credit. Dense score:89.23%. '
        'Each cell is accuracy at the measured count-weighted physical sparsity.', '',
        '|Method|50% target: actual sparsity %|Score %|75% target: actual sparsity %|Score %|', '|---|---:|---:|---:|---:|']
    names=[('BLASST','blasst'),('Mass-only','mass'),('Full-dimensional','full_centered')]+[(f'Gaussian{r}'+(' (reused)' if r==2 else ''),f'jl_gaussian_r{r}') for r in study.DIAGNOSTIC_RANKS]
    for label,name in names:
        a,b=summary[name+'_s50'],summary[name+'_s75']
        lines.append(f'|{label}|{pct(a["overall_physical_sparsity"])}|{pct(a["accuracy"])}|{pct(b["overall_physical_sparsity"])}|{pct(b["accuracy"])}|')
    r=comparisons['r32_vs_full'];g=comparisons['r32_vs_r2'];m=comparisons['r32_vs_mass']
    lines += ['', '## Interpretation', '',
        f'At75%, Gaussian32 gains{pct(g["delta"])} percentage points over Gaussian2 (paired95% CI [{pct(g["paired_ci95"][0])}, {pct(g["paired_ci95"][1])}]), '
        f'recovering{100*recovered:.1f}% of the observed Gaussian2-to-full score gap. '
        'Its actual sparsity is76.04%, versus75.64% for Gaussian2 and75.89% for full. '
        'Gaussian32/full global sparsities are76.54/76.26%; local sparsities75.38/75.40%, so the recovery is not explained by retaining substantially more tiles.', '',
        f'Gaussian32 minus full is{pct(r["delta"])}pp (95% CI [{pct(r["paired_ci95"][0])}, {pct(r["paired_ci95"][1])}]). '
        'The observed scores are close and the difference is inconclusive; this is not a formal equivalence result. '
        f'Gaussian32 exceeds mass-only by{pct(m["delta"])}pp (95% CI [{pct(m["paired_ci95"][0])}, {pct(m["paired_ci95"][1])}]) at0.81pp more physical sparsity.', '',
        'The identical-support diagnostics provide direct evidence of information loss: at75%, Gaussian2’s RMS relative update-norm distortion is0.477, '
        'versus0.128 for Gaussian32. More-than-twofold underestimation affects27.20% versus0.00033% of nonzero supported row/block updates. '
        'Common-threshold physical-tile decision disagreement falls from11.27% to3.68%. '
        'The normalizer, retained history, QKV state and full-dimensional threshold are identical in this comparison; counterfactual sketch decisions do not alter history.', '',
        'Distortion improves across all three prespecified diagnostic seeds: Gaussian2 RMS error0.461–0.511 versus Gaussian32 0.117–0.128. '
        'The reduction appears in both global and local layers: primary-seed Gaussian2→32 RMS errors are0.493→0.129 global and0.453→0.128 local. '
        'These seed checks test diagnostic robustness, not end-to-end accuracy robustness.', '',
        'Generation and operator evidence are consistent for Gaussian32: shared-QKV relative output error0.585 versus0.588 for full, '
        'and token agreement with dense28.27% versus27.36%. Retained mass is nearly identical at40.00% versus39.92%. '
        'Gaussian2 retains38.67% mass yet has only7.24% token agreement and2.50% score. Total retained mass alone does not explain the accuracy gap.', '',
        'An important exception is Gaussian16: it scores16.46%, below Gaussian8’s30.91%, despite lower common-support norm distortion. '
        'The8-minus16 paired difference is14.45pp (95% CI7.08–21.68pp). '
        'Thus scalar distortion is not a sufficient predictor of final accuracy. Rank-dependent projection directions, calibrated decision boundaries, '
        'which individual updates are lost, and adaptive generation histories remain possible contributors; this experiment does not isolate their individual effects. '
        'The deployed-support shared-state operator error is also slightly worse for16 than8 (0.645 versus0.634).', '',
        'At50%, all Gaussian ranks score91.54–94.73% and full scores91.92%; there is no systematic accuracy penalty as rank decreases. '
        'Do not select Gaussian4 as an established winner from this sample. Its apparent advantage includes the VT answer-budget effect described below.', '',
        'Projection is not the whole problem at75%: even full-dimensional centered routing falls from the dense89.23% score to49.78%. '
        'Removing projection loss does not make this aggressive pruning regime accuracy-preserving.', '',
        '## Caveats and reproducibility', '',
        'All10 dense VT outputs exhaust the unchanged30-token official budget and score0. Some sparse runs answer more tersely. '
        'Excluding VT post-hoc gives dense96.67%; at75%, Gaussian2 2.71%, Gaussian32 49.67%, full50.93%, preserving the main conclusion. '
        'At50%, Gaussian4’s primary94.73% vs full91.92% reverses on the non-VT sensitivity:95.96% vs96.25%. '
        'This is a disclosed sensitivity analysis, not the primary benchmark or grounds to alter budgets after seeing scores.', '',
        'Thresholds were independently verified within2pp on26 disjoint calibration questions, then frozen. Final examples/settings match v14; '
        'no kernel, operator or decoding changes were made. Shared diagnostics cover390 states across13 tasks/30 layers at step0, not later adaptive sparse trajectories. '
        'The full retained-support probe is diagnostic-only and uses the full threshold, not the separately calibrated deployed rank threshold.', '',
        'All ranks use the existing Gaussian generator with projection seed1729 for generation; matrices are fixed per layer/native KV head, but not nested across ranks. '
        'One generation seed and previously examined examples limit causal/generalization claims. Reported paired intervals are unadjusted across multiple comparisons. '
        'No hardware speedup is measured or claimed.', '',
        '![Rank versus accuracy and distortion](figures/rank_distortion.png)', '',
        'Full methods, local/global statistics, thresholds, per-task scores and diagnostics: [canonical report](report.md). '
        'Raw results/CSV/JSON and independent regeneration proof are in this directory. '
        'This interpretation is a separately provenance-checked post-audit artifact; it does not overwrite the frozen canonical report.', '']
    (root/'interpretation.md').write_text('\n'.join(lines))
    proof=dict(passed=True,sources=sources,artifacts={n:base.sha((root/n).read_bytes()) for n in ('interpretation.md','interpretation.json')},inference_performed=False)
    base._write(root/'interpretation_proof.json',proof)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=study.ROOT);args=p.parse_args()
    before=build(args.output)
    if build(args.output)!=before:raise ValueError('Interpretation regeneration differs')
    print('Post-audit interpretation independently regenerated; frozen canonical audit unchanged.')
