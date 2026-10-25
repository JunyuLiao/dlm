"""Reviewable wording only; preserve unrelated shared-file bodies exactly."""
import json
from pathlib import Path
import re

S=Path(__file__).resolve().parents[1];REPO=S.parents[1]
path=S/'private/shared_own_changes.json';old=json.loads(path.read_text())
date='2026-10-26'
prefix=f'''## {date} HKT — Response replication and stopping-headroom studies complete

Read the [study handoff](results/response_behavior_and_stopping_headroom_20261025/summary.md)
and [verification receipt](results/response_behavior_and_stopping_headroom_20261025/closeout001/verification.json).
Branch: `denoising_trajectory_research`. Starting HEAD: `3a1e6e9c`.
Verified generation checkpoints: R `5ebd3e66`, H `2480aa68`;
R readout commit: `e2f5d47d`.

Track R completed 700 audit/clean cells: 30 AIME and 40 HumanEval questions,
two seeds, five arms. The 24-cell frozen legacy gate and all 200 supplemental
legacy checks pass. On questions new to Response, AIME is 21/40 versus FULL
21/40, current-error 22/40 and sham 21/40; HumanEval is 56/60 versus FULL
58/60. No primary quality/work gate passes. Paired clean latency versus FULL
is 4.059× on AIME and 2.590× on HumanEval. Legacy behavior reproduces exactly;
the earlier AIME advantage does not generalize to confirmation.

Track H qualified all eight fresh-control cells, three directional banks and
ten deleting requests. Forty deleting GQA checks pass. Remaining native calls
can decrease, but neither directional rule beats current-error/sham in replicated
development. No confirmation pulse ran. One local improvement from 15 to 12/13
calls accompanies a complete-request increase from 767 to 2,963/2,945 calls.
All deleting requests score wrong and reach the output cap. No accuracy
preservation, useful global acceleration or clean speedup is established.

All 31 focused CPU tests pass. The studies used 1,520 complete requests,
326,053 explicit model forwards and 5.2836 allocated GPU-worker hours, including
startup, qualification, warmups and cleanup. No GPU process failed. One relay
preflight error and one derived CPU-reporting error were preserved and repaired
without GPU reruns. The H100 is idle. No paused campaign, Stage B/C, production
kernel campaign or push. Next: independent review of saved native pulse/commit/RNG
evidence before designing another selector.

The following pre-existing handoff content and edits remain preserved.

'''
decision=f'''

## {date} — Response confirmation and signed-bias headroom (verified)

Source: `results/response_behavior_and_stopping_headroom_20261025/summary.md`.
R protocol: `e1e54486d484d8a83b2870208c438a32de8b982fdaf1af17b2dec349fd580c93`;
generation checkpoint `5ebd3e66`. All 700 audit/clean cells complete, with 30 AIME
and 40 HumanEval questions, two seeds and five arms. Frozen legacy checks pass
24/24; supplemental legacy checks pass 200/200. New-to-Response AIME is 21/40,
equal to FULL/sham and below current-error 22/40. HumanEval is 56/60, below
FULL/sham 58/60 and current-error 59/60. No primary promotion gate passes.
Paired N ratios versus FULL are 1.1045 and 0.9766; simultaneous intervals cross
one. Clean W ratios are 4.0587 and 2.5901. No speedup or noninferiority claim.

H protocol: `849443b09b3e60934710c781804d4bb52f5211bab0b464c742e0848f84c1ede6`;
generation checkpoint `2480aa68`. Eight fresh-control cells, three identifying
development banks and ten deleting requests qualify. Forty deleting GQA checks
pass, with maximum relative error 0.004264. Neither directional rule passes
development; confirmation has no identifying banks and no pulse was launched.
Local native headroom exists without a score advantage: 15 remaining calls fall
to 12/13, while total calls increase from 767 to 2,963/2,945. Every deleting
reference and continuation scores wrong, so this cannot certify preservation.

A1 reporting corrections preserve its negative results. Exact MarginGuard and
norm-only ValidityReuse stay parked. No Stage B/C, production optimization,
paused-campaign resume or push. Preserve all negative/mixed outcomes, missing
cells, source identities, clustered intervals and physical-work accounting.
Independent review precedes any proposed follow-up.
'''
ledger=f'''

## {date} — Completed Response replication and gated native headroom leaf

- Source: `results/response_behavior_and_stopping_headroom_20261025/summary.md`.
- R: 30 AIME and 40 HumanEval questions, two seeds, five arms, 700 complete
  audit/clean cells. Protocol
  `e1e54486d484d8a83b2870208c438a32de8b982fdaf1af17b2dec349fd580c93`,
  generation `5ebd3e66`. New AIME: Response 21/40, FULL 21/40, current-error
  22/40, sham 21/40. New HumanEval: Response 56/60, FULL 58/60, current-error
  59/60, sham 58/60. No clear quality/work gain; clean latency is worse.
- H: four LongBench questions, eight qualified control cells and three gated
  development intervention cells. Protocol
  `849443b09b3e60934710c781804d4bb52f5211bab0b464c742e0848f84c1ede6`,
  generation `2480aa68`. Current-QKV deleting paths qualify. Local calls can
  fall, but no score advantage or useful global gain confirms. No confirmation
  pulse; zero identifying confirmation banks.
- Completion: both tracks' `final_complete.json` and combined `closeout001/`.
  All 31 focused CPU tests pass; 1,520 complete requests, 326,053 explicit
  model forwards and 5.2836 allocated GPU-worker hours. Exploratory and targeted
  negative/mixed evidence, without an accuracy-noninferiority claim.
'''
new={'HANDOFF.md':dict(mode='prepend',text=prefix),'docs/DECISIONS.md':dict(mode='append',text=decision),'docs/RESULTS_LEDGER.md':dict(mode='append',text=ledger)}
for name,entry in new.items():
    p=REPO/name;text=p.read_text();prior=old[name]['text']
    if entry['mode']=='prepend':assert text.startswith(prior);p.write_text(entry['text']+text[len(prior):])
    else:assert text.endswith(prior);p.write_text(text[:-len(prior)]+entry['text'])
path.write_text(json.dumps(new,indent=2));(S/'private/running_handoff_prefix.txt').write_text(prefix)

# Keep exact tables and source identifiers; make prose spacing readable.
p=S/'summary.md';lines=[]
for line in p.read_text().splitlines():
    if line.startswith('|'):lines.append(line);continue
    parts=line.split('`')
    for i in range(0,len(parts),2):
        t=parts[i]
        t=re.sub(r',(?=\S)',', ',t)
        t=re.sub(r':(?=\S)',': ',t)
        t=re.sub(r'\b(The|All|all|only|from|to|use|passes|include|with|simultaneous|bound|exactly|are|is|and|have|has|complete|including|across|on|totalN|rises)(?=\d)',r'\1 ',t)
        t=re.sub(r'(?<=\d)(?=AIME|HumanEval|HE\b|GLOBAL|overall|calls\b|requests\b|forwards\b|remaining\b|floor\b|GPU-worker|allocated)', ' ',t)
        for a,b in [('not140','not 140'),('All70','All 70'),('The299','The 299'),('The24','The 24'),('supplemental200','supplemental 200'),('and18/179','and 18/179'),('onHE','on HumanEval'),('AIME15/20','AIME 15/20'),('FULL12/20','FULL 12/20'),('is21/40','is 21/40'),('is56/60','is 56/60'),('only3','only 3'),('so0/4confirmatory','so 0/4 confirmatory'),('atlayer29','at layer29'),('include94/42','include 94/42'),('with69','with 69'),('from15remaining','from 15 remaining'),('to12','to 12'),('or13','or 13'),('andT4546','and T 4546'),('On672464','On 672464'),('reduces20','reduces 20'),('→17remaining','→17 remaining'),('whileN','while N'),('retains20','retains 20'),('and2905','and 2905'),('across19','across 19'),('including19','including 19'),('eitherHdirection','either H direction'),('norm-onlyValidityReuse','norm-only ValidityReuse'),('3-call','3-call'),('individual10%','individual 10%'),('caps8192','caps 8192'),('bound0.03','bound 0.03'),('maxerror','maximum error')]:t=t.replace(a,b)
        parts[i]=t
    lines.append('`'.join(parts))
p.write_text('\n'.join(lines)+'\n')
print('Polished task-owned prose; preserved unrelated shared-file bodies')
