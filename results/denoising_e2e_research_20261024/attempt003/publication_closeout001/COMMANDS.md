# Final review and publication commands

All commands ran from /home/exouser/ljy/dlm. Exact worker/scorer/readout argument
vectors and launch environments are in ../review002/execution_commands.json
and ../review002/COMMANDS.md. Those inputs are already frozen; do not execute
again into existing attempt directories.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_e2e_research_20261024/attempt003/verify_execution_bindings.py --root results/denoising_e2e_research_20261024/attempt003
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_e2e_research_20261024/attempt003/build_review.py --root results/denoising_e2e_research_20261024/attempt003
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_e2e_research_20261024/attempt003/build_review_v2.py --root results/denoising_e2e_research_20261024/attempt003
```

The original build_review.py failed on an absent dense skipped-tile field and
is preserved with review001/failure.json. v2 derived skipped=eligible-kept and
completed450 rows. It never changed generation or official scoring.

The final focused CPU command and36-test JUnit receipt are linked from
../review002/COMMANDS.md and ../tests/final_cpu001/results.xml.

The active non-V31 launcher was attempt006/launch_long.py. The interrupted
attempt007 and namespace-check failure attempt008 are preserved. Persistent
attempt009/finish.py waited for all8 non-V31 long complete markers, then ran
attempt005/launch_v31.py --suite long and all pinned CPU jobs. Its final
receipt and the parent final_complete.json both passed; no retries of accepted
cells or method tuning occurred.

Source checkpoints are listed in the review. Final result checkpoint command:

```bash
git commit -m 'Record complete 25-question native sparse selector end-to-end screen'
```

Only this campaign and its verified doc additions are staged. Existing edits
in AGENTS.md, HANDOFF.md and docs/DECISIONS.md are retained separately in the
worktree. No push. All earlier frozen attempts and paused campaigns remain intact.
