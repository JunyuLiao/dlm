# CLAUDE.md

Read `AGENTS.md` first. It holds the rules, terminology and reading order. Then read `HANDOFF.md`, then
`docs/RESEARCH_CONTEXT.md`, `docs/DECISIONS.md` and `docs/RESULTS_LEDGER.md`. Do not duplicate research context here.

Claude-specific workflow:
- Reply to the user in Chinese. Give times and ETAs in the user's local time (US Central, UTC−5), checking the clock
  first.
- Do not spawn subagents unless the user asks.
- Keep long GPU chains as background shell jobs. On Windows, stopping a task can leave the bash chain orphaned;
  list and kill leftover `bash` processes whose command line contains `v27_`.
- Git: this checkout needs `git -c safe.directory=*`. Never commit `third_party/dinfer/assets/Wechat.JPG` (a local
  modification that is not ours) or Office lock files (`~$*.pptx`). Never force-push.
- After substantial work, update `HANDOFF.md` and the docs as `AGENTS.md` describes, in the same commit as the work.
