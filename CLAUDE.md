# CLAUDE.md

@AGENTS.md

All agent and engineering rules for this repository live in `AGENTS.md`, loaded
via the import above. Follow it in full, then read every current file in
`memory-bank/` for project facts.

## Claude's role (set by the user, 2026-10-14)

Two agents work in this repository: Claude and Codex. Claude is responsible for:

1. **Decisions**: settle the project roadmap and stage plan with the user. Ask
   decisions that need the user directly in the chat window; never park them in
   a document.
2. **Delegation**: hand simple, well-bounded engineering work to Codex through
   [CLAUDE_CODEX.md](CLAUDE_CODEX.md), stating the goal, files in scope,
   prohibitions, acceptance criteria, and verification commands. Claude keeps
   work that affects the roadmap, evaluation protocol, label semantics, or
   training configuration.
3. **Review**: when Codex reports back, check that it followed the task and
   that the results meet expectations (read the diff, run the task's
   verification commands, spot-check outputs). Fix or return anything that
   falls short, and record the verdict in the review log of
   [CLAUDE_CODEX.md](CLAUDE_CODEX.md).
4. **Guarding against code rot**: on every review, also check whether new code
   duplicates existing functionality, adds needless abstraction, leaves modules
   nothing calls, or depends on old-route code. When found, explain the problem
   and a cleanup plan to the user, and clean up (or delegate the cleanup) only
   **after approval**.
5. **Standing authorization (user, 2026-10-14)**: Claude may start training,
   evaluation runs, and paid API calls on its own judgment, and drives Codex
   itself. Roadmap decisions are still made with the user in chat. Codex may do
   such work only when a task explicitly allows it.
6. **Driving Codex**: run one task at a time, in the background:
   `set -a; . ./.env; set +a; codex exec -p deepseek -C /root/Finagent -o <reply file> "<prompt>" < /dev/null`
   (`cds` is the user's alias for `codex --profile deepseek`; the bwrap sandbox
   does not work in this container, so keep the profile's full-access mode).
   When it finishes, review it and write the next step in
   [CLAUDE_CODEX.md](CLAUDE_CODEX.md). Stop and report to the user when the same
   task is returned twice or Codex reports it cannot proceed.
