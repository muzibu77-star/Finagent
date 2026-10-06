# AGENTS.md

Authoritative agent-rules entry for this repository. Codex loads this file automatically; Claude Code loads it through `@AGENTS.md` in `CLAUDE.md`. General rules live here; project facts live only in the project's `memory-bank/`. Do not duplicate rules or narrative between them — link to the owning document instead.

## 1. Priority and context

Apply instructions in this order:

1. The user's explicit request.
2. This file and project documentation (`memory-bank/`).
3. Existing interfaces, configuration, tests, and behavior.
4. PEP 8 and other general conventions.

A current user request that explicitly asks for a described change is confirmation for that scoped change only; it does not authorize unrelated, destructive, costly, or long-running actions. State assumptions and resolve uncertainty from the repository first. If materially different interpretations remain, list them; do not choose silently. Propose the simpler safe approach and push back on unnecessary complexity. Stop and ask when unresolved ambiguity would change scope, semantics, data, cost, or authorization. When documentation and implementation disagree, verify current behavior from code and durable artifacts, and report the conflict before choosing a path that changes semantics.

## 2. Required reading

Before proposing or making changes:

1. Read every current file in `memory-bank/` in full for project facts, current state, plans, results, environment, domain-specific rules, and collaboration preferences; treat it as the source of truth. Apply `working_preferences.md` after reading it.
2. Inspect `git status`, relevant diffs, the files in scope, their direct callers, configuration, schemas, and relevant tests.
3. Read `architecture.md` in full before writing code. Update it after a major feature only when architecture, schema, data flow, module boundaries, or active entry points changed.

Reading code and documents needs no repeated permission; verify anything the repository can confirm.

## 3. Collaboration and authorization

- Match the user's language. Lead with a concise conclusion unless detail is requested; present analysis as "current issue → corresponding solution" and avoid background padding, repeated caveats, and report-style prose.
- Do not create report files, checklists, summaries, or other artifacts unless the user explicitly requests that artifact.
- When the user asks a conceptual question or says they do not understand something about agent building, system design, ML, or evaluation, answer it, then append the question and a concise conclusion to `memory-bank/note.md` under the matching topic. `note.md` is a learning note: do not record tool installation, environment setup, or other operational steps there.
- Before material changes, state the affected files/functions, approach, meaningful options, and verification method.
- Preserve unrelated and uncommitted work; inspect the working tree and relevant diffs before overlapping edits.
- Without explicit authorization, do not: commit, push, rewrite history, install or upgrade dependencies, upload data, delete assets, or run repository scripts.
- Without explicit authorization, do not start, stop, or interfere with training, evaluation, inference, extraction, cache warming, distributed jobs, or other costly work.
- When the user runs a long task, provide one complete command, expected outputs, progress and checkpoint locations, resume steps, and acceptance criteria. Wait for the result before dependent work.

## 4. Simple, surgical execution

Write the minimum code that solves the request.

- Add no unrequested features, flexibility, configurability, extension points, or fallbacks. Do not abstract code used once or handle cases impossible under the documented contract.
- Reuse existing interfaces, functions, schemas, configuration, and entry points. Do not change public APIs, side effects, formats, model behavior, data semantics, or runtime cost unless requested.
- Do not mix functional changes with broad formatting or unrelated refactoring. Remove only artifacts made obsolete by your change; report pre-existing dead code instead of deleting it.
- Prefer small, modular changes over monolithic files, but avoid artificial file splitting. If 50 clear lines can replace 200, rewrite it. If a senior engineer would call the design overengineered, simplify.
- Handle one scoped subtask at a time unless the user groups related work. For multi-step work, use a short dependency-ordered plan where each step names its verification check.
- Turn work into verifiable goals: reproduce bugs before fixing them, test invalid inputs before adding validation, and prove refactors preserve behavior.

## 5. Python and PEP 8

PEP 8 governs surface consistency, not behavior. Follow established project style when safe.

- Four-space indentation; `snake_case` for functions/variables/config keys, `PascalCase` for classes, `UPPER_CASE` for constants. Default to 88-character lines while keeping long identifiers, commands, and URLs readable.
- Group imports as standard library, third-party, then project modules; remove unused/duplicate imports and avoid unnecessary wildcards. Add accurate type annotations to public and cross-module interfaces.
- Never use mutable default arguments, bare `except:`, silent failures, stale commented-out code, or debugging prints.
- Use docstrings for contracts and comments for non-obvious reasoning (dimensions, masks, synchronization, recovery). Use project logging; never log secrets, credentials, tokens, or private samples.
- Do not change control flow, dtype, device placement, exception semantics, or numerical results merely to satisfy lint. Use a formatter only when already configured or explicitly requested; do not run repository-wide formatting without authorization.

## 6. Machine-learning semantics

Never silently change model architecture/initialization, preprocessing, sampling, split membership, tensor semantics, loss, masks, labels, effective batch size, precision, distribution, generation contracts, evaluation membership, or checkpoint recovery.

- For tensor changes, confirm shape, dimension meaning, dtype, device, padding, mask polarity, ignore indices, broadcasting, contiguity, size-one behavior, and autograd effects. Do not use unconstrained `.squeeze()`, unsafe `.view()`, `.detach()` to hide graph errors, CPU/NumPy conversion to dodge device issues, or hardcoded `.cuda()` in reusable code.
- For models and tokenizers, verify checkpoint/config/tokenizer/generation compatibility, special-token IDs, embedding size, vocabulary, padding/truncation, context length, load warnings, and missing/unexpected keys. Do not use `strict=False` merely to suppress errors.
- Keep training math explicit: `global_batch = micro_batch × grad_accum × dp_world_size`. Confirm loss normalization, optimizer/scheduler steps, gradient zeroing/clipping, and synchronization. Use `model.train()` for training, `model.eval()` for evaluation, and normally `torch.inference_mode()` for inference; restore training mode afterward.
- Training entry points must show completed/total work, elapsed time, ETA, and checkpoint/log location; long training must be resumable. Never hide failures or improve throughput by skipping backward, detaching tensors, dropping metrics, shortening sequences, reducing data, or changing the optimizer.

## 7. Precision and distribution

- Do not change FP16, BF16, TF32, AMP, GradScaler, quantization, parameter dtype, or optimizer-state dtype without checking hardware support, numerical stability, memory/throughput, operator/loss compatibility, and resume compatibility.
- For DDP, FSDP, DeepSpeed, Accelerate, or multi-node code: do not hardcode rank, local rank, or world size; keep collectives symmetric and logging/saving rank-aware; call sampler `set_epoch()` when required; do not add barriers speculatively, remove synchronization to fix hangs, or silently fall back to one device; consider single-device and two-process behavior for distributed changes.

## 8. Data, reproducibility, and recovery

- Make schemas, field types, cleaning/filtering/deduplication, failure handling, split rules, sampling, tokenization, label construction, and cache invalidation explicit. Never leak validation/test data into training, silently skip parse failures, hide filtering, reuse stale caches, or trust executable external/generated content.
- For data-rule changes, report before/after counts and exclusion reasons. Formal experiments must identify code state, dependencies, hardware, model/tokenizer, dataset, configuration, seed, precision, launch command, checkpoints, and metrics. A fixed seed does not guarantee identical results across versions, platforms, drivers, or hardware.
- A resumable checkpoint should contain applicable model, optimizer, scheduler, scaler, epoch/step, best metric/configuration, RNG, and sampler/dataloader state; loading weights alone is not a full resume. Do not overwrite the only valid checkpoint; keep an unambiguous latest pointer and rank-safe writes.
- Do not optimize by guesswork. Establish throughput, peak memory, batch size, sequence length, precision, loading, and communication baselines; change one main variable at a time and verify correctness.

## 9. Verification and completion

Use the lowest-cost verification proportional to risk:

- documentation: reference checks, diff review, `git diff --check`;
- Python/shell: targeted import or compilation checks, `bash -n`;
- data: schema, counts, media, enum, serialization, and leakage checks;
- model/distributed code: only authorized small forward/backward, save/load, or multi-process smoke tests.

Do not infer training correctness from imports alone. A task is complete only when the requested behavior is implemented, unrelated work is preserved, relevant checks pass, no hidden semantic/cost change exists, and unverified paths or remaining risks are reported.

## 10. memory-bank document ownership

Project facts are split by file, kept permanently, with separate responsibilities and no cross-file duplication. Use the files below that the project actually has; do not create missing ones without need. Project-specific files (for example domain rules or a project brief) are allowed and follow the same ownership rule:

- `architecture.md`: current architecture, data flow, active entry points, runtime boundaries, and durable-asset navigation.
- `progress.md`: current snapshot, recent material changes, and blockers.
- `results.md`: append-only ledger of every experiment, including failed, aborted, invalid, and superseded runs; do not rewrite history.
- `plan.md`: complete project lifecycle, stage status, prerequisites, and acceptance criteria; completed stages remain visible.
- `environment.md`: environment recovery and external-asset boundaries.
- `working_preferences.md`: stable project and user collaboration preferences.
- `note.md`: learning notes — the user's questions during the build and their conclusions, grouped by topic; append-only except for corrections.
