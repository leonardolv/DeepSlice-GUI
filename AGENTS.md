# Autonomous Agent Guidelines: GUI & UX Maintenance (DeepSlice-GUI)

These rules apply to all autonomous and scheduled agent runs (Antigravity, Claude, etc.) operating on this repository.

## 1. Testing & Headless Execution
- **Never spawn blocking GUI windows or modal popups** during automated runs.
- Always execute tests with `$env:QT_QPA_PLATFORM="offscreen"` (PowerShell) or `export QT_QPA_PLATFORM=offscreen`.
- Always mock dialog functions (such as `QFileDialog`, `QMessageBox`, `QProgressDialog`).
- Use `pytest tests/ -v` or `pytest -v` to run non-interactive automated tests.

## 2. Design & Ergonomics
- Reuse the existing stylesheets, layout configs and color palettes instead of hardcoding styles.

## 3. Continuity Log (`GUI_UX_TASK_LOG.md`)
- The repository root file `GUI_UX_TASK_LOG.md` is the single source of truth across runs.
- Follow the lifecycle: `## Backlog` -> `## In Progress` -> `## Completed` (or `## Blocked / Needs Review`).
- Limit work to **one verified, discrete task per scheduled run**.

## 4. Git & Sync
- Run existing automated tests before committing.
- Ensure all tests pass with 0 errors.
- Sync with GitHub using conventional commits: `git add ...`, `git commit -m "fix(gui): ..."`, `git push`.

## Start of every run (Addendum)
1. Read AGENT_TASK_LOG.md (Summary first, then In Progress).
2. Claim a task with a UTC ISO 8601 timestamp under In Progress. Other agents can override this lock if idle > 24h.
3. Do not duplicate another agent's active work.

## Agent roles
- Claude: Hard tasks, but allowed to do easier tasks if invoked directly.
- Gemini / Antigravity: Hard tasks, complex code, architecture, GUI polish.
- Codex: small bug fixes, tests, typos.
- Copilot: in-editor edits and small completions.
- Perplexity: literature and web research.

## Rules
- Fix bugs directly in the code if you are really sure, otherwise leave a note.
- Important work (data analysis, mathematical calculations) needs review by a different agent before merge. Verify every number twice with different methods and report it only if both agree. Never auto-merge these; require a PR/Branch.

## Style
Strictly use hyphens only (no em dashes, no en dashes). Concise.
