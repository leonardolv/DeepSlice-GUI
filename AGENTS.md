# Autonomous Agent Guidelines: GUI & UX Maintenance (DeepSlice-GUI)

These rules apply to all autonomous and scheduled agent runs (Antigravity, Claude, etc.) operating on this repository.

## 1. Testing & Headless Execution
- **Never spawn blocking GUI windows or modal popups** during automated runs.
- Always execute tests with `$env:QT_QPA_PLATFORM="offscreen"` (PowerShell) or `export QT_QPA_PLATFORM=offscreen`.
- Always mock dialog functions (such as `QFileDialog`, `QMessageBox`, `QProgressDialog`).
- Use `pytest tests/ -v` or `pytest -v` to run non-interactive automated tests.

## 2. Design & Ergonomics
- Harmonize with existing stylesheets, layout configs, and color palettes. Avoid ad-hoc, hardcoded styling.
- Keep click targets legible and accessible.
- Provide clear visual feedback (loading spinners, progress indicators, status bars) for long-running operations.

## 3. Continuity Log (`GUI_UX_TASK_LOG.md`)
- The repository root file `GUI_UX_TASK_LOG.md` is the single source of truth across runs.
- Follow the lifecycle: `## Backlog` -> `## In Progress` -> `## Completed` (or `## Blocked / Needs Review`).
- Limit work to **one verified, discrete task per scheduled run**.

## 4. Git & Sync
- Run existing automated tests before committing.
- Ensure all tests pass with 0 errors.
- Sync with GitHub using conventional commits: `git add ...`, `git commit -m "fix(gui): ..."`, `git push`.
