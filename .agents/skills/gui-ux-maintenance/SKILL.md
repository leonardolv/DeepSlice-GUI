---
name: gui-ux-maintenance
description: >-
  Use this skill to execute continuous GUI/UX audits, fix UI and backend bugs,
  run headless non-interactive tests, maintain GUI_UX_TASK_LOG.md, and safely
  synchronize validated improvements with GitHub.
---

# GUI & UX Maintenance Skill (DeepSlice-GUI)

This skill provides the standard operational procedure for autonomous agents tasked with maintaining and upgrading GUI applications, hunting bugs, and improving ergonomics.

## Workflow Overview

```
[Read GUI_UX_TASK_LOG.md]
        │
        ├── Has active <!-- RESUME -->? ──> Resume in-progress task
        │
        ├── Backlog has items? ────────────> Claim highest-priority item
        │
        └── Backlog empty? ────────────────> Run diagnostic audit & populate Backlog
                                                    │
                                                    ▼
                                          [Implement & Localize Fix]
                                                    │
                                                    ▼
                                          [Headless Verification]
                                                    │
                                   ┌────────────────┴────────────────┐
                                   ▼                                 ▼
                              [Tests Pass]                      [Tests Fail]
                                   │                                 │
                                   ▼                                 ▼
                       [Update Log to Completed]             [Debug & Fix]
                                   │
                                   ▼
                         [Sync with GitHub]
```

---

## 1. State Tracking (`GUI_UX_TASK_LOG.md`)

The file `GUI_UX_TASK_LOG.md` at repository root is the single source of truth across runs.

### Structure:
```markdown
# GUI & UX Maintenance Log

## In Progress
<!-- Active task claimed with timestamp -->

## Blocked / Needs Review
<!-- Items requiring human credentials, hardware, or external decisions -->

## Completed
<!-- Finished tasks with root cause, changes, and verification output -->

## Backlog
<!-- Prioritized list of identified issues and ergonomics tasks -->
```

### Claiming a Task:
- Check for `<!-- RESUME: ... -->` marker under `## In Progress`.
- If none, select the highest priority item from `## Backlog`, move it to `## In Progress`, and record the start timestamp.
- Focus strictly on **ONE discrete task per scheduled run**.

---

## 2. Priority Hierarchy

When triaging or populating the backlog:
1. **[BUG - Critical]**: Unhandled crashes, frozen main/UI threads, broken signals/slots, worker thread deadlocks, data pipeline exceptions connected to user actions.
2. **[UX - High]**: Usability friction, broken responsiveness, missing progress/loading indicators, unformatted dates/numbers, absent empty states, tiny or misaligned click targets.
3. **[POLISH - Low]**: Visual inconsistencies, theme harmonizations, deprecation warnings, minor layout spacing.

---

## 3. Discovery Audit (When Backlog is Empty)

If `## Backlog` has no actionable items:
1. **Run Automated Tests**:
   ```powershell
$env:QT_QPA_PLATFORM="offscreen"
pytest tests/ -v
```
2. **Run Static Code Linters**:
   - Run `ruff check .` or `flake8` to identify unhandled errors, undefined names, and unused broken imports.
3. **Inspect Runtime Logs & Exception Handling**:
   - Inspect console outputs, recent log files, and signal wiring for uncaught exceptions.
4. **Audit Widget Ergonomics**:
   - Inspect UI layouts for missing tooltips, unclear button states, missing error messages, or lack of progress bars.
5. Populate `## Backlog` with prioritized items.

---

## 4. Implementation Guardrails

- **Localized Scope**: Restrict code edits to the specific widget, layout, or controller responsible. Avoid sprawling refactors unless shared state is fundamentally broken.
- **Theme Integrity**: Match existing stylesheets and layout configs. Never inject mismatched arbitrary inline styles.
- **Non-Interactive Execution**: Never invoke blocking message boxes, file pickers, or GUI event loops in automated tests without headless mocks.

---

## 5. Non-Interactive Verification

Every fix must be validated using headless automated testing:
```powershell
$env:QT_QPA_PLATFORM="offscreen"
pytest tests/ -v
```
- Ensure 0 errors, 0 failures, and clean exit codes.

---

## 6. Blocked Tasks Protocol

If a task requires user-specific credentials, specialized hardware, or ambiguous domain decisions:
1. Move the task to `## Blocked / Needs Review`.
2. Detail the exact requirement needed from the user.
3. Return to `## Backlog` and claim the next highest priority item.

---

## 7. Completion & Git Synchronization

Once verification is 100% clean:
1. Move the task from `## In Progress` to `## Completed` in `GUI_UX_TASK_LOG.md`:
   - Include date, issue summary, solution details, and test output snippet.
2. Check `git status` to ensure only intended changes are present.
3. Commit and sync to GitHub:
   ```powershell
   git add GUI_UX_TASK_LOG.md <modified_files>
   git commit -m "fix(gui): concise summary of fix"
   git push
   ```
