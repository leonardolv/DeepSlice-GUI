# DeepSlice-GUI - AI Development Guidelines

## Project Overview
DeepSlice-GUI is a PyQt/PySide application.

## Autonomous GUI & UX Maintenance & Continuity
- **Continuity Log**: `GUI_UX_TASK_LOG.md` at repository root is the single source of truth across runs (`## In Progress`, `## Blocked / Needs Review`, `## Completed`, `## Backlog`).
- **Testing Standard**: Never spawn blocking GUI dialogs or popups during tests. Run headless non-interactive tests.
- **Workflow & Rules**: Follow `AGENTS.md` and `.agents/skills/gui-ux-maintenance/SKILL.md` for prioritized triage, test verification, and automated GitHub sync protocol.
