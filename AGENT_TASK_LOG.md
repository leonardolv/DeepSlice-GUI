# Agent Task Log — Continuous Improvement
Shared continuity file for automated maintenance runs. Multiple agents may
work this file; always append, never overwrite another agent's entries.

This file is the coordination point for the recurring maintenance task only.
`CHANGELOG.md` remains the repo's own release log — record new maintenance work
here.

## In Progress

_(nothing claimed)_

## Completed

### 2026-08-21 UTC — The quality-gate checkbox controlled nothing, and the scan it did not gate blocked the UI thread
Branch `claude/gallant-brahmagupta-86mvu9` · PR
[#12](https://github.com/leonardolv/DeepSlice-GUI/pull/12) · Status: **done, merged**

Claimed and finished in one pass. Nothing was in progress (In Progress was
empty and no commit since the 2026-08-20 entry below was unreflected here).
Took the top-ranked, already-diagnosed Backlog item: "The 'Enable input
quality gate' checkbox controls nothing, and the scan it does not gate
blocks the UI thread."

**Root cause.** `_run_alignment` (`gui/main_window.py`) called
`self.state.screen_input_quality()` — which fully decodes every input image
via `neural_network.inspect_image_batch` — unconditionally, on the GUI
thread, before the actual prediction's `FunctionWorker` was ever created.
`state.quality_gate_enabled` (the checkbox's backing field) only selected
the resulting warning dialog's title string ("Quality Gate Warning" vs.
"Input Quality Warning") and its default button — it did not skip the scan
or the modal in either state. On a large dataset (the Backlog entry cited
~300 slices) this froze the UI for minutes with no progress indicator and
no way to opt out.

**Solution.**
* `_run_alignment` now branches on `self.state.quality_gate_enabled` before
  doing anything quality-related: when off, it skips straight to
  `_start_prediction_worker()` (the renamed tail of the old method); when
  on, it dispatches `self.state.screen_input_quality()` through a
  `FunctionWorker` (`_run_quality_gate_scan`), disabling the Run button and
  labeling it "Scanning inputs..." for the duration — matching every other
  long-running operation in this window (`_start_auto_fix` is the existing
  precedent for the pattern).
* `_on_quality_gate_finished` re-enables the button, builds the same
  resolution-mismatch / flagged-issue warnings the old inline code did (now
  `_build_quality_gate_warnings`, a plain staticmethod), and only then
  proceeds to prediction — with a single, no-longer-vestigial "Quality Gate
  Warning" dialog (the old title/default-button branch on
  `quality_gate_enabled` is gone, since this path is now only reachable when
  the gate is on).
* `_on_quality_gate_error` reports a failed scan via `_show_logged_error`
  and asks the user whether to proceed without the gate, rather than either
  silently blocking prediction forever or silently ignoring the failure.
* **Fixed in passing, same block:** the Backlog entry also flagged that
  `neural_network.py:404` swallows unreadable images with a bare `continue`
  while `report["total"]` still counts them — so a corrupt/unreadable file
  passed the gate silently and would only fail later, during actual
  inference. `inspect_image_batch` now records `unreadable_paths`/
  `unreadable_count`, and `_build_quality_gate_warnings` surfaces them in
  the same pre-flight warning dialog ("N image(s) could not be read and will
  likely fail during prediction (name.tif, ...)"), so a bad file is caught
  before a multi-minute prediction run rather than after.

**Not done:** scan cancellation (the old synchronous scan had none either,
so this is not a regression) and moving the per-image decode work inside
`inspect_image_batch` onto multiple threads (the whole batch already runs
off the UI thread as a unit, which is what the Backlog item asked for).

**Validation.** New `tests/test_quality_gate_wiring.py` (14 tests) plus 2
new/extended tests in `tests/test_neural_network_utils.py`. **11 of the 16
are red on the pre-fix tree** (verified by `git stash` on the three
production files and re-running): direct behavioral tests on
`inspect_image_batch`/`state.screen_input_quality` for the new
`unreadable_*` fields, plus AST-based structural checks on
`_run_alignment`/`_run_quality_gate_scan`/`_on_quality_gate_finished`/
`_on_quality_gate_error` (driving the real `DeepSliceMainWindow` needs a
fully-constructed `QMainWindow` with app-level `QSettings` state that this
sandbox can't stand up outside the real app entry point — confirmed
directly, and `test_angle_convergence_reaches_the_user.py` already
documents the same limitation) that pin: the scan is no longer called
synchronously inside `_run_alignment`; the gate-disabled branch skips
straight to `_start_prediction_worker` without invoking the scan; the scan
runs inside a `FunctionWorker`; a scan failure is reported rather than
silently proceeding.

Full suite: **204 passed, 6 failed** (up from 194 passed, 6 failed) — the 6
(`test_weight_loader.py` ×5, one `test_spacing_and_indexing.py` assertion)
reproduce identically on the unmodified tree in this environment (a
TensorFlow-version weight-naming mismatch, already documented by the
2026-08-19 20:10 entry below), 0 failures caused by this change.
`ruff check` on the three touched production files: 294 → 295 findings, the
+1 being a second `List[str]`-annotation site in the file's own established
`typing.List` style (not a new class of finding); the two touched/added test
files are ruff-clean except for `test_neural_network_utils.py`'s 4
pre-existing findings, unchanged before/after.

**Environment note:** this sandbox needed `libegl1`/`libgl1-mesa-dri`
(`apt-get install`) for PySide6's `QApplication` to import at all under
`QT_QPA_PLATFORM=offscreen`, on top of the `pandas`/`tensorflow`/
`scikit-image`/`matplotlib`/`PySide6`/`pytest-qt` set the 2026-08-19 20:10
entry already documented.

**CI note.** This repo's `.github/workflows/` has never produced a single
workflow run (`actions_list list_workflow_runs` returns `total_count: 0`
across every workflow, not just this PR) — GitHub Actions is effectively
inactive here, so merging relied on the local validation above rather than
a CI gate.

**PR.** [#12](https://github.com/leonardolv/DeepSlice-GUI/pull/12) — merged.

### 2026-08-20 UTC — `is_dirty` was set before validation in the remaining eight mutators
Branch `claude/gallant-brahmagupta-vjwqhq` · PR: see below · Status: **done**

Picked the standing Backlog item this file already ranked as trivial:
`propagate_angles` was fixed by the 2026-08-19 20:10 run (below), and the
same drift was still present in `run_prediction`, `set_bad_sections`,
`flag_low_confidence_sections`, `interpolate_bad_section_depths`,
`apply_manual_order`, `adjust_angles`, `enforce_index_order` and
`enforce_index_spacing` (`gui/state.py`). Each set `self.is_dirty = True` as
its first statement, so an action that raised on unmet preconditions (no
predictions loaded, no images selected, a mismatched manual-order length)
or that legitimately changed nothing (`flag_low_confidence_sections`
finding no new low-confidence sections, `interpolate_bad_section_depths`
finding no bad sections or no gap short enough to interpolate) still marked
the session unsaved — eroding trust in the unsaved-changes prompt, which is
the only thing standing between a user and lost curation work. `undo()`
already states the rule in a comment ("Do not flip is_dirty until we know
the swap can succeed") and it is the pattern every fix here follows: the
flag now moves past every raise and every no-op early return, landing
immediately before the first line that is guaranteed to actually mutate
state. `run_prediction` was the one non-mechanical case — its flag now sets
right after the `model.predict(...)` call succeeds (before
`self.predictions` is reassigned), rather than before the image-count
check or before the settings fields are copied onto `self`, so a
`PartialPredictionAvailable`/other prediction failure no longer marks the
session dirty over settings echoed onto `self` moments earlier with no
completed prediction to show for it.

Added `tests/test_mutators_defer_is_dirty.py` (20 new tests) covering all
eight: a precondition raise doesn't dirty, a real no-op return doesn't
dirty (where the function has one), and a genuine edit does. Modeled on
`test_angle_convergence_reaches_the_user.py`'s existing pattern for the
ninth (already-fixed) mutator, `propagate_angles`.

**Validation:** `pytest tests/test_mutators_defer_is_dirty.py
tests/test_angle_convergence_reaches_the_user.py
tests/test_session_roundtrip.py` — 33/33 pass. Full suite (`pytest tests/
--ignore=tests/gui`, environment has no display): 194 passed, 6
pre-existing failures confirmed unrelated by reproducing them identically
via `git stash` on the branch tip before this change (a TensorFlow-version
weight-naming mismatch in `test_weight_loader.py` and one unrelated
`test_spacing_and_indexing.py` assertion) — none touch `gui/state.py`.
Environment note: this environment's venv had neither the package nor its
heavy deps (`tensorflow`, `h5py`, `requests`, etc.) installed; installed
them fresh from `setup.py`'s `install_requires` to run the suite at all.

### 2026-08-19 20:10-20:40 UTC — "Normalize Angles" reported success when the solver had not converged
Branch `claude/gallant-brahmagupta-v91hcu` · PR
[#10](https://github.com/leonardolv/DeepSlice-GUI/pull/10) · Status: **done**

First run against this repo. `AGENT_TASK_LOG.md` did not exist and was created
per the task template; the Backlog below is seeded from a survey of ~17k LOC
(`DeepSlice/gui/main_window.py` alone is 7375 lines), `state.py`,
`neural_network.py`, `main.py`, `diagnostics.py`, `reporting.py`, the tests and
`CHANGELOG.md`. Took the cheapest high-value item off that list in the same
pass; the rest are ranked in the Backlog below.

**Root cause — a return value dropped twice.** `DSModel.propagate_angles`
(`main.py:361-405`) is documented to return `True` only on convergence, and
does the right thing: after six non-converging iterations it logs `DS-008`
with structured diagnostics, writes "using best available estimate", and
returns `False`. Both layers above it threw that away.
`DeepSliceAppState.propagate_angles` (`gui/state.py:1108`) called
`model.propagate_angles()` as a statement and returned `None`;
`MainWindow._normalize_angles` (`gui/main_window.py:6348`) called *that* as a
statement and ran `_mark_curation_modified()` unconditionally. So the user
clicked the button, saw the curation views refresh, got no warning of any
kind, and shipped half-normalised coordinates.

The reason this is silent rather than obvious is worth stating: **a
non-converging run still produces output.** It writes its best available
estimate into `predictions` exactly as a converged run does, so from every
layer above the model the two are indistinguishable without the flag. The
CHANGELOG's "Added convergence-based angle propagation loop stability check"
landed in the model layer and stopped there.

**Fix.** `state.propagate_angles` returns `bool`; `_normalize_angles` binds it
and, when `False`, raises a `warning`-level toast naming what happened and
what to check before exporting. Toast rather than a modal because the
operation *did* apply a result — this is a caveat on a completed action, not a
failure to interrupt, and `_show_toast(..., level="warning")` is the app's
existing idiom for exactly that (`main_window.py:1767` and elsewhere).

**One thing fixed in passing**, because the test for it was already being
written: `state.propagate_angles` set `is_dirty = True` as its *first*
statement, before the `predictions is None` check that raises — so an action
that could not run still marked the session unsaved. The codebase already
knows this rule and states it in a comment on `undo()` (`state.py:799-803`:
*"Do not flip is_dirty until we know the swap can succeed"*); eight further
mutators have drifted from it the same way and are filed to the Backlog rather
than swept in here.

**Validation.** New `tests/test_angle_convergence_reaches_the_user.py`, 9
tests. **6 are red on the pre-fix tree** (verified by stashing only
`DeepSlice/`). The state-layer half drives the real
`DeepSliceAppState.propagate_angles` against a `FakeModel`, so it needs no
weights and no GPU; the window half is an AST read of `_normalize_angles`,
because driving it needs a whole `QMainWindow` — and the AST check is the
precise shape of what regressed, since the *call* was always there and only
its result was discarded. That half locates `main_window.py` through
`importlib.util.find_spec` rather than importing it: the module imports
matplotlib's Qt backend at module scope, which this check does not need.
One test deliberately asserts the warning is inside the branch the flag
selects, not merely somewhere in the handler — the handler already had
`_show_logged_exception` for the raising path, and a looser check passed on
both sides.

Full suite **174 passed, 6 failed** (from 165 passed, 6 failed). The 6 are
`tests/test_weight_loader.py` and reproduce identically on the unmodified tree
in this environment (confirmed by `git stash`) — a TensorFlow-version issue,
unrelated.

**Environment note for the next run:** the suite needs `pandas`, `numpy`,
`tensorflow`(-cpu) and `scikit-image`; without them
`pytest --collect-only` reports 7 collected and 12 collection errors, which
reads like a broken suite and is not one. `matplotlib` is additionally needed
to import `gui/main_window.py` at all.

**One negative result worth keeping**, because it is the expensive thing to
re-derive: an AST pass over every class looking for `self.<name>(...)` with no
matching `def`/assignment, plus a second pass checking every `self.state.X`
against `DeepSliceAppState`'s real members, found **zero real hits** — the only
results were inherited Qt/Keras methods. There is no AttributeError-drift in
this repo, unlike its siblings. Do not spend another run looking for it.

## Backlog

Seeded by the 2026-08-19 20:10 run, ranked by user impact. Each was verified
against the code, not inferred from docs.

- ~~**The "Enable input quality gate" checkbox controls nothing, and the scan
  it does not gate blocks the UI thread.**~~ Done by the 2026-08-21 run — see
  the Completed entry. Took the "medium" option (moved the scan into a
  `FunctionWorker` rather than just honouring the checkbox around an
  otherwise-still-synchronous scan), and also fixed the noted
  `neural_network.py:404` swallowed-unreadable-image gap in the same pass.
  (original entry follows)
- **The "Enable input quality gate" checkbox controls nothing, and the scan it
  does not gate blocks the UI thread.** `gui/main_window.py:4425-4459`:
  `self.state.screen_input_quality()` runs unconditionally, and
  `state.quality_gate_enabled` only selects the dialog's **title string**
  (`"Quality Gate Warning"` vs `"Input Quality Warning"`) and its default
  button. Unchecking it skips neither the scan nor the modal. Worse,
  `screen_input_quality()` → `inspect_image_batch()`
  (`neural_network.py:382-420`) fully decodes **every** image synchronously on
  the GUI thread, *before* the `FunctionWorker` is created at `:4496` — on a
  300-slice TIFF dataset the app is frozen with no progress indicator for
  minutes and the user's only off switch does not work. Small fix to honour
  the checkbox (`if self.state.quality_gate_enabled:` around the block);
  medium if the scan also moves into the worker. Note `neural_network.py:404`
  swallows unreadable images with `continue` while `report["total"]` stays
  `len(image_paths)`, so a corrupt file passes the gate and fails later during
  inference.
- ~~**"Normalize Angles" reports success when the solver did not converge.**~~
  Done by the 2026-08-19 20:10 run — see the Completed entry. The entry's
  "trivial" estimate held. Note the run also found `state.propagate_angles`
  setting `is_dirty` before the check that raises, which is the item below.
  (original entry follows)
- **"Normalize Angles" reports success when the solver did not converge.**
  `main.py:361-405`'s `propagate_angles` is documented to return `True` only
  on convergence and correctly returns `False` with a `DS-008` log after 6
  non-converging iterations — but `gui/state.py:1108-1116` drops the return
  value, and `gui/main_window.py:6348-6362` calls it inside a `try`, ignores
  the result, and runs `_mark_curation_modified()` unconditionally. The user
  gets no warning and ships half-normalised coordinates. The CHANGELOG's
  "convergence-based angle propagation loop stability check" landed in the
  model layer and was never plumbed to the UI. **Trivial** — return the bool
  through `state.py` and warn on `False`. This is the cheapest high-value item
  in the list and is the one to take first.
- **The progress bar and cancellation cover pass 1 of up to 12 inference
  passes.** `neural_network/neural_network.py:955-993` attaches
  `PredictionProgressCallback` only when `pass_idx == 0`, but
  `_build_inference_pass_specs` (`:898-930`) yields 12 passes with TTA +
  multi-scale both on (4 flips × 3 scales) plus one per dropout pass. The bar
  therefore fills to 100% after ~8% of the work and then sits there while the
  app looks hung — and the GUI's own ETA (`main_window.py:2394-2408`) *does*
  scale for TTA, so the two indicators contradict each other. `cancel_check`
  reaches the Keras callback on pass 0 only, so for passes 2-12 cancellation
  is honoured once per full pass over the dataset rather than at the "safe
  batch boundary" the button's tooltip promises. Small: attach on every pass
  and offset reported `completed` by `pass_idx * total_images`.
- **A failed session load is swallowed, and then the same file is re-parsed as
  QuickNII on top of half-applied state.** `gui/main_window.py:6976-6994`:
  anything raising after `load_session_dict` (which has already mutated
  `self.state`) is discarded by a bare `except Exception: pass` with no message
  and no logging, and control falls through to the `FunctionWorker` at `:6996`
  that reads the same session file as a QuickNII export. The
  `.deepslice-session.json` branch three lines above does this correctly with
  `_show_logged_exception` and is the shape to copy. Trivial — narrow the
  `except` to `json.JSONDecodeError`/`OSError` around the parse only.
- **Two PDF report sections emit boilerplate, and both are on by default.**
  `gui/reporting.py:144-158` writes a heading literally reading `"Sample Images
  (Placeholder)"` and an angle section saying metrics "are summarized in the
  GUI export panel"; `main_window.py:3418-3421` `setChecked(True)` on both. So
  every report a user hands a collaborator contains a section headed
  "(Placeholder)", while the tooltips promise real content. The angle data
  already exists — `state.summary_metrics()` is computed at `:6683` and passed
  in. Small to render the angle stats for real; trivial to default the boxes
  off until it is done. Decide which; do not leave it as is.
- ~~**`is_dirty = True` is set before validation in the remaining eight
  mutators.**~~ Done by the 2026-08-20 run — see the Completed entry.
- **The drag-and-drop toast counts paths requested, not images added.**
  `gui/main_window.py:775-787` reports `len(dropped_paths)` while
  `_handle_dropped_paths` → `state.add_images` → `set_images`
  (`state.py:399-419`) drops non-files, unsupported extensions and duplicates.
  Dropping one folder of 200 TIFFs says "Added 1 dropped path(s)"; dropping 5
  PDFs says "Added 5" when zero were added. The toast is the only feedback on
  that path. Trivial — snapshot `len(state.image_paths)` before and after and
  report the delta.
- **The diagnostics subsystem is inert, and its rule catalogue is stale enough
  to be actively wrong.** `diagnostics.py:193-268` — `flush_log`, `clear_log`,
  `get_issues_by_severity`, `get_trivial_fixes` and `run_static_audit` have
  **zero** call sites anywhere in `DeepSlice/` or `tests/`; only `log_issue` is
  used (`main.py:388`), so the module-level `ISSUES` list grows for the life of
  the process and is never surfaced. Separately, `RULE_CATALOGUE:27-147` still
  lists DS-001, DS-003, DS-006, DS-009 and DS-011 as live when all five are
  **already fixed in code** (verified individually) and only DS-007 carries a
  `status: "resolved"` marker — and `run_static_audit` does not filter on
  status, so wiring it up as-is would report six fixed bugs as current. This is
  "wire it, replace it, or delete it — but decide", and the decision comes
  before any code.
- **Three smaller dead-code items**, worth folding into whichever pass touches
  their file rather than their own run: `gui/workers.py:34-39`'s
  `FunctionWorker.request_cancel()` has no caller (`_cancel_alignment` uses a
  separate `_prediction_cancel_event`), so every *other* long-running worker
  has a cancel API no UI reaches; and `gui/state.py:699-718`'s GPU-probing
  auto-batch-size branch is unreachable, because `run_prediction` always passes
  a non-`None` `requested_batch_size` (`:846`) and
  `_recommended_inference_batch_size` therefore always returns at `:697`.
