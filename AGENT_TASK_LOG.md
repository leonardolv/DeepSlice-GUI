# Agent Task Log — Continuous Improvement
Shared continuity file for automated maintenance runs. Multiple agents may
work this file; always append, never overwrite another agent's entries.

This file is the coordination point for the recurring maintenance task only.
`CHANGELOG.md` remains the repo's own release log — record new maintenance work
here.

## In Progress

_(nothing claimed)_

## Completed

### 2026-09-22 UTC — `load_quint()` cleared `is_dirty` before the file it was loading had actually been parsed
Branch `claude/exciting-wright-xzjgca` · PR
[#26](https://github.com/leonardolv/DeepSlice-GUI/pull/26) · Status: **done, merged**

**Claimed:** not a pre-existing Backlog entry — every entry in
`AGENT_TASK_LOG.md`'s Backlog is struck through and `list_pull_requests`
returned zero open PRs, so a fresh triage pass was run per this task's own
instructions. Found by re-auditing every `self.is_dirty = ...` assignment
in `gui/state.py` for the exact bug class this log has now fixed three
times (2026-08-19 PR #10, 2026-08-20, 2026-09-18 PR #25) — those three
sweeps covered every mutator that sets `is_dirty = True`; `load_quint` sets
it to `False` and sits in a different part of the file, which is why it
was missed by all three.

**The bug.** `DeepSliceAppState.load_quint` (`gui/state.py`) set
`self.is_dirty = False` as its very first statement, before
`ensure_model()`/`model.load_QUINT(filename)` — both of which can raise for
entirely realistic, GUI-reachable input, not an edge case:
`DSModel.load_QUINT` (`main.py:408-427`) raises
`ValueError("File must be a JSON or XML")` outright for any filename that
isn't `.json`/`.xml`, and `_load_session_file`'s QuickNII fallback
(`main_window.py:7113`) is exactly the path that hands `load_quint` any
file that wasn't recognized as a native DeepSlice session — a `.txt`, a
`.csv`, or any other extension dropped onto "Load Session." A malformed
`.json`/`.xml` (bad QuickNII content) or an `ensure_model()` failure
(species-model weights download) raise the same way. So: a session with
unsaved curation edits (`is_dirty == True`) that then attempts to load an
incompatible file as QuickNII silently had `is_dirty` cleared to `False`
*before* the load failed — the close-confirmation prompt (`closeEvent`)
and the window-title `*`/status-bar `●` indicator (`_update_session_status`)
both went quiet immediately, even though the prior unsaved edits were
still unsaved and the attempted load never completed. A user who then
closed the app believing nothing was unsaved would lose that work with no
warning.

**Fix.** Same rule as the three prior fixes and `undo()`'s own comment
("Do not flip is_dirty until we know the swap can succeed"): `is_dirty =
False` now runs immediately after `model.load_QUINT(filename)` returns
successfully, mirroring the ordering `propagate_angles`/`adjust_angles`/
`enforce_index_order`/`enforce_index_spacing` already use a few lines
below it in the same file (past `ensure_model()` *and* past the call that
can still raise). `clear_partial_prediction_candidate()` stays at the top,
unconditional — pinned as deliberate (matching `run_prediction`'s own
existing, unchanged behaviour) rather than swept up as a second bug.

**Validation.** Extended `tests/test_mutators_defer_is_dirty.py` (32 tests,
up from 28) with a `QuintFakeModel` + `TestLoadQuintDefersIsDirty` class:
a `load_QUINT` failure with a pre-existing `is_dirty = True` stays `True`;
an `ensure_model()` failure the same; a genuine successful load still
clears it to `False`; and the partial-prediction-candidate-clearing
ordering is pinned as intentional. **2 of the 4 new tests fail on the
pre-fix tree** (verified via `git stash push -- DeepSlice/gui/state.py`,
re-running just the new class, then `git stash pop`) — both reproduce the
exact bug: `is_dirty` reads `False` after a failed load that started
`True`.

Full suite (fresh venv, `pip install numpy pandas scikit-image scipy
"tensorflow>=2.13,<2.16" h5py requests protobuf lxml Pillow matplotlib
PySide6 nibabel reportlab pytest pytest-qt coverage ruff`,
`QT_QPA_PLATFORM=offscreen PYTHONPATH=. python -m pytest tests/ -q`): **312
passed, 0 failed** (up from 308 before this session's new tests). `ruff
check DeepSlice/gui/state.py`: 94 findings before and after (verified via
the same stash technique — identical count, no new finding). `ruff check
tests/test_mutators_defer_is_dirty.py`: 1 pre-existing `I001` import-sort
finding, present identically before this session's additions (same as the
2026-09-18 entry already documents for this file) — not introduced by the
new test class.

**PR.** [#26](https://github.com/leonardolv/DeepSlice-GUI/pull/26) — merged.

### 2026-09-18 UTC — Four curation mutators marked the session dirty (and pushed a no-op undo snapshot) before the edit that could still fail
Branch `claude/exciting-wright-hwyngk` · PR
[#25](https://github.com/leonardolv/DeepSlice-GUI/pull/25) · Status: **done, merged**

**Claimed:** not a pre-existing Backlog entry — every Backlog item was
struck through. The one loose thread this run's brief flagged,
`gui/workers.py`'s `FunctionWorker.request_cancel()`, turned out to be
already fully resolved: `grep -rn request_cancel` across the repo shows
only comments/tests describing its **deletion** (the 2026-09-09 run removed
`request_cancel`/`is_cancel_requested` outright alongside the GPU-probing
branch cleanup, per `gui/workers.py:34-39`'s own docstring and
`tests/test_function_worker.py:149`'s `assert not hasattr(worker,
"request_cancel")`). Found by re-auditing `gui/state.py` for the exact bug
class this log has fixed twice before (2026-08-19 PR #10, 2026-08-20) —
`is_dirty`/the undo snapshot flipping before an operation that can still
fail — since both prior fixes assumed "past `ensure_model()` and into the
`DSModel` call" was the safe boundary, which is not the same as "the edit
is guaranteed to succeed."

**The bug.** `DeepSliceAppState.propagate_angles`/`adjust_angles`/
`enforce_index_order`/`enforce_index_spacing` (`gui/state.py`, previously
lines 1096-1150) all followed the same shape: `is_dirty = True` +
`snapshot_predictions()` (pushes an undo entry), *then* `ensure_model()`
and the actual `DSModel.<op>()` call — which can still raise for entirely
realistic, GUI-reachable input, not just the "no predictions loaded"
precondition the 2026-08-20 fix covered:
- `adjust_angles()` → `DSModel.adjust_angles` (`main.py:343-358`) raises
  `ValueError` for a non-finite or out-of-[-90, 90] ML/DV angle. The GUI's
  own spinboxes clamp to that range (`main_window.py:3082-3088`), so this
  specific raise is unreachable *through the shipped UI* today, but it is
  live, uncontained `DeepSliceAppState` API behaviour (tests, scripts, a
  future control without the same clamp).
- `enforce_index_order()` → `spacing_and_indexing.enforce_section_ordering`
  (`spacing_and_indexing.py:160-168`) raises `ValueError` for either a
  predictions table with no `"nr"` column, or exactly one section —
  reachable by loading a QuickNII session that never had section-number
  parsing enabled or that has a single section
  (`gui/state.py`'s `load_quint` puts no floor on section count or columns)
  and then clicking "Enforce Index Order."
- `enforce_index_spacing()` → `spacing_and_indexing.space_according_to_index`
  (`spacing_and_indexing.py:224-227`) raises the same two `ValueError`s
  (single section / missing `"nr"`) — reachable the same way; the GUI's own
  "at least 2 sections" gate at `main_window.py:4357` only guards the
  *prediction* flow, not a loaded QuickNII file.
- `propagate_angles()` → `depth_estimation.calculate_brain_center_depth`
  raises `ValueError("Cannot estimate brain center depth for a plane
  parallel to the Y axis")` for a degenerate/near-degenerate U/V/O vector
  triple — reachable from hand-edited or unusual QuickNII coordinates.

In every case the user got a `_show_logged_exception` dialog (the GUI
handlers already wrap these calls in `try/except`) **and** a phantom
"unsaved changes" flag plus a no-op entry sitting on the undo stack for a
change that never happened — eroding the same unsaved-changes-prompt trust
the 2026-08-20 fix was written to restore, and (for `enforce_index_order`/
`enforce_index_spacing`, both realistically QuickNII-reachable with no
range clamp standing in the way) not merely a latent API gap.

**Fix.** Same rule as the earlier fixes and as `undo()`'s own long-standing
comment ("Do not flip is_dirty until we know the swap can succeed"), pushed
one call further out: `ensure_model()` + the `DSModel.<op>()` call now run
*before* `is_dirty = True`/`snapshot_predictions()`, using the model's own
still-unmodified copy of `self.predictions`; only once that call returns
without raising do the flag flip, the undo entry get pushed (of the correct
pre-edit `self.predictions`, since it hasn't been overwritten yet), and
`self.predictions` get replaced with the model's result. Applied
identically to all four methods; no behavioural change to the
already-correct success path (`propagate_angles`'s non-convergence case
still dirties and snapshots, since the model call itself did not raise —
only its return value signals no-convergence, per the 2026-08-19 fix this
preserves).

**Validation.** Extended `tests/test_mutators_defer_is_dirty.py` (24 tests,
up from 20) with a `RaisingFakeModel` + `TestAModelRaiseAfterEnsureModelNeverDirties`
class covering all four methods: asserts both `is_dirty is False` and
`len(undo_stack) == 0` after the raise. **4 of 4 new tests fail on the
pre-fix tree** (verified via `git stash push -- DeepSlice/gui/state.py`,
confirming `is_dirty` ends up `True`), 20/20 pre-existing tests in that file
unaffected either way.

Full suite (fresh venv, `pip install numpy pandas scikit-image scipy
"tensorflow>=2.13,<2.16" h5py requests protobuf lxml Pillow matplotlib
PySide6 nibabel reportlab pytest pytest-qt coverage ruff`,
`QT_QPA_PLATFORM=offscreen PYTHONPATH=. python -m pytest tests/ -q`): **308
passed, 0 failed** (up from 304 before this session's new tests), including
`test_weight_loader.py`'s TensorFlow-dependent tests (green, no
sandbox-drift issue this run). `ruff check DeepSlice/gui/state.py`: 94
findings before and after this change (identical count/positions outside
the edited region — no new finding introduced, matching this file's
pre-existing, already-tracked style baseline). `ruff check
tests/test_mutators_defer_is_dirty.py`: 1 pre-existing `I001` import-sort
finding, present identically before this session's additions (confirmed via
the same stash technique) — not introduced by the new test class.

**Found but not taken (left for a future run, not re-filed to Backlog since
each is low-value/low-urgency on its own):**
`DeepSlice/coord_post_processing/angle_methods.py:11-41`'s
`calculate_brain_center_coordinate` has zero callers anywhere in
`DeepSlice/` or `tests/` (confirmed via a whole-repo AST reference sweep,
not a single grep) and is not part of the package's public surface
(`DeepSlice/__init__.py`'s `__all__` is only `["DSModel", "launch_gui"]`).
Safe, boundable dead-code deletion whenever someone is next in this file,
but has zero user-facing effect on its own, unlike the fix above.

**PR.** [#25](https://github.com/leonardolv/DeepSlice-GUI/pull/25) — merged.

### 2026-09-16 UTC — A superseded atlas-preview request's stale failure/progress could blank out a newer, already-succeeded preview
Branch `claude/exciting-wright-06xj0c` · PR
[#24](https://github.com/leonardolv/DeepSlice-GUI/pull/24) · Status: **done, merged**

**Claimed:** not a pre-existing Backlog entry — the Backlog is fully
resolved (every entry struck through) and `list_pull_requests` returned
zero open PRs, so nothing was already in flight. This branch's own prior
PR (#23) had already merged into `main` before this run started, so the
branch was reset fresh from `origin/main` per this task's own instructions
before picking new work. Found by auditing every `FunctionWorker` call site
in `gui/main_window.py` for the same "stale worker result" bug class this
log has fixed once already (`_on_atlas_ready`'s own `request_token` check).

**The bug.** `_request_atlas_preview` (fired on every curation-row
selection change — arrow-key navigation, clicking a row, changing the
atlas volume, toggling the blend overlay) mints a fresh
`self._atlas_request_token` per call and spawns a `FunctionWorker` to fetch
that row's atlas slice. `_on_atlas_ready` (the `finished` handler) already
discards a result whose `request_token` no longer matches
`self._atlas_request_token` — the correct behaviour for a request the user
has since scrubbed past. `_on_atlas_progress`/`_on_atlas_error` (the
`progress`/`error` handlers on the exact same worker) had no such check:
`WorkerSignals` is one shared class used by every `FunctionWorker` in the
app, so neither signal carries a token of its own, and both handlers
applied unconditionally. Scrubbing quickly through several rows while an
older row's atlas fetch is still in flight — normal use when reviewing many
sections — means an old, superseded request can report its failure (e.g. a
transient download error) or a late progress tick *after* a newer request
has already rendered successfully. Pre-fix, a stale failure unconditionally
called `atlas_viewer.clear_with_text(...)` and set "Atlas: failed",
discarding whatever the current, correct preview had just shown; a stale
progress tick could overwrite the label with an outdated percentage after
the current request had already finished.

**Fix.** `_request_atlas_preview` now binds each worker's own
`request_token` to its `progress`/`error` signal connections via a small
lambda default-arg capture (the established idiom already used one call
away, in `_track_worker`'s own `error`/`finished` connections) rather than
widening `WorkerSignals`' signature, which is shared by every other
`FunctionWorker` consumer (auto-fix, quality-gate scan, prediction,
QuickNII load) and must not change. `_on_atlas_progress`/`_on_atlas_error`
gained an optional `request_token: Optional[int] = None` parameter and the
same staleness guard `_on_atlas_ready` already applies: a token that
disagrees with the current `self._atlas_request_token` is ignored outright;
`None` (nothing bound it, defensively) behaves exactly as before.

**Validation.** New `tests/test_atlas_preview_staleness.py` (9 tests).
Following the established convention for `main_window.py` methods that
can't be driven directly (no full `QMainWindow` needed — see
`test_load_session_file.py`/`test_drop_event_toast.py`), these call the
unbound `DeepSliceMainWindow._request_atlas_preview`/`_on_atlas_error`/
`_on_atlas_progress` against a lightweight stub, with the real production
`_on_atlas_progress`/`_on_atlas_error` bound onto the stub via
`types.MethodType` so the wiring tests exercise the actual staleness guard
rather than a mock of it, and a synchronous fake `thread_pool` (matching
`test_function_worker.py`'s "drive `FunctionWorker.run()` directly, no real
`QThreadPool` needed" precedent). **6 of the 9 fail on the pre-fix tree**
(verified via `git stash` on just `gui/main_window.py`): the two
`_on_atlas_error`/`_on_atlas_progress` staleness-guard unit tests raise
`TypeError` outright (the old signatures take no `request_token`), and the
end-to-end wiring test reproduces the exact bug — a stale failure/late
progress tick from a superseded request still reaching the label/viewer.

Full suite (fresh venv per this file's documented `pip install numpy
pandas scikit-image scipy "tensorflow>=2.13,<2.16" h5py requests protobuf
lxml Pillow matplotlib PySide6 nibabel reportlab pytest pytest-qt
coverage` workaround, `QT_QPA_PLATFORM=offscreen PYTHONPATH=. python -m
pytest tests/ -q`): **304 passed, 0 failed** (up from 295 passed, 0 failed
before this session's new test file). `ruff check DeepSlice/gui/main_window.py`:
180 → 182 findings (both new ones are the same pre-existing `UP045`
"`Optional[int]` → `int | None`" style finding this file already carries
dozens of, on the two new parameters — no new finding *category*, 0
`ruff`-clean regressions); `ruff check tests/test_atlas_preview_staleness.py`:
clean (`All checks passed!`).

**PR.** [#24](https://github.com/leonardolv/DeepSlice-GUI/pull/24) — merged.



### 2026-09-16 UTC — "Try Auto-Fix" could silently reinstall an incompatible TensorFlow, bypassing the project's own Keras-3 pin
Branch `claude/focused-dirac-01nad7` · PR
[#23](https://github.com/leonardolv/DeepSlice-GUI/pull/23) · Status: **done, merged**
(this repo's GitHub Actions runs never actually fire — confirmed via the
Actions API showing 0 workflow runs ever recorded for either configured
workflow — so merged on local validation, matching this file's own
established precedent, not on a green CI check.)

**Claimed:** not a pre-existing Backlog entry — the Backlog is fully
resolved (verified: every entry struck through, and `list_pull_requests`
against the repo returned zero open PRs, so nothing was already in flight
elsewhere). Found by auditing `DeepSlice/error_auto_fix.py`, the module
behind the "Try Auto-Fix Last Error" menu action
(`gui/main_window.py:1291-1299`), which had **zero** test coverage despite
running real `pip install` commands (`_install_and_verify`) when a user
clicks it.

**The bug.** `MODULE_PACKAGE_MAP["tensorflow"]` hardcoded the install spec
`"tensorflow<3.0"` — independent of, and strictly looser than, `setup.py`'s
actual pin, `tensorflow>=2.13,<2.16` (whose own comment reads: *"Keras 3
(TF 2.16+) changed callback APIs we depend on; pin until tested."*). If a
user's TensorFlow install ever went missing (a corrupted venv, a bad
manual `pip uninstall`, an interrupted install) and they clicked "Try
Auto-Fix", the auto-fixer would run `pip install "tensorflow<3.0"` for
real — which today resolves to a current TF 2.16+/Keras 3 release, the
exact incompatibility class `setup.py`'s pin exists to avoid, and the same
class of bug PR #17 (2026-09-09, same log) fixed
(`initialise_network()` crashing unconditionally against an
unpinned-environment TensorFlow). The app's own self-healing feature would
have been the thing that broke it. `_install_and_verify`'s own
verification step only checks `import tensorflow` succeeds — not that
model construction/prediction still works — so this would not have been
caught by the auto-fixer itself; it would surface later as a fresh,
confusing crash report.

**Fix.** Changed the `tensorflow` entry in `MODULE_PACKAGE_MAP` to
`"tensorflow>=2.13,<2.16"`, matching `setup.py` exactly, with a comment
pointing at both `setup.py`'s pin and the regression test. Every other
entry in the map was audited against `setup.py`'s `install_requires` too:
all of them are `>=`-only (no upper bound) in `setup.py`, so a bare
package name (which `pip install` resolves to "latest compatible") is
already correct for those — `tensorflow` was the one exception, and the
only one that needed a change.

**Validation.** New `tests/test_error_auto_fix.py` (34 tests, the module's
first ever coverage): behavioral tests for `analyze_error`/
`format_analysis`/`try_auto_fix`/`_extract_missing_module`/
`_resolve_install_target` across all five classification categories
(missing dependency with/without a safe mapping, workflow precondition,
filename validation, permission error, unknown), plus end-to-end wiring
tests with `subprocess.run` mocked (so nothing is actually installed) that
assert the exact string handed to `pip install` for a real "No module
named 'tensorflow'" error, a failed install, and an install command that
raises. The regression pin,
`test_tensorflow_install_spec_matches_setup_py_pin`, extracts the real
pinned spec straight out of `setup.py` via regex (not a copy-pasted
literal) so a future change to the pin without a matching update here
fails loudly instead of silently drifting again. **5 of the 34 tests fail
on the pre-fix tree** (verified via `git stash` on just
`DeepSlice/error_auto_fix.py`): the regression pin itself, the
`_resolve_install_target("tensorflow")` behavioral test, the
`analyze_error`/`format_analysis` plan-text assertions, and the
end-to-end pip-install wiring test — the last of which shows the exact
pre-fix command: `pip install tensorflow<3.0`.

Full suite (`QT_QPA_PLATFORM=offscreen PYTHONPATH=. python -m pytest
tests/ -q`, fresh venv per this file's documented `pip install numpy
pandas scikit-image scipy "tensorflow>=2.13,<2.16" h5py requests protobuf
lxml Pillow matplotlib PySide6 nibabel reportlab pytest pytest-qt
coverage` workaround — no `pyproject.toml`/`setup.py` install issues hit
this session): **295 passed, 0 failed** (up from 261 passed, 0 failed
before this session's new test file). `ruff check
DeepSlice/error_auto_fix.py`: 14 pre-existing findings before and after
(confirmed via `git stash`, 0 new); `ruff check
tests/test_error_auto_fix.py`: clean (`All checks passed!`).

**PR.** See repository pull requests for this branch.

### 2026-09-16 UTC — Resolved a duplicate-claim collision on the diagnostics item; merged and closed the competing PR (process note)
Branch `claude/serene-fermat-qjb06m` · PR: see below · Status: **done**

This run started with the Backlog showing only this repo's diagnostics
item as previously claimed/in-progress, and no other open Backlog entries
worth a few hours. Before starting fresh work, it checked open PRs
(`list_pull_requests`) rather than trusting the log alone, and found two
independent, unmerged resolutions of the same item already sitting open:
**#19** ("wire it" — fixed 5 of 12 stale `RULE_CATALOGUE` statuses, added a
"Diagnostics Log" toolbar dialog reading the in-memory `ISSUES` list) and
**#20** ("delete it" — independently re-verified **all twelve** rules
against the live source, found all resolved, and removed the five
never-called functions after showing `log_issue()`'s events already
propagate to the app's real on-disk log via logger inheritance).

Checked out #20 into a worktree and independently ran `pytest
tests/test_diagnostics.py` (19/19 passed) and spot-verified three of its
additional rule claims (DS-004, DS-010, DS-012) directly against the
source before trusting the rest. Merged #20 (more thorough — 12/12 rules
vs. 5/12 — and its deletion decision rests on a demonstrated, tested fact
rather than a preference) and closed #19 with a comment crediting its
"Diagnostics Log" UI idea as a reasonable independent follow-up, since
reviving the functions #20 correctly identified as redundant would be the
wrong way to get it.

**Takeaway (same one recorded in VALIS-GUI's log the same day):** two
concurrent runs against one repo can each build a full, valid resolution
of the same Backlog item before either merges, because a completed run's
own log entry lives only on its own branch until its PR merges. Checking
`list_pull_requests` for the target repo before starting — not just this
file — is now worth doing on every run, everywhere in this task.

### 2026-09-15 — The diagnostics subsystem is inert, and its rule catalogue is stale enough to be actively wrong
Branch `claude/serene-fermat-by8wa1` · PR [#20](https://github.com/leonardolv/DeepSlice-GUI/pull/20) · Status: **done, merged**

Claimed from Backlog. `DeepSlice/diagnostics.py`'s `RULE_CATALOGUE` documents
twelve historical bugs (DS-001..DS-012) an AI agent is meant to be able to
query; only DS-007 carried `status: "resolved"`, and `run_static_audit()`
(had it ever been called) does not filter on status — so it would have
reported eleven fixed bugs as still current.

**Verified all twelve against the live code, individually, before touching
anything** (not trusting the catalogue's own wording): read the exact call
site named by each `file` field and confirmed the described bug is gone.
**All twelve are in fact already fixed** — DS-001 (`gray_scale()` now
reshapes to the image's own `(h, w)`), DS-002
(`_resolve_generator_metadata()` validates the cached width/height lists
against the resolved source paths and recomputes when they disagree),
DS-003 (`Path(filename).name`), DS-004 (`is None`/`is not None`, not a
falsy check), DS-005 (every species-switch message now routes through
`DSModel._log()`; the one remaining `print()` is `_log()`'s own documented
no-callback fallback), DS-006 (`training=False`), DS-008
(`propagate_angles()` returns a bool, logs DS-008 and warns on
non-convergence — this is the fix the 2026-08-19 "Normalize Angles" entry
below made, which this rule had never been updated to reflect), DS-009
(`total_bytes > 0` gate), DS-010 (`df['bad_section'] = False` set
unconditionally before the conditional writes), DS-011 (renamed to
`depth_min`/`depth_max`), DS-012 (both call sites now read
`metadata_loader.get_species_depth_range(species)`). Marked all eleven
`status: "resolved"` with a `resolved_in` note naming what actually fixed
each one.

**The "wire it, replace it, or delete it" decision, on `flush_log`,
`clear_log`, `get_issues_by_severity`, `get_trivial_fixes` and
`run_static_audit`**: deleted, not wired. The reason is concrete, not a
preference — `DeepSlice.error_logging.configure_error_logging()` (called
from `gui/app.py` at startup) attaches a `RotatingFileHandler` to the
`"DeepSlice"` logger, and `diagnostics.py`'s own `_logger =
logging.getLogger("DeepSlice.diagnostics")` is a child of it with
`propagate` left at its default `True` — so every `log_issue()` event
**already** reaches `~/.deepslice/logs/errors.log` today, through the
app's one real, rotating, persistent error-logging path. `flush_log()`'s
separate in-memory `ISSUES` list + JSON dump was a second, never-called,
untested persistence mechanism duplicating one that already works, so
kept it deleted rather than plumbing it into the GUI. `log_issue()`,
`RULE_CATALOGUE` and `monitored()` are kept — real callers exist
(`main.py:388`'s DS-008 log, `neural_network.py`'s `@monitored("DS-006")`),
and they are now the module's whole surface.

**Validation**: `git grep` confirms nothing else in the repo (app or
tests) referenced any of the five removed names. Added
`tests/test_diagnostics.py` (19 tests, all passing) — the module had zero
test coverage before this: pins `log_issue()`'s schema/enrichment/severity
normalization, `monitored()`'s pass-through and log-and-reraise behavior,
that every current `RULE_CATALOGUE` entry is `resolved` with a
`resolved_in` note (so a future genuinely-open rule stands out rather than
blending in), the exact logger-propagation claim the deletion decision
rests on, and that the five removed names stay removed rather than
silently creeping back without a fresh decision. `python -m py_compile` and
`python -m pytest tests/test_diagnostics.py` both clean; a full-suite run
was not possible in this sandbox (`tensorflow` not installed — same
constraint prior runs hit), but `log_issue`/`monitored`'s call signatures
are byte-identical to before, so the two real call sites in
`main.py`/`neural_network.py` are unaffected by construction, not just by
inspection.

**Impact**: an AI agent (or a person) consulting `RULE_CATALOGUE` now sees
an accurate "0 open, 12 resolved" picture instead of 11 false positives;
the module's surface shrinks to only what has a real caller; the
persistence story for real runtime `log_issue()` events (like a genuine
future DS-008 non-convergence) is now correctly documented rather than
silently relying on two undocumented facts (propagation defaults, and
`configure_error_logging()` running before any `log_issue()` call) holding
by accident.

**Future recommendation**: if a new rule is ever added to
`RULE_CATALOGUE` for a bug that is *not* yet fixed, leave it without
`status: "resolved"` — `test_every_catalogued_rule_currently_known_is_marked_resolved`
only pins today's fully-resolved state, not a rule that every entry must
carry that status forever.

### 2026-09-09 UTC — `_recommended_inference_batch_size`'s unreachable GPU-probing branch removed
Branch `claude/dazzling-darwin-7fqguu` · PR [#18](https://github.com/leonardolv/DeepSlice-GUI/pull/18) · Status: **done, PR open (watching CI)**

**Claimed:** the Backlog's "`_recommended_inference_batch_size`'s GPU-probing
branch is still unreachable dead code" item — the one third of the
2026-09-08 "Three smaller dead-code items" entry that run left open, having
fixed `request_cancel()` and the crash it uncovered.

**Verified before touching code:** `run_prediction` (`gui/state.py`) is the
method's only caller anywhere in the package, and its one call site always
passes `requested_batch_size=self.inference_batch_size` — a plain `int`
dataclass field defaulting to `8`, never `None`. So the early-return
validation branch always fired, and everything below it (a
`progress_callback is None` check, then a `try/except`-wrapped
`tensorflow.config.list_physical_devices("GPU")` probe defaulting to a
batch size of 2 or 8) could never run.

**Took the "trivial" option the Backlog named** rather than the "real
auto-detect UI control" alternative — that one needs a product decision
(does emptying the spinbox mean "auto", or does it need its own checkbox?)
this run wasn't in a position to make. Deleted the dead branch, dropped the
unused `progress_callback` parameter (it only ever gated that branch), and
made `requested_batch_size` a required `int` instead of `Optional[int]` —
matching the codebase's own stated preference against defensive handling
for scenarios that can't happen. The one call site (`run_prediction`)
updated to match; `Optional` stays imported (used elsewhere in the file).

**Validation.** New `tests/test_inference_batch_size.py`, 6 tests: the
signature no longer accepts `progress_callback` (confirmed red on the
pre-fix method via `git stash` of just `gui/state.py`, green after), valid
batch sizes pass through unchanged, out-of-range values still raise, the
log message is unchanged, and an end-to-end `run_prediction` call (fake
model, `inference_batch_size=32`) confirms the configured batch size still
reaches `model.predict(batch_size=...)`. Full suite:
`QT_QPA_PLATFORM=offscreen xvfb-run -a python -m pytest tests/ -q` —
**236 passed** (up from 230), same **4 pre-existing failures** this run
believed were an environment-only `tensorflow`/`keras` incompatibility
(`test_weight_loader.py`'s two `Xception`-building tests × 2 species,
`Xception() got an unexpected keyword argument 'name'`), reproduced on a
stash of just this change to (wrongly) confirm it as unrelated. `ruff
check DeepSlice/gui/state.py`: same 7 pre-existing `E402` findings before
and after (0 new).

**Correction, found merging this branch against `main`:** those 4 failures
were a real, unrelated bug (an invalid `name=` kwarg on `Xception(...)`),
not an environment artifact — fixed by a concurrent session's PR #17 (see
the entry above), merged to `main` while this PR was open. Pulling that
merge into this branch and re-running the full suite: **242 passed, 0
failed** (up from 236/4-failed), confirming this branch's own change is
independent of and unaffected by that fix.

**Environment note for a future run:** `pip install -e ".[dev]"` fails in
this sandbox with `AttributeError: 'NoneType' object has no attribute
'get'` inside setuptools' `pyproject.toml`/`setup.py` hybrid-config
handling (`_apply_project_table` → `_long_description`), reproduced both
in-place and in a fresh venv, so it isn't specific to a dirty environment.
Worked around by installing the plain dependency list from `setup.py`'s
`install_requires`/`extras_require` directly (`pip install numpy pandas
scikit-image scipy "tensorflow>=2.13,<2.16" h5py requests protobuf lxml
Pillow matplotlib PySide6 nibabel reportlab pytest pytest-qt coverage`)
into a venv and running pytest with `PYTHONPATH`/cwd at the repo root
instead of an editable install. Also needed `apt-get install libegl1
libegl-mesa0 libxcb-cursor0 libxcb-image0 libxcb-render-util0
libxcb-util1` for `pytest-qt`'s `QtGui` import (`libEGL.so.1` missing) —
`libegl-mesa0` 404'd until `apt-get update` was run first. Not
investigated further since it didn't block this run, but worth fixing
properly (drop the stray `[project]` table's partial metadata, or
declare `dynamic` correctly) if a future run needs `pip install -e` to
just work.

### 2026-09-09 UTC — `initialise_network()` crashed on EVERY prediction run against the officially pinned TensorFlow range
Branch `claude/dazzling-darwin-qcsak4` · PR
[#17](https://github.com/leonardolv/DeepSlice-GUI/pull/17) · Status: **done**

**Claimed:** not a pre-existing Backlog entry. Found while building a fresh
venv to re-check the Backlog's "`test_weight_loader.py`'s two `Xception`-
building tests fail in this sandbox's environment" item, which the two prior
runs (2026-08-22, 2026-09-08) both reproduced but explicitly left
uninvestigated as possibly sandbox-specific ("could be a genuine
incompatibility... or an artifact of this sandbox's specific pip resolution
differing from what `pip install -e .[dev]` resolves in real CI"). It is
neither — see below.

**Root cause.** `d216d79` ("resolve neural network layers by name in weight
loader (DS-007)", 2026-09-07, this same maintenance track's own prior
session under `GUI_UX_TASK_LOG.md`) changed
`Xception(include_top=True, weights=xception_weights)` to
`Xception(include_top=True, weights=xception_weights,
name=XCEPTION_BASE_LAYER_NAME)` in `initialise_network()`
(`DeepSlice/neural_network/neural_network.py`), as part of switching
`load_xception_weights` to resolve layers by name instead of fragile
positional indices. `keras.applications.xception.Xception` is a plain
builder **function**, not a `Layer`/`Model` subclass constructor —
`inspect.signature(Xception)` against a real `pip install
"tensorflow>=2.13,<2.16"` (`setup.py`'s own pin, resolves to
`tensorflow==2.15.1`/`keras==2.15.0`) shows exactly
`(include_top=True, weights='imagenet', input_tensor=None,
input_shape=None, pooling=None, classes=1000,
classifier_activation='softmax')` — no `name`, no `**kwargs`. Calling it
with `name=...` raises `TypeError: Xception() got an unexpected keyword
argument 'name'` **unconditionally, on every call, for every species** —
verified directly (`python -c "from tensorflow.keras.applications.xception
import Xception; Xception(weights=None, name='xception')"` raises the exact
error). `initialise_network()` is the one function both mouse and rat
prediction call to build the model for every single run, so this is not an
edge case: it means the shipped code, run against its own declared
dependency range, cannot complete a single prediction.

This is **not** sandbox/pip-resolution drift, and the two prior sessions'
hedge was reasonable but wrong: `setup.py` pins exactly
`tensorflow>=2.13,<2.16`, this sandbox's `pip install` resolved exactly
into that range, and the failure is a hard Python-level `TypeError` from a
function's real signature, not a numerical/behavioural difference that
could plausibly vary by patch version or resolver quirk.

**Why the introducing session's own tests passed.** Unclear, and not fully
resolvable after the fact — `GUI_UX_TASK_LOG.md`'s entry for `d216d79`
claims `pytest tests/test_weight_loader.py -v` passed 7/7 in that session's
own environment immediately after the change. Whatever TensorFlow/Keras
build that session's `pip install` resolved evidently accepted `name=` on
`Xception()` (or the test import silently skipped via
`pytest.importorskip("tensorflow")` without that session noticing) — either
way, it did not match `setup.py`'s own declared range as resolved in this
session's fresh venv, this session's `2026-08-22` predecessor's venv, or
`2026-09-08`'s.

**Fix.** `DeepSlice/neural_network/neural_network.py`: drop the invalid
`name=XCEPTION_BASE_LAYER_NAME` kwarg from the `Xception(...)` call.
Verified this does not undo DS-007's actual point (deterministic
name-based layer resolution): a freshly-built `Xception(...)` model's own
`.name` already defaults to exactly `"xception"` (`== 
XCEPTION_BASE_LAYER_NAME`) on **every** independent call in the same
process — confirmed directly, including that its internal weighted
sub-layers (`block1_conv1`, `block1_conv1_bn`, ...) are consistently named
across repeated calls too (only the unweighted `Input` layer's
auto-numbered name differs, which nothing depends on). So
`load_xception_weights`'s `model.get_layer(XCEPTION_BASE_LAYER_NAME)` keeps
working with zero behavioural change. Added an `assert base_model.name ==
XCEPTION_BASE_LAYER_NAME` right after construction (with a comment
explaining why) so a future Keras release changing that default fails
loudly at the call site instead of silently breaking name-based weight
resolution again the same way this session's regression did silently the
first time.

**Validation.** New `tests/test_weight_loader.py::test_xception_is_never_called_with_a_name_kwarg`
is a TensorFlow-independent AST check (deliberately not gated by
`pytest.importorskip("tensorflow")`, unlike every other test in the file —
gating it would have hidden this exact regression in any environment
without TensorFlow installed, which is how it shipped unnoticed for two
days across two prior maintenance sessions) asserting no `Xception(...)`
call site in the module passes `name=`. Also added
`test_xception_base_layer_name_matches_the_real_default`, pinning the
"default name already matches" assumption the fix depends on, directly
against the real Keras function. **5 of the file's 9 tests fail on the
pre-fix tree** (verified via `git stash` on just the production file): the
pre-existing `test_initialise_network_produces_named_layers`/
`test_forward_pass_produces_9_vector` (×2 species each, exactly the 4
failures the 2026-08-22/2026-09-08 sessions already knew about) plus the
new AST test.

`tests/test_weight_loader.py` alone: **9 passed** (was 3 passed / 4 failed
pre-fix). Full suite (fresh venv: `pip install "tensorflow>=2.13,<2.16"
numpy pandas scikit-image scipy h5py requests protobuf lxml Pillow
matplotlib PySide6 pytest pytest-qt reportlab`, `QT_QPA_PLATFORM=offscreen`,
`PYTHONPATH=.`): **236 passed, 0 failed** (up from 234 collected / 230
passed / 4 failed under the same fresh install — the prior sessions' `234
collected` baseline). `ruff check` on both changed files: `neural_network.py`
17 pre-existing findings before and after (unchanged, confirmed via `git
stash`); `test_weight_loader.py` clean (`All checks passed!`).

**PR.** [#17](https://github.com/leonardolv/DeepSlice-GUI/pull/17) (draft).

### 2026-09-08 UTC — Loading a QuickNII/QuINT session or previewing the atlas always crashed
Branch `claude/dazzling-darwin-im7ui8` · PR
[#16](https://github.com/leonardolv/DeepSlice-GUI/pull/16) · Status: **done**

**Claimed:** the Backlog's "Three smaller dead-code items" entry
(`FunctionWorker.request_cancel()` and the unreachable GPU-probing
auto-batch-size branch), plus fixed two already-stale strikethroughs for
items PR #15 had already resolved (see the Backlog section). Investigating
`request_cancel()`'s only real use — `FunctionWorker.run()`'s
`inject_callbacks` auto-injecting `cancel_check=self.is_cancel_requested` —
turned up a much bigger, previously-unreported bug in the same mechanism.

**Root cause.** `FunctionWorker.run()` (`gui/workers.py`) used to add
`progress_callback`/`log_callback`/`cancel_check` to every
`inject_callbacks=True` call unconditionally. That was safe only as long as
every target declared all three — true of `_run_prediction_task` — but
`_atlas_preview_task` (atlas depth preview, `main_window.py:5772`) and
`_load_quint_task` (Load Session's QuickNII/QuINT fallback,
`main_window.py:7087`) both declare only `progress_callback`/`log_callback`
and have no `**kwargs` catch-all. So the blind `cancel_check` injection
raised `TypeError` on **every single call** to either — silently caught by
`run`'s own broad `except Exception` and surfaced to the user as a generic
"Failed to load QuickNII file" / atlas-preview error dialog. Verified
directly (not just read): a standalone reproduction of the exact
kwargs-building logic against a stand-in with `_load_quint_task`'s real
signature raises `TypeError: _load_quint_task() got an unexpected keyword
argument 'cancel_check'` every time. Since `_load_session_file` routes every
`.json`/other file that is *not* the app's own `deepslice_gui_v1` format
through this exact worker, **loading any real QuickNII/QuINT session file
was completely broken** — only the app's own native session format ever
loaded successfully. `tests/test_load_session_file.py` (the file this exact
bug lives one call away from) mocks `FunctionWorker` out entirely, so this
was invisible to the existing suite. Introduced by `134b393` ("Complete
codebase audit and 10-phase remediation"), which added the unconditional
`cancel_check` injection without updating either function's signature.

**Fix.** `FunctionWorker._accepts_kwarg(name)` inspects `self.fn`'s real
signature (accepting either a declared parameter or a `**kwargs`
catch-all) and `run()` now only injects a callback the target actually
declared. Also deleted `request_cancel`/`is_cancel_requested`/
`_cancel_event` (and the `cancel_check` auto-injection itself) rather than
fixing them: a repo-wide grep confirmed `request_cancel()` has zero
callers anywhere in the app — no UI action ever requested a worker-level
cancel for *any* `FunctionWorker`, atlas/quint included — so the mechanism
it fed was permanently a no-op. The one real caller of `cancel_check`,
`_run_prediction_task`, already has its own working cancellation source
(`self._prediction_cancel_event`, wired to the visible Cancel button) and
its `is_cancelled()` already treated a `None` `cancel_check` as "skip that
check" — so removing the injected (always-`False`) one changes no
observable behaviour there, confirmed by `test_quality_gate_wiring.py` and
the rest of the suite staying green.

**Not done.** The third item, `gui/state.py`'s unreachable GPU-probing
auto-batch-size branch (`_recommended_inference_batch_size`'s
`progress_callback is not None` path, dead because `run_prediction`'s one
call site always passes a non-`None` `requested_batch_size`) — left as-is.
Deleting it cleanly means deciding what `Optional[int]` on
`requested_batch_size` is still for, and wiring an "auto-detect" UI control
that passes `None` would be a real feature addition, not a dead-code
cleanup; out of scope for this pass. Filed to the Backlog with this
context so a future run doesn't have to re-derive it.

**A second, independent bug found while validating: any test that
constructs a real `DeepSliceMainWindow` hangs a LATER, unrelated test
forever.** `MainWindow.__init__` arms `QTimer.singleShot(150,
self._show_startup_dialogs)`, which pops a real, blocking `QMessageBox`
(first-run onboarding, or "what's new" on a version bump). Running the
full suite (not just the files this session's own change touches) hit
this directly: `tests/test_pdf_reporting.py::TestMainWindowPdfDefaults::
test_pdf_checkboxes_defaults_in_main_window` constructs a real window and
itself passes — but its 150ms timer is still pending when the test
returns (`win.close()` does not cancel it, and monkeypatching
`QMessageBox.information` inside that test alone would not help either,
since the patch reverts before the deferred call fires). The timer then
fires during the *next* test's `pytest-qt` teardown `app.processEvents()`
call, which is a direct violation of this repo's own CLAUDE.md ("Never
spawn blocking GUI dialogs or popups during tests") and hung the entire
suite indefinitely at whatever alphabetically-next file happened to run
(`tests/test_quality_gate_wiring.py` here). Reproduced identically on the
pre-fix tree (confirmed by `git stash`-ing this session's other changes
and re-running) — completely unrelated to the `FunctionWorker` fix above,
just found while trying to get a clean full-suite baseline. Fixed by
patching `QTimer.singleShot` itself (not just the dialog call) for the
duration of that one test's window construction, so `_show_startup_dialogs`
is never scheduled in the first place.

**Validation.** New `tests/test_function_worker.py` (7 tests, driving the
real `FunctionWorker.run()` synchronously against stand-ins shaped like the
app's actual `inject_callbacks=True` targets) — **4 of 7 fail on the
pre-fix tree** (verified by `git stash`-ing just the two source files and
re-running): the exact `_load_quint_task`-shaped `TypeError`, the
`cancel_check`-still-accepted-but-now-`None` case, and the
`request_cancel`/`is_cancel_requested` removal. Existing
`test_load_session_file.py`/`test_quality_gate_wiring.py` unaffected (they
mock `FunctionWorker` entirely, so they could not have caught this bug and
do not need to change to keep passing).

Full suite (built against a Python 3.11 venv with `tensorflow<2.16` + the
GUI/PDF/atlas extras installed, `QT_QPA_PLATFORM=offscreen`,
`PYTHONPATH=.`, `pytest-timeout` for diagnosing the hang above; this
sandbox's network handled `pip install tensorflow` fine, unlike some other
repos in this account's fleet whose full lockfiles pull in CUDA wheels):
**234 collected, 230 passed, 4 failed**. The 4 failures
(`test_weight_loader.py`'s `test_initialise_network_produces_named_layers`/
`test_forward_pass_produces_9_vector`, both species) are **pre-existing and
unrelated** — `Xception(..., name=...)` raises `TypeError: unexpected
keyword argument 'name'` against this environment's resolved
`tensorflow==2.15.1`/`keras==2.15.0`, reproduced identically with this
session's changes fully `git stash`-ed. Not investigated further (out of
scope, and may be specific to this pip resolution rather than the pinned
`tensorflow>=2.13,<2.16` the real CI installs via `pip install -e .[dev]`)
— filed to the Backlog rather than silently left for a future run to
re-discover as a regression.

**PR.** [#16](https://github.com/leonardolv/DeepSlice-GUI/pull/16).

### 2026-08-24 UTC — Swallowed session-load failure re-parsed the same file as QuickNII on half-applied state; drag-and-drop toast miscounted
Branch `claude/gallant-brahmagupta-0wdkdo` · PR [#15](https://github.com/leonardolv/DeepSlice-GUI/pull/15) · Status: **done, merged**

**Claimed:** the Backlog's "A failed session load is swallowed, and then the
same file is re-parsed as QuickNII on top of half-applied state" item and
"The drag-and-drop toast counts paths requested, not images added" item.

**1. Swallowed session-load failure.** `gui/main_window.py`'s
`_load_session_file`, `.json` branch (not the `.deepslice-session.json`
branch, which already did this correctly): a file could declare
`"session_format": "deepslice_gui_v1"` and then fail partway through
`state.load_session_dict(payload)` — which mutates `self.state` field by
field before it can raise — and that exception was caught by a bare
`except Exception: pass` wrapping the *entire* read-parse-apply sequence,
with no message and no logging. Control then fell through unconditionally to
the `FunctionWorker` a few lines below, which re-opens the *same file* and
parses it as a QuickNII export, on top of whatever half-applied state the
failed session load left behind. Fixed by splitting the file read/parse
(which legitimately means "not ours, try QuickNII" on `OSError`/
`JSONDecodeError`) from the `state.load_session_dict(...)` application (which
now reports through `_show_logged_exception` — the same call the
`.deepslice-session.json` branch above it already used — instead of falling
through). A `.json` file with no `deepslice_gui_v1` marker, or one that
isn't valid JSON at all, is unchanged: still falls through to QuickNII, since
neither case ever claimed to be a DeepSlice session.

**2. Drag-and-drop toast miscounted.** `dropEvent` reported
`len(dropped_paths)` — the raw URL count the OS handed over — while
`_handle_dropped_paths` → `state.add_images` → `set_images` filters out
non-files, unsupported extensions and duplicates. A folder of 200 TIFFs said
"Added 1 dropped path(s)" (one folder URL); 5 unsupported files said "Added
5" when zero were added. Fixed by snapshotting `len(state.image_paths)`
before and after `_handle_dropped_paths` and reporting the delta, with a
`level="warning"` toast (existing `ToastOverlay` level, used elsewhere for
errors) when the delta is zero rather than a misleadingly neutral "Added 0".

**Validation.** Both `_load_session_file` and `dropEvent` are methods on
`DeepSliceMainWindow`, a `QMainWindow` subclass with a heavy `__init__`
(builds the full tabbed UI) — nothing in the existing suite instantiates it;
coverage here is deliberately at the state layer
(`tests/test_session_roundtrip.py`'s own docstring says so). Followed the
same convention: new `tests/test_load_session_file.py` (4 tests) and
`tests/test_drop_event_toast.py` (3 tests) call the unbound method against a
lightweight stub object exposing only the attributes/methods each method
actually touches, rather than a real window. Each new test was confirmed to
fail on the pre-fix code and pass after (`git stash` round-trips) — in
particular `test_a_broken_deepslice_session_reports_the_failure_and_does_not_fall_through`
reproduces the exact swallow-and-refallback sequence, and
`test_toast_reports_zero_when_nothing_was_actually_added` reproduces the
"Added 5" / zero-actually-added report from the original entry verbatim.

Full suite (`pip install numpy pandas scikit-image scipy "tensorflow>=2.13,<2.16"
h5py requests protobuf lxml Pillow matplotlib PySide6 pytest pytest-qt`,
`QT_QPA_PLATFORM=offscreen` — `pip install -e ".[dev]"` itself fails on this
Python/setuptools combination with an unrelated `pyproject.toml`/`setup.py`
`long_description` config conflict, worth a separate look): **213 passed, 6
failed** before and after this change, unchanged either way — the 6 are
pre-existing `test_weight_loader.py`/`test_spacing_and_indexing.py` failures
against TensorFlow 2.15.1 in this sandbox, confirmed by running the same two
files against the pre-change tree via `git stash`. Not investigated further:
unrelated files, outside this run's claimed scope.

### 2026-08-22 UTC — The progress bar and cancellation covered only pass 1 of up to 12 inference passes

### 2026-08-22 UTC — The progress bar and cancellation covered only pass 1 of up to 12 inference passes
Branch `claude/gallant-brahmagupta-j61ee7` · PR [#13](https://github.com/leonardolv/DeepSlice-GUI/pull/13) · Status: **done, merged**

**Claimed:** the Backlog's "The progress bar and cancellation cover pass 1 of
up to 12 inference passes" item.

**The bug.** `neural_network.py`'s `_run_inference_passes` iterates
`pass_specs` — up to 12 of them with TTA (4 flips) and multi-scale (3
scales) both on, plus one per section-dropout pass — but only attached a
`PredictionProgressCallback` when `pass_idx == 0`. Two consequences from the
same line: **(1)** the progress bar filled to 100% after the first pass
(~8% of the real work with TTA+multi-scale) and then sat there, unmoving,
while passes 2-12 silently ran — the GUI's own pre-run time estimate
(`_estimate_runtime_seconds`) does scale for TTA/multi-scale, so the two
indicators visibly contradicted each other. **(2)** `cancel_check` is
called from inside the Keras callback on every batch; with no callback for
passes past the first, cancellation was only reachable at the outer loop's
per-pass boundary check — once per *entire pass* over the dataset, not at
the "safe batch boundary" the Cancel button's tooltip promises. Cancelling
during pass 6 of 12 meant waiting for all of pass 6 to finish for nothing.

**The fix.** Attach a `PredictionProgressCallback` on every pass, not just
the first. Its `progress_callback` is now a small per-pass wrapper
(`_scaled_progress_callback`) that offsets the pass-local `completed` value
by `pass_idx * generator.n` and reports the *whole run's* image count
(`total_passes * generator.n`) as the total, so `completed`/`total` describe
progress across all passes rather than resetting/stalling per pass.
Cancellation now falls out of the same change for free, since the
per-batch `_raise_if_cancelled()` inside `PredictionProgressCallback` runs
on every pass once its callback is attached — no separate cancellation
logic was needed. `generator.n` is constant across passes (TTA/multi-scale
only transform pixels, not the underlying image list — confirmed via
`ImageGenerator.n = len(self.paths)`, and `clone_with` never changes
`paths`), so a single offset arithmetic covers every pass without needing
to precompute all passes' generators up front.

**Validation.** Two new tests in `tests/test_neural_network_utils.py`
drive `_run_inference_passes` against a fake Keras-shaped model/generator
(no TF model/GPU needed — just the `on_predict_batch_begin`/`_end`
callback protocol):
`test_run_inference_passes_progress_covers_every_pass_not_just_the_first`
asserts every progress call reports the same, whole-run total and that
`completed` only reaches 100% at the very last batch of the very last pass;
`test_run_inference_passes_cancellation_reaches_every_pass` asserts
cancellation set mid-way through pass 1 (the second pass) is caught at
pass 1's very first batch, and that pass 1 never runs a single batch to
completion once cancelled. **Both fail on the pre-fix tree** (confirmed by
stashing only the source change and re-running: the progress test asserts
`total == 16` and gets `4`; the cancellation test asserts pass 1 was
interrupted after 0 batches and finds it ran all 2 to completion instead —
i.e. pre-fix, cancelling mid-pass-1 silently discards a whole pass of work
before being noticed one pass boundary later). Full
`tests/test_neural_network_utils.py` — 32 passed (30 pre-existing + 2 new).
Broader `pytest tests/` (excluding the three heavy training-pipeline files,
which need real training fixtures unrelated to this change) — 153 passed, 6
failed; all 6 failures reproduced identically on the pre-fix tree in
isolation (Keras layer-naming/h5py fixture issues in `test_weight_loader.py`
and one in `test_spacing_and_indexing.py`, confirmed unrelated to this
change and pre-existing). `ruff check` on both changed files reports only
pre-existing, unrelated findings (`warnings`/`pandas`/`inspect_image_quality`
unused imports that predate this diff, confirmed via `git show HEAD:...`).

**Future recommendation.** The three other items this same Backlog entry's
neighbours describe — the swallowed session-load exception, the
drag-and-drop toast's request-vs-added count, and the two boilerplate PDF
report sections — are all still open and independently small; any one of
them is a reasonable next pick.

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
- ~~**The progress bar and cancellation cover pass 1 of up to 12 inference
  passes.**~~ Done by the 2026-08-22 run — see the Completed entry.
  (original entry follows)
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
- ~~**A failed session load is swallowed, and then the same file is re-parsed
  as QuickNII on top of half-applied state.**~~ Done by the 2026-08-24 run
  (PR #15) — see the Completed entry. Left unstruck here until the
  2026-09-08 run noticed the omission while looking for its next item; a
  future run should not re-discover this as new.
  (original entry follows)
- **A failed session load is swallowed, and then the same file is re-parsed as
  QuickNII on top of half-applied state.** `gui/main_window.py:6976-6994`:
  anything raising after `load_session_dict` (which has already mutated
  `self.state`) is discarded by a bare `except Exception: pass` with no message
  and no logging, and control falls through to the `FunctionWorker` at `:6996`
  that reads the same session file as a QuickNII export. The
  `.deepslice-session.json` branch three lines above does this correctly with
  `_show_logged_exception` and is the shape to copy. Trivial — narrow the
  `except` to `json.JSONDecodeError`/`OSError` around the parse only.
- ~~**Two PDF report sections emit boilerplate, and both are on by default.**~~
  Done by the 2026-09-08 run: Enriched `state.summary_metrics()` to retain `mean_dv`,
  `mean_ml`, `std_dv`, and `std_ml`; rendered real quantitative angle metrics
  in `gui/reporting.py`; defaulted `pdf_include_images` to `False` in
  `gui/main_window.py`; and removed placeholder wording in favor of professional
  GUI inspection notes. Also fixed `self._settings` initialization order bug in `DeepSliceMainWindow.__init__`.
- ~~**`is_dirty = True` is set before validation in the remaining eight
  mutators.**~~ Done by the 2026-08-20 run — see the Completed entry.
- ~~**The drag-and-drop toast counts paths requested, not images added.**~~
  Done by the 2026-08-24 run (PR #15) — see the Completed entry.
  (original entry follows)
- **The drag-and-drop toast counts paths requested, not images added.**
  `gui/main_window.py:775-787` reports `len(dropped_paths)` while
  `_handle_dropped_paths` → `state.add_images` → `set_images`
  (`state.py:399-419`) drops non-files, unsupported extensions and duplicates.
  Dropping one folder of 200 TIFFs says "Added 1 dropped path(s)"; dropping 5
  PDFs says "Added 5" when zero were added. The toast is the only feedback on
  that path. Trivial — snapshot `len(state.image_paths)` before and after and
  report the delta.
- ~~**The diagnostics subsystem is inert, and its rule catalogue is stale
  enough to be actively wrong.**~~ Done by the 2026-09-15 run — see the
  Completed entry. All twelve catalogue rules, not just the five this entry
  named, turned out to already be fixed in code; the five unreachable
  functions were deleted rather than wired, since `error_logging.py`'s
  already-active logger propagation makes them redundant, not merely unused.
  (original entry follows)
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
- ~~**Three smaller dead-code items**~~ Two of the three done by the
  2026-09-08 run — see the Completed entry, which found a real, severe bug
  (`_load_quint_task`/`_atlas_preview_task` crashing on every call) hiding
  behind the `request_cancel()` half of this item. The GPU-probing branch is
  the one still open — struck through only for the two resolved halves;
  restated below.
  (original entry follows)
- **Three smaller dead-code items**, worth folding into whichever pass touches
  their file rather than their own run: `gui/workers.py:34-39`'s
  `FunctionWorker.request_cancel()` has no caller (`_cancel_alignment` uses a
  separate `_prediction_cancel_event`), so every *other* long-running worker
  has a cancel API no UI reaches; and `gui/state.py:699-718`'s GPU-probing
  auto-batch-size branch is unreachable, because `run_prediction` always passes
  a non-`None` `requested_batch_size` (`:846`) and
  `_recommended_inference_batch_size` therefore always returns at `:697`.
- ~~**`_recommended_inference_batch_size`'s GPU-probing branch is still
  unreachable dead code.**~~ Done by the 2026-09-09 run — see the Completed
  entry. Took the "delete the dead branch and drop `Optional`" resolution
  this entry named, not the auto-detect-UI alternative.
  (original entry follows)
- **`_recommended_inference_batch_size`'s GPU-probing branch is still
  unreachable dead code.** The one remaining third of the item above.
  `run_prediction`'s single call site always passes
  `requested_batch_size=self.inference_batch_size`, a plain `int` field
  (default `8`, never `None`) — so the early-return branch always fires and
  the `tensorflow`-import GPU-count probe below it (lines ~700-717) never
  runs. `git log -S` shows no sign of an abandoned caller that used to pass
  `None`. Two honest resolutions, neither attempted yet: delete the dead
  branch and drop `requested_batch_size`'s `Optional`, or add a real
  "auto-detect batch size" UI control that passes `None` — the latter is a
  feature addition, not a cleanup, and needs a design decision (does
  auto-detect need its own checkbox, or does emptying the spinbox mean
  "auto"?) before it's a small fix.
- ~~**`test_weight_loader.py`'s two `Xception`-building tests fail in this
  sandbox's environment.**~~ Done by the 2026-09-09 run — see the Completed
  entry. It was **not** sandbox drift: `Xception()` is a plain builder
  function with a fixed signature on the officially pinned
  `tensorflow>=2.13,<2.16` range and genuinely does not accept `name=` —
  every real install in that range crashes on every prediction run, for
  both species. Fixed by dropping the invalid kwarg (the model's own
  default name already matches what the name-based weight loader looks
  up, so DS-007's actual fix is unaffected).
  (original entry follows)
- **`test_weight_loader.py`'s two `Xception`-building tests fail in this
  sandbox's environment**, `TypeError: Xception() got an unexpected
  keyword argument 'name'` from `neural_network.py:609`'s
  `Xception(include_top=True, weights=xception_weights,
  name=XCEPTION_BASE_LAYER_NAME)`. Found by the 2026-09-08 run while
  chasing a full-suite baseline for an unrelated fix; reproduces on a
  clean checkout with no session changes applied, against
  `tensorflow==2.15.1`/`keras==2.15.0` resolved into a fresh venv by `pip
  install "tensorflow>=2.13,<2.16"` (the exact range `setup.py` pins).
  Not investigated further: could be a genuine incompatibility between
  that pin range and `Xception`'s `name=` kwarg at some patch version
  within it, or an artifact of this sandbox's specific pip resolution
  differing from what `pip install -e .[dev]` resolves in real CI (which
  this session had no way to compare against directly). Worth a real CI
  log check before assuming either way.
