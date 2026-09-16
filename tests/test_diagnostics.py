"""Tests for DeepSlice.diagnostics: log_issue()'s schema/enrichment, the
monitored() decorator, RULE_CATALOGUE's status bookkeeping, and the logger
propagation that lets DeepSlice.error_logging's RotatingFileHandler pick up
every event without this module needing a persistence path of its own."""
import logging
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice import diagnostics


# ---------------------------------------------------------------------------
# RULE_CATALOGUE bookkeeping
# ---------------------------------------------------------------------------

def test_every_catalogued_rule_currently_known_is_marked_resolved():
    """All twelve DS-* rules in the catalogue are verified fixed in the
    current codebase (see AGENT_TASK_LOG.md's 2026-09-15 entry for the
    per-rule verification). run_static_audit() used to emit every catalogue
    entry regardless of status, which is exactly what made the catalogue
    "actively wrong" once a rule was fixed but left unmarked. A rule newly
    discovered and not yet fixed should NOT set status="resolved" — this
    test intentionally pins today's fully-resolved state, not a permanent
    invariant that every future entry must be resolved."""
    for rule_id, entry in diagnostics.RULE_CATALOGUE.items():
        assert entry.get("status") == "resolved", (
            f"{rule_id} is not marked resolved: {entry}"
        )
        resolved_in = entry.get("resolved_in", "")
        assert isinstance(resolved_in, str) and resolved_in.strip(), (
            f"{rule_id} is marked resolved but has no resolved_in note"
        )


def test_catalogue_entries_have_title_and_file():
    for rule_id, entry in diagnostics.RULE_CATALOGUE.items():
        assert entry.get("title"), rule_id
        assert entry.get("file"), rule_id


# ---------------------------------------------------------------------------
# log_issue()
# ---------------------------------------------------------------------------

def test_log_issue_returns_expected_schema():
    event = diagnostics.log_issue("DS-011", "warning", "shadowed builtin names")
    assert event["rule_id"] == "DS-011"
    assert event["severity"] == "WARNING"
    assert event["title"] == "get_mean_angle() shadows built-in names min/max"
    assert event["description"] == "shadowed builtin names"
    assert event["location"]["file"] == diagnostics.RULE_CATALOGUE["DS-011"]["file"]
    assert event["suggested_fix"] == diagnostics.RULE_CATALOGUE["DS-011"]["suggested_fix"]
    assert event["traceback"] is None
    assert "timestamp" in event


def test_log_issue_normalizes_unknown_severity_to_info():
    event = diagnostics.log_issue("DS-999", "critical", "not a real severity")
    assert event["severity"] == "INFO"


def test_log_issue_is_case_insensitive_on_severity():
    event = diagnostics.log_issue("DS-999", "eRRoR", "mixed case severity")
    assert event["severity"] == "ERROR"


def test_log_issue_unknown_rule_id_falls_back_to_description_as_title():
    event = diagnostics.log_issue("DS-999", "info", "an issue with no catalogue entry")
    assert event["title"] == "an issue with no catalogue entry"
    assert event["location"]["file"] == "unknown"
    assert event["suggested_fix"] == {}


def test_log_issue_records_traceback_when_exception_given():
    try:
        raise ValueError("boom")
    except ValueError as exc:
        event = diagnostics.log_issue("DS-999", "error", "caught boom", exc=exc)
    assert event["traceback"] is not None
    assert "ValueError: boom" in event["traceback"]


def test_log_issue_honors_explicit_location():
    location = {"file": "somewhere.py", "function": "f", "line": 42}
    event = diagnostics.log_issue("DS-011", "info", "custom location", location=location)
    assert event["location"] == location


def test_log_issue_emits_a_log_record(caplog):
    caplog.set_level(logging.WARNING, logger="DeepSlice.diagnostics")
    diagnostics.log_issue("DS-011", "warning", "should be logged")
    messages = [r.message for r in caplog.records if r.name == "DeepSlice.diagnostics"]
    assert any("DS-011" in m and "should be logged" in m for m in messages)


# ---------------------------------------------------------------------------
# monitored()
# ---------------------------------------------------------------------------

def test_monitored_passes_through_return_value_on_success():
    @diagnostics.monitored("DS-999")
    def ok(x):
        return x * 2

    assert ok(21) == 42


def test_monitored_logs_and_reraises_on_exception(caplog):
    caplog.set_level(logging.ERROR, logger="DeepSlice.diagnostics")

    @diagnostics.monitored("DS-999")
    def boom():
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        boom()

    records = [r for r in caplog.records if r.name == "DeepSlice.diagnostics"]
    assert any("DS-999" in r.message for r in records)


def test_monitored_uses_the_given_severity(caplog):
    caplog.set_level(logging.WARNING, logger="DeepSlice.diagnostics")

    @diagnostics.monitored("DS-999", severity="WARNING")
    def boom():
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError):
        boom()

    records = [r for r in caplog.records if r.name == "DeepSlice.diagnostics"]
    assert records and records[-1].levelno == logging.WARNING


# ---------------------------------------------------------------------------
# Persistence: log_issue() has no writer of its own, by design (see the
# module docstring) — it relies on "DeepSlice.diagnostics" propagating up to
# the "DeepSlice" logger that DeepSlice.error_logging attaches a
# RotatingFileHandler to. If that propagation is ever turned off, events
# silently stop reaching the on-disk log with nothing else to catch them.
# ---------------------------------------------------------------------------

def test_diagnostics_logger_propagates_to_the_deepslice_root_logger():
    assert diagnostics._logger.name == "DeepSlice.diagnostics"
    assert diagnostics._logger.propagate is True
    # logging.getLogger("DeepSlice") is a lazy PlaceHolder until something
    # actually requests it by name -- which is exactly what
    # error_logging.configure_error_logging() does at app startup (it is
    # what attaches the RotatingFileHandler this whole design relies on).
    # Forcing that same lookup here reproduces the real hierarchy without
    # needing to run configure_error_logging() itself (which would touch
    # ~/.deepslice/logs).
    deepslice_logger = logging.getLogger("DeepSlice")
    assert diagnostics._logger.parent is deepslice_logger


# ---------------------------------------------------------------------------
# The dead query/dump API was removed rather than wired up (see the module
# docstring for the reasoning) — pin that it stays gone rather than being
# silently reintroduced without a fresh decision.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    ["flush_log", "clear_log", "get_issues_by_severity", "get_trivial_fixes", "run_static_audit", "ISSUES"],
)
def test_dead_query_api_was_removed_not_reintroduced(name):
    assert not hasattr(diagnostics, name)
