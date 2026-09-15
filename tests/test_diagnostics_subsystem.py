"""Pins the fixes for the inert diagnostics subsystem (AGENT_TASK_LOG.md,
2026-09-15 entry: "The diagnostics subsystem is inert, and its rule
catalogue is stale").

Three things were wrong:

1. `RULE_CATALOGUE` listed DS-001/003/006/009/011 as live, current bugs even
   though every one of them was already fixed in code -- verified
   independently against the real source (see the AGENT_TASK_LOG.md entry
   for what was checked at each site) rather than trusted from the prior
   session's note alone.
2. `run_static_audit()` didn't look at `status` at all, so wiring it up
   would have reported six already-fixed issues (five newly-marked plus the
   pre-existing DS-007) as current, active problems.
3. `log_issue`/`get_issues_by_severity`/`clear_log` had no UI caller
   anywhere in the app -- issues accumulated in the module-level `ISSUES`
   list for the life of the process and were never shown to a user. A
   "Diagnostics Log" toolbar button (next to About/Errors, this app having
   no traditional Help menu) now opens a read-only dialog wired to both.
"""

from __future__ import annotations

import pytest

from DeepSlice.diagnostics import (
    ISSUES,
    RULE_CATALOGUE,
    clear_log,
    get_issues_by_severity,
    log_issue,
    run_static_audit,
)


RESOLVED_RULE_IDS = ["DS-001", "DS-003", "DS-006", "DS-009", "DS-011"]


@pytest.fixture(autouse=True)
def _clean_issues():
    """Every test in this file starts and ends with an empty ISSUES list --
    it is module-level global state shared across the whole process."""
    clear_log()
    yield
    clear_log()


class TestTheCatalogueNoLongerMisreportsFixedRulesAsLive:
    @pytest.mark.parametrize("rule_id", RESOLVED_RULE_IDS)
    def test_rule_is_marked_resolved(self, rule_id):
        entry = RULE_CATALOGUE[rule_id]
        assert entry.get("status") == "resolved", (
            f"{rule_id} is fixed in code (see AGENT_TASK_LOG.md) but its "
            "catalogue entry does not carry status: 'resolved'"
        )

    def test_ds007_is_unaffected_by_the_new_markings(self):
        """DS-007 already carried the marker this fix mirrors -- make sure
        it wasn't touched or duplicated."""
        assert RULE_CATALOGUE["DS-007"].get("status") == "resolved"

    def test_rules_with_open_bugs_are_left_alone(self):
        """DS-002/004/005/008/010/012 were not verified as fixed by this
        session and must not have been swept into 'resolved' along with
        the five that were."""
        still_open = ["DS-002", "DS-004", "DS-005", "DS-008", "DS-010", "DS-012"]
        for rule_id in still_open:
            assert RULE_CATALOGUE[rule_id].get("status") != "resolved", (
                f"{rule_id} was marked resolved without independent "
                "verification -- that is exactly the mistake this fix "
                "exists to avoid making in the other direction"
            )


class TestRunStaticAuditExcludesResolvedRules:
    def test_resolved_rule_ids_are_absent_from_the_emitted_events(self):
        emitted = run_static_audit()
        emitted_ids = {event["rule_id"] for event in emitted}
        for rule_id in RESOLVED_RULE_IDS + ["DS-007"]:
            assert rule_id not in emitted_ids

    def test_still_open_rule_ids_are_present(self):
        emitted = run_static_audit()
        emitted_ids = {event["rule_id"] for event in emitted}
        for rule_id in ["DS-002", "DS-004", "DS-005", "DS-008", "DS-010", "DS-012"]:
            assert rule_id in emitted_ids

    def test_resolved_rules_stay_excluded_even_if_issues_already_contains_a_match(self):
        """`ISSUES` already having an entry for a resolved rule id (e.g. a
        real, pre-fix `log_issue("DS-001", ...)` call logged earlier this
        session) must not make the audit report it as active -- the audit
        must filter on RULE_CATALOGUE's own status, not merely avoid
        emitting a duplicate of whatever ISSUES already holds."""
        log_issue("DS-001", "ERROR", "a stale in-memory entry for a fixed rule")
        assert any(event["rule_id"] == "DS-001" for event in ISSUES)

        emitted = run_static_audit()
        emitted_ids = {event["rule_id"] for event in emitted}
        assert "DS-001" not in emitted_ids

        # get_issues_by_severity reads ISSUES directly and is deliberately
        # NOT filtered by rule status (it answers "what was logged", not
        # "what is currently active") -- the pre-existing entry must still
        # be visible there, proving the audit's exclusion is its own
        # filter rather than something that mutated ISSUES.
        assert any(event["rule_id"] == "DS-001" for event in get_issues_by_severity("ERROR"))

    def test_run_static_audit_does_not_mutate_issues_for_resolved_rules(self):
        run_static_audit()
        assert not any(event["rule_id"] in RESOLVED_RULE_IDS for event in ISSUES)
