"""build_index_report must not materialise range(min, max) of filename numbers."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.gui import state as state_mod
from DeepSlice.gui.state import DeepSliceAppState


def _report(names):
    s = DeepSliceAppState()
    s.image_paths = list(names)
    return s.build_index_report()


def test_small_gaps_are_listed_exactly():
    r = _report(["a_s1.png", "a_s2.png", "a_s5.png"])
    assert r["missing_indices"] == [3, 4]
    assert r["missing_total"] == 2


def test_huge_index_does_not_exhaust_memory():
    r = _report(["a_s1.png", "a_s2.png", "a_s900000000.png"])
    assert len(r["missing_indices"]) == state_mod.MAX_REPORTED_MISSING_INDICES
    assert r["missing_indices"][:3] == [3, 4, 5]
    assert r["missing_total"] == 900000000 - 3


def test_no_gaps_and_duplicates():
    r = _report(["a_s1.png", "b_s1.png", "a_s2.png"])
    assert r["missing_indices"] == [] and r["missing_total"] == 0
    assert r["duplicate_indices"] == [1]
