"""Spacing with every section flagged bad fails with a clear message."""
import numpy as np
import pytest

from DeepSlice.coord_post_processing.spacing_and_indexing import ideal_spacing


def test_all_bad_sections_raise_clear_error():
    nr = np.array([1, 2, 3])
    depths = np.array([100.0, 110.0, 120.0])
    with pytest.raises(ValueError, match="every section is marked as bad"):
        ideal_spacing(nr, depths, 10.0, bad_sections=[True, True, True], method=None)


def test_some_good_sections_still_work():
    nr = np.array([1, 2, 3])
    depths = np.array([100.0, 110.0, 120.0])
    out = ideal_spacing(nr, depths, 10.0, bad_sections=[False, True, False], method=None)
    assert np.allclose(out, depths)
