"""A user-specified section thickness must follow the series' indexing direction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from DeepSlice.coord_post_processing import spacing_and_indexing as si


def _df(oys):
    n = len(oys)
    return pd.DataFrame(
        {
            "Filenames": [f"a_s{i + 1}.png" for i in range(n)],
            "nr": list(range(1, n + 1)),
            "ox": 0.0, "oy": oys, "oz": 0.0,
            "ux": 456.0, "uy": 0.0, "uz": 0.0,
            "vx": 0.0, "vy": 0.0, "vz": 320.0,
        }
    )


def test_caudal_rostro_series_keeps_its_order_with_given_thickness():
    out = si.space_according_to_index(
        _df([130.0, 120.0, 110.0, 100.0]), section_thickness=10, voxel_size=1, suppress=True
    )
    assert np.all(np.diff(out["oy"].to_numpy()) < 0)
    assert np.allclose(np.diff(out["oy"].to_numpy()), -10.0)


def test_rostro_caudal_series_unchanged():
    out = si.space_according_to_index(
        _df([100.0, 110.0, 120.0, 130.0]), section_thickness=10, voxel_size=1, suppress=True
    )
    assert np.allclose(np.diff(out["oy"].to_numpy()), 10.0)


def test_given_thickness_matches_inferred_for_reversed_series():
    a = si.space_according_to_index(
        _df([130.0, 120.0, 110.0, 100.0]), section_thickness=10, voxel_size=1, suppress=True
    )
    b = si.space_according_to_index(
        _df([130.0, 120.0, 110.0, 100.0]), voxel_size=1, suppress=True
    )
    assert np.allclose(a["oy"], b["oy"])


def test_all_bad_sections_raise_a_clear_error():
    with pytest.raises(ValueError, match="every section is flagged as bad"):
        si.ideal_spacing(
            np.array([1, 2, 3]), np.array([10.0, 20.0, 30.0]), 10.0, [True, True, True]
        )
