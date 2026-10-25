"""space_according_to_index with a ``bad_section`` column.

Regression: the numpy bool flags were compared with ``is False`` (always
False for ``np.bool_``), zeroing every weight and raising ZeroDivisionError;
and the flags were read before the frame was sorted by section number.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.coord_post_processing import spacing_and_indexing as si


def _df(nrs, oys, bad=None):
    df = pd.DataFrame(
        {
            "Filenames": [f"a_s{n}.png" for n in nrs],
            "nr": nrs,
            "ox": 0.0, "oy": oys, "oz": 0.0,
            "ux": 456.0, "uy": 0.0, "uz": 0.0,
            "vx": 0.0, "vy": 0.0, "vz": 320.0,
        }
    )
    if bad is not None:
        df["bad_section"] = bad
    return df


def test_numpy_bool_flags_are_honoured_in_weights():
    w = si.calculate_weighted_accuracy(
        [1, 2, 3, 4], [100.0, 110.0, 120.0, 130.0], "mouse",
        np.array([False, False, True, False]),
    )
    assert w[2] == 0.0
    assert all(w[i] > 0 for i in (0, 1, 3))


def test_all_good_flags_do_not_crash():
    out = si.space_according_to_index(
        _df([1, 2, 3, 4], [100.0, 110.0, 120.0, 130.0], [False] * 4),
        section_thickness=10, voxel_size=1, suppress=True,
    )
    assert np.isfinite(out["oy"]).all()


def test_flags_follow_sections_when_input_is_unsorted():
    # Section 3 is the outlier (by nr); rows are shuffled.
    nrs = [3, 1, 4, 2]
    oys = [500.0, 100.0, 130.0, 110.0]
    bad = [True, False, False, False]
    out = si.space_according_to_index(
        _df(nrs, oys, bad), section_thickness=10, voxel_size=1, suppress=True
    )
    out = out.sort_values("nr").reset_index(drop=True)
    # Good sections already lie on a 10-unit grid, so they must stay put.
    assert np.allclose(out.loc[out.nr != 3, "oy"], [100.0, 110.0, 130.0], atol=1e-6)
