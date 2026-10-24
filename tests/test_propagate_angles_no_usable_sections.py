"""propagate_angles must fail clearly when every section is flagged bad.

Previously the empty selection produced a NaN mean ("Mean of empty slice"),
surfacing as an unrelated ValueError/ZeroDivisionError from deep inside the
plane maths.
"""

import pandas as pd
import pytest

from DeepSlice.coord_post_processing.angle_methods import propagate_angles

COLS = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]
ROW = [0, 200, 0, 456, 20, 0, 0, 10, -320]


def _df(bad):
    df = pd.DataFrame([ROW] * len(bad), columns=COLS)
    df["bad_section"] = bad
    return df


@pytest.mark.parametrize("method", ["mean", "weighted_mean"])
def test_all_bad_sections_raise_clear_error(method):
    with pytest.raises(ValueError, match="no usable sections"):
        propagate_angles(_df([True, True]), method, "mouse")


def test_empty_predictions_raise_clear_error():
    with pytest.raises(ValueError, match="no usable sections"):
        propagate_angles(_df([]), "mean", "mouse")


def test_some_good_sections_still_work():
    out = propagate_angles(_df([True, False, False]), "mean", "mouse")
    assert len(out) == 3
