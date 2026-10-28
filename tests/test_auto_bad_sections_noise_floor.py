import numpy as np
import pandas as pd

from DeepSlice.coord_post_processing.spacing_and_indexing import (
    set_bad_sections_util,
)


def _coronal_series(depths):
    n = len(depths)
    df = pd.DataFrame(
        {"Filenames": [f"s_s{i}.png" for i in range(n)], "nr": np.arange(1, n + 1)}
    )
    df["ox"], df["oz"], df["oy"] = 0.0, 0.0, np.asarray(depths, dtype=float) - 0.0
    df["ux"], df["uy"], df["uz"] = 450.0, 0.0, 0.0
    df["vx"], df["vy"], df["vz"] = 0.0, 0.0, -400.0
    return df


def test_perfectly_linear_series_flags_nothing():
    df = _coronal_series(100 + 10.0 * np.arange(1, 13))
    out = set_bad_sections_util(df, [], auto=True)
    assert not out["bad_section"].any()


def test_real_outlier_is_still_flagged():
    depths = 100 + 10.0 * np.arange(1, 13)
    depths[5] += 80.0
    out = set_bad_sections_util(_coronal_series(depths), [], auto=True)
    assert out["bad_section"].tolist()[5] is True or out["bad_section"].iloc[5]
    assert out["bad_section"].sum() <= 2
