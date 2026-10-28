"""All sections bad + a user-specified thickness raised a bare ZeroDivisionError."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_space_according_to_index_bad_sections import _df

from DeepSlice.coord_post_processing import spacing_and_indexing as si


def test_all_bad_with_explicit_thickness_gives_clear_error():
    df = _df([1, 2, 3, 4], [100.0, 110.0, 120.0, 130.0], [True] * 4)
    with pytest.raises(ValueError, match="No usable sections"):
        si.space_according_to_index(
            df, section_thickness=10, voxel_size=1, suppress=True
        )


def test_one_good_section_still_works():
    df = _df([1, 2, 3, 4], [100.0, 110.0, 120.0, 130.0], [True, True, True, False])
    out = si.space_according_to_index(
        df, section_thickness=10, voxel_size=1, suppress=True
    )
    assert out["oy"].notna().all()
