"""A full filename identifies its section even if it is a substring of others."""
import pandas as pd
import pytest

from DeepSlice.coord_post_processing.spacing_and_indexing import set_bad_sections_util


def _df(names):
    return pd.DataFrame({"Filenames": names})


def test_exact_filename_wins_over_substring_matches():
    df = set_bad_sections_util(_df(["1.png", "11.png", "21.png"]), ["1.png"])
    assert df["bad_section"].tolist() == [True, False, False]


def test_exact_basename_matches_full_path_entry():
    df = set_bad_sections_util(_df(["/d/1.png", "/d/11.png"]), ["1.png"])
    assert df["bad_section"].tolist() == [True, False]


def test_several_exact_names_flagged_together():
    df = set_bad_sections_util(_df(["1.png", "11.png", "21.png"]), ["1.png", "11.png"])
    assert df["bad_section"].tolist() == [True, True, False]


def test_ambiguous_substring_still_rejected():
    with pytest.raises(ValueError):
        set_bad_sections_util(_df(["1.png", "11.png", "21.png"]), [".png"])
