"""Bad-section identifiers are plain substrings of filenames, not regexes."""
import pandas as pd
import pytest

from DeepSlice.coord_post_processing.spacing_and_indexing import set_bad_sections_util


def _df(names):
    return pd.DataFrame({"Filenames": names})


def test_parentheses_are_matched_literally():
    df = set_bad_sections_util(_df(["brain_(1).png", "brain_(2).png"]), ["(1)"])
    assert df["bad_section"].tolist() == [True, False]


def test_dot_does_not_act_as_wildcard():
    df = set_bad_sections_util(_df(["a.1.png", "aX1.png"]), ["a.1"])
    assert df["bad_section"].tolist() == [True, False]


def test_plus_and_brackets_literal():
    df = set_bad_sections_util(_df(["s[1]+x.png", "s1x.png"]), ["s[1]+"])
    assert df["bad_section"].tolist() == [True, False]


def test_still_rejects_ambiguous_identifier():
    with pytest.raises(ValueError):
        set_bad_sections_util(_df(["_s001.png", "_s0012.png"]), ["_s001"])
