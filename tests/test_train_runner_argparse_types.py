"""argparse type helpers keep the original conversion error as ``__cause__``."""

import argparse

import pytest

from DeepSlice.training.train_runner import _bounded_float, _positive_float


@pytest.mark.parametrize("factory", [lambda: _bounded_float("--x", 0.0, 1.0), lambda: _positive_float("--x")])
def test_non_numeric_input_chains_the_original_error(factory):
    with pytest.raises(argparse.ArgumentTypeError, match="must be a number") as excinfo:
        factory()("abc")
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_range_checks_still_apply():
    assert _bounded_float("--x", 0.0, 1.0)("0.5") == 0.5
    with pytest.raises(argparse.ArgumentTypeError):
        _bounded_float("--x", 0.0, 1.0)("2")
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_float("--x")("0")
