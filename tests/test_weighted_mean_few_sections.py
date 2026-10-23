"""weighted_mean angle propagation must work for one or two sections.

With <= 2 sections the weights were a list of length n but were then indexed
by brain-centre depth (e.g. 200), raising IndexError, so Normalize Angles
crashed on tiny series.
"""

import pytest

from DeepSlice.coord_post_processing.angle_methods import get_mean_angle


@pytest.mark.parametrize("n", [1, 2])
def test_weighted_mean_with_few_sections(n):
    dv = [1.0, 3.0][:n]
    ml = [2.0, 4.0][:n]
    depths = [200.0, 210.0][:n]
    got_dv, got_ml = get_mean_angle(dv, ml, "weighted_mean", depths, "mouse")
    assert got_dv == pytest.approx(sum(dv) / n)
    assert got_ml == pytest.approx(sum(ml) / n)


def test_weighted_mean_clamps_out_of_range_depths():
    got_dv, _ = get_mean_angle([1.0, 3.0], [0.0, 0.0], "weighted_mean", [-50.0, 99999.0], "mouse")
    assert got_dv == pytest.approx(2.0)
