import pytest
from jkepler.kepler import t0_to_tau
from jkepler.kepler import get_ta


def test_t0_to_tau():
    t0 = -2.0
    period = 3.0
    ecc = 0.2
    omega = 0.5
    expected_tau = -2.355786224622348
    assert t0_to_tau(t0, period, ecc, omega) == pytest.approx(expected_tau)


def test_get_ta():
    t = 2.0
    porb = 3.0
    ecc = 0.2
    tau = -2.0
    expected_ta = 2.39756138429192
    assert get_ta(t, porb, ecc, tau) == expected_ta


if __name__ == "__main__":
    test_t0_to_tau()
    test_get_ta()
