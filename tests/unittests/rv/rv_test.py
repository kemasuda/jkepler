import pytest
from jkepler.tests.read_testdata import read_testdata_rv
from jkepler.rv import rv_unit_amplitude as getrv
from jkepler.kepler.kepler import t0_to_tau
import jax.numpy as jnp
import numpy as np


def _bestfit_value_rv():
    """best fit values for RVs, derived using rv.ipynb in example
    
    Returns:
        float: K, period, ecc, omega, t0, offset
    """
    logK = 1.82
    K = 10**logK
    ecc = 0.02
    cosw = 0.07
    sinw = 1.23
    t0 = 45.69
    omega = jnp.arctan2(sinw, cosw)
    period = 59.94
    offset = 1.83
    return K, period, ecc, omega, t0,  offset


def _plot_rv(t, y, pred):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.xlim(0, 100)
    plt.xlabel("time from epoch (days)")
    plt.ylabel("radial velocity (km/s)")
    plt.plot(t, y, ".")
    plt.plot(t, pred)
    plt.show()

def test_t0_to_tau():
    """Test t0_to_tau function
    """
    K, period, ecc, omega, t0, offset = _bestfit_value_rv()
    tau = t0_to_tau(t0, period, ecc, omega)
    
    assert pytest.approx(tau) == 45.16903350976443

def test_getrv(fig=False):
    """Test getrv function

    Args:
        fig (bool, optional): If True show a plot. Defaults to False.
    """
    t, y, yerr, tepoch = read_testdata_rv()
    K, period, ecc, omega, t0, offset = _bestfit_value_rv()
    tau = t0_to_tau(t0, period, ecc, omega)
    pred = K * getrv(t, period, ecc, omega, tau) + offset
    if fig:
        _plot_rv(t, y, pred)
    res = np.sum((y - pred) ** 2)

    assert pytest.approx(res) == 37.17380922375936


if __name__ == "__main__":
    test_t0_to_tau()
    test_getrv(fig=True)

