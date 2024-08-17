import pytest
from jkepler.rv import rv_unit_amplitude as getrv
from jkepler.kepler.kepler import t0_to_tau
import pkg_resources
import pandas as pd
import jax.numpy as jnp
import numpy as np


def read_testdata_rv():
    """Read in test data for RVs

    Returns:
        jnp.array: time, rv, rv_err
    """
    test_rvfile = pkg_resources.resource_filename(
        "jkepler", "data/tests/v723mon_s12_rv.csv"
    )
    rvdata = pd.read_csv(test_rvfile)
    t, y, yerr = jnp.array(rvdata.BJD), jnp.array(rvdata.rv), jnp.array(rvdata.rv_err)
    idx = t != 2454073.62965  # bad data
    t, y, yerr = t[idx], y[idx], yerr[idx]
    tepoch = t[0]
    t -= tepoch

    return t, y, yerr


def _bestfit_value_rv():
    """best fit values for RVs, derived using rv.ipynb in example
    
    Returns:
        float: K, period, ecc, omega, tau, offset
    """
    logK = 1.82
    K = 10**logK
    ecc = 0.02
    cosw = 0.07
    sinw = 1.23
    t0 = 45.69
    omega = jnp.arctan2(sinw, cosw)
    period = 59.94
    tau = t0_to_tau(t0, period, ecc, omega)
    offset = 1.83
    return K, period, ecc, omega, tau, offset


def _plot_rv(t, y, pred):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.xlim(0, 100)
    plt.xlabel("time from epoch (days)")
    plt.ylabel("radial velocity (km/s)")
    plt.plot(t, y, ".")
    plt.plot(t, pred)
    plt.show()


def test_getrv(fig=False):
    """Test getrv function

    Args:
        fig (bool, optional): If True show a plot. Defaults to False.
    """
    t, y, yerr = read_testdata_rv()
    K, period, ecc, omega, tau, offset = _bestfit_value_rv()
    pred = K * getrv(t, period, ecc, omega, tau) + offset
    if fig:
        _plot_rv(t, y, pred)
    res = np.sum((y - pred) ** 2)

    assert pytest.approx(res) == 37.17380922375936


if __name__ == "__main__":
    test_getrv(fig=True)
