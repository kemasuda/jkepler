from jkepler.astrometry import xyzv_from_elements
import pytest
import numpy as np
import jax.numpy as jnp
from jkepler.tests.read_testdata import read_testdata_astrometry


def bestfit_astrometry_values():
    """Best fit values for astrometry from astrometry.ipynb"""

    cosL = 0.61
    cosi = -0.82
    cosw = 1.24
    ecc = 0.36
    lna = 5.74
    lnjit = 1.78
    period = 1663.52
    sinL = 1.10
    sinw = -0.34
    tau = 58166.78
    lnode = jnp.arctan2(sinL, cosL)
    omega = jnp.arctan2(sinw, cosw)
    a = jnp.exp(lna)
    return cosL, cosi, cosw, ecc, lna, lnjit, period, sinL, sinw, tau, lnode, omega, a


def _plot_orbit(xobs, yobs, xmodel, ymodel):
    xymax = max(np.max(np.abs(xobs)), np.max(np.abs(yobs))) * 1.1
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.axes().set_aspect("equal")
    plt.xlim(-xymax, xymax)
    plt.ylim(-xymax, xymax)
    plt.xlabel("$\Delta\mathrm{RA}\cos(\mathrm{Dec})$ $(\mathrm{mas})$")
    plt.ylabel("$\Delta\mathrm{Dec}$ $(\mathrm{mas})$")
    plt.plot([0], [0], "*", color="k", markersize=10)
    plt.plot(yobs, xobs, "o", label="data", alpha=0.7)
    plt.plot(ymodel, xmodel, ".", label="model")
    plt.show()


def test_xyzv_from_elements(fig=False):
    """test xyzv_from_elements

    Args:
        fig (bool, optional): If True, plots the orbit figure. Defaults to False.
    """
    mjds, sep, e_sep, pa, e_pa, xobs, yobs = read_testdata_astrometry()
    cosL, cosi, cosw, ecc, lna, lnjit, period, sinL, sinw, tau, lnode, omega, a = (
        bestfit_astrometry_values()
    )
    x, y, _, _ = xyzv_from_elements(
        mjds, period, ecc, jnp.arccos(cosi), omega, lnode, tau
    )
    xmodel = a * x
    ymodel = a * y
    if fig:
        _plot_orbit(xobs, yobs, xmodel, ymodel)
    res = np.sum((xobs - xmodel) ** 2 + (yobs - ymodel) ** 2)

    assert pytest.approx(res) == 3219.656230415603


if __name__ == "__main__":
    test_xyzv_from_elements(fig=True)
