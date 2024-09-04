""""""

from jkepler.transit.transitfit import TransitFit
from jkepler.transit.transitfit import compute_prediction
from jkepler.transit.transitfit import q_to_u
from jkepler.transit.transitfit import a_over_rstar_kep3

from jkepler.tests.read_testdata import read_testdata_transit
from jkepler.tests.read_testdata import read_testdata_transit_koiinfo
from jkepler.tests.read_testdata import get_popt_refs
from jkepler.transit.transit import b_to_cosi


import pytest
import numpy as np
import jax.numpy as jnp
from scipy.stats import binned_statistic


def _phasefold(time, flux, err, t0, period, pcenter=0.5):
    t_fold = (time - t0 + pcenter * period) % period - pcenter * period
    index = np.argsort(t_fold)
    return t_fold[index], flux[index], err[index]


def _binning(x, y, binwidth, statistic="median"):
    bins = np.arange(np.min(x), np.max(x), binwidth)
    vals, _, _ = binned_statistic(x, y, statistic=statistic, bins=bins)
    stds, _, _ = binned_statistic(x, y, statistic="std", bins=bins)
    counts, _, _ = binned_statistic(x, y, statistic="count", bins=bins)
    return 0.5 * (bins[1:] + bins[:-1]), vals, stds / np.sqrt(counts), counts


def _set_transit_test():
    time, flux, error = read_testdata_transit()
    t0, period, b, rstar, rp_over_r, t0err, perr = read_testdata_transit_koiinfo()
    ecc, omega = 0 * t0, 0 * t0
    tpred = np.arange(
        np.min(time), np.max(time), np.median(np.diff(time)) * 0.2
    )  # dense time grid
    popt_refs = get_popt_refs()
    tf = TransitFit(time, exposure_time=29.4 / 1440.0, supersample_factor=10)
    return time, flux, error, tpred, popt_refs, tf


def _bestfit_hmc_ref():
    """Returns the best fit values for the HMC sampler."""
    b = jnp.array([0.37, 1.00, 0.37])
    lna = -8.09
    lnc = -3.39
    lnjitter = -10.99
    lnrp = jnp.array([-2.62, -2.55, -2.31])
    meanflux = 0.00
    period = jnp.array([45.1552, 85.3165, 130.1770])
    q1 = 0.48
    q2 = 0.27
    rstar = 0.91
    t0 = jnp.array([159.1072, 295.3150, 212.0371])
    return b, lna, lnc, lnjitter, lnrp, meanflux, period, q1, q2, rstar, t0


def _check_folded_transits(t, f, e, fmodel, t0s, ps, bin_frac=1e-4):
    for j, (t0, p) in enumerate(zip(t0s, ps)):
        tp, fp, ep = _phasefold(t, f, e, t0, p)
        tbin, fbin, ebin, _ = _binning(tp, fp, p * bin_frac)
        tm, fm, _ = _phasefold(t, fmodel, fmodel * 0, t0, p)
        fmin = np.min(fbin[np.abs(tbin) < 0.5])
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.title("planet %d" % (j + 1))
        plt.ylabel("flux")
        plt.xlabel("time (days)")
        plt.xlim(-0.5, 0.5)
        plt.plot(tbin, fbin, "o", mfc="none", label="folded and binned data")
        plt.plot(tm, fm, ".", markersize=3, zorder=-1000, label="model")
        plt.legend(loc="lower right")
        plt.show()


def test_b_to_cosi():
    b = np.array([0.2, 0.4, 0.6])
    ecc = 0.1
    omega = 0.5
    a_over_r = 10.0
    expected_result = np.array([0.02117056, 0.04234111, 0.06351167])
    result = b_to_cosi(b, ecc, omega, a_over_r)

    assert np.allclose(result, expected_result, atol=1e-8)


def test_q_to_u():
    q1 = np.array([0.2, 0.4, 0.6])
    q2 = np.array([0.1, 0.3, 0.5])
    expected_u1 = np.array([0.08944272, 0.37947332, 0.77459667])
    expected_u2 = np.array([0.35777088, 0.25298221, 0.0])
    u1, u2 = q_to_u(q1, q2)

    assert np.allclose(u1, expected_u1)
    assert np.allclose(u2, expected_u2)


def test_compute_prediction():
    _, flux, error, tpred, popt_refs, tf = _set_transit_test()
    fpred, gppred = compute_prediction(tf, flux, error, popt_refs, tpred)
    valrefs = [-3.0828440805885613, 0.010000640972700758]
    res_f = np.sqrt(np.sum((flux - fpred) ** 2) / len(flux))

    assert pytest.approx(np.sum(fpred)) == valrefs[0]  # this is consistency check
    assert pytest.approx(np.sum(gppred)) == valrefs[1]  # this is consistency check
    assert (
        res_f < 0.0004287
    )  # check if the prediction error is small, res_f = 0.0004286138395168485


def test_compute_flux(fig=False):
    """unit test for compute_flux

    Notes:
        As unit test, we check if the difference between the model and the data is less than 0.00044.
        If fig is True, we plot the folded transit model and data for each planet.
        In this case, we can visually check if the model fits the data.

    Args:
        fig (bool, optional): if True, plot folded transit model and data for each planet. Defaults to False.
    """
    time, flux, error, tpred, popt_refs, tf = _set_transit_test()
    b, lna, lnc, lnjitter, lnrp, meanflux, period, q1, q2, rstar, t0 = (
        _bestfit_hmc_ref()
    )
    u1, u2 = q_to_u(q1, q2)
    a = a_over_rstar_kep3(period, 1.0, rstar)
    zeros = np.zeros_like(period)
    ecc = zeros
    omega = zeros
    rp = jnp.exp(lnrp)
    fluxmodel = tf.compute_flux(t0, period, ecc, omega, b, a, rp, u1, u2)
    if fig:
        _check_folded_transits(time, flux, error, fluxmodel, t0, period, bin_frac=1e-4)
    res = np.sqrt(np.sum((flux - fluxmodel) ** 2) / len(flux))

    assert res < 0.00044


if __name__ == "__main__":
    test_b_to_cosi()
    test_q_to_u()
    test_compute_prediction()
    test_compute_flux(fig=True)
