""""""

from jkepler.transit.transitfit import TransitFit
from jkepler.transit.transitfit import compute_prediction

from jkepler.tests.read_testdata import read_testdata_transit
from jkepler.tests.read_testdata import read_testdata_transit_koiinfo
from jkepler.tests.read_testdata import get_popt_refs

from pykepler.utils import phasefold, binning

import pytest
import numpy as np


def _set_transit_test():
    time, flux, error = read_testdata_transit()
    t0, period, b, rstar, rp_over_r, t0err, perr = read_testdata_transit_koiinfo()
    ecc, omega = 0 * t0, 0 * t0
    tpred = np.arange(
        np.min(time), np.max(time), np.median(np.diff(time)) * 0.2
    )  # dense time grid
    popt_refs = get_popt_refs()
    tf = TransitFit(time, exposure_time=29.4 / 1440.0, supersample_factor=10)
    return flux, error, tpred, popt_refs, tf


def test_compute_prediction():
    flux, error, tpred, popt_refs, tf = _set_transit_test()
    fpred, gppred = compute_prediction(tf, flux, error, popt_refs, tpred)
    valrefs = [-3.0828440805885613, 0.010000640972700758]
    res_f = np.sqrt(np.sum((flux - fpred) ** 2) / len(flux))

    assert pytest.approx(np.sum(fpred)) == valrefs[0]  # this is consistency check
    assert pytest.approx(np.sum(gppred)) == valrefs[1]  # this is consistency check
    assert (
        res_f < 0.0004287
    )  # check if the prediction error is small, res_f = 0.0004286138395168485

#def check_folded_transits(t, f, e, fmodel, t0s, ps, bin_frac=1e-4):

#check_folded_transits(t, f, e, fpred, popt['t0'], popt['period'])
def test_folded_transits(fig=False):
    flux, error, tpred, popt_refs, tf = _set_transit_test()
    for j, (t0, p) in enumerate(zip(popt_refs["t0"], popt_refs["period"])):
        tp, fp, ep = phasefold(t, f, e, t0, p)
        tbin, fbin, ebin, _ = binning(tp, fp, p * bin_frac)
        tm, fm, _ = phasefold(t, fmodel, fmodel * 0, t0, p)
        fmin = np.min(fbin[np.abs(tbin) < 0.5])

        if fig:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.title("planet %d" % (j + 1))
            plt.ylabel("flux")
            plt.xlabel("time (days)")
            plt.ylim(1.5 * fmin, -0.5 * fmin)
            plt.xlim(-0.5, 0.5)
            plt.plot(tbin, fbin, "o", mfc="none", label="folded and binned data")
            plt.plot(tm, fm, ".", markersize=3, zorder=-1000, label="model")
            plt.legend(loc="lower right")


if __name__ == "__main__":
    #    test_optimize_transit_params()
    test_compute_prediction()
