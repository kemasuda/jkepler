""""""

from jkepler.transit.transitfit import TransitFit
from jkepler.transit.transitfit import compute_prediction
from jkepler.tests.read_testdata import read_testdata_transit
from jkepler.tests.read_testdata import read_testdata_transit_koiinfo
from jkepler.tests.read_testdata import get_popt_refs
import pytest
import numpy as np


def test_compute_prediction():
    time, flux, error = read_testdata_transit()
    t0, period, b, rstar, rp_over_r, t0err, perr = read_testdata_transit_koiinfo()
    ecc, omega = 0 * t0, 0 * t0
    tpred = np.arange(
        np.min(time), np.max(time), np.median(np.diff(time)) * 0.2
    )  # dense time grid
    popt_refs = get_popt_refs()
    tf = TransitFit(time, exposure_time=29.4 / 1440.0, supersample_factor=10)
    fpred, gppred = compute_prediction(tf, flux, error, popt_refs, tpred)
    refs = [-3.0828440805885613, 0.010000640972700758]

    assert pytest.approx(np.sum(fpred)) == refs[0]
    assert pytest.approx(np.sum(gppred)) == refs[1]


if __name__ == "__main__":
    #    test_optimize_transit_params()
    test_compute_prediction()