from jkepler.transit.transitfit import TransitFit
from jkepler.tests.read_testdata import read_testdata_transit
from jkepler.tests.read_testdata import read_testdata_transit_koiinfo
from jkepler.tests.read_testdata import get_popt_refs

import numpy as np



def test_optimize_transit_params():
    """test the optimization of transit parameters"""
    time, flux, error = read_testdata_transit()
    t0, period, b, rstar, rp_over_r, t0err, perr = read_testdata_transit_koiinfo()
    ecc, omega = 0 * t0, 0 * t0
    tpred = np.arange(
        np.min(time), np.max(time), np.median(np.diff(time)) * 0.2
    )  # dense time grid
    tf = TransitFit(time, exposure_time=29.4 / 1440.0, supersample_factor=10)
    popt = tf.optimize_transit_params(
        flux, error, t0, period, ecc, omega, b, rstar, rp_over_r, fit_ttvs=False
    )  # require jaxopt>=0.8.3
    #refs = np.array([0.45452353, 0.99861012, 0.45813456])
    popt_refs = get_popt_refs()["b"]

    assert np.allclose(popt["b"], popt_refs, atol=1e-2)


if __name__ == "__main__":
    test_optimize_transit_params()
