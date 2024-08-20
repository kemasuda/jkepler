from jkepler.tests.read_testdata import read_testdata_rv, read_testdata_astrometry, read_testdata_transit
import numpy as np
import pytest
import jax.numpy as jnp


def test_read_testdata_rv():
    t, y, yerr, tepoch = read_testdata_rv()
    assert t is not None
    assert y is not None
    assert yerr is not None
    assert tepoch is not None
    assert isinstance(t, jnp.ndarray)
    assert isinstance(y, jnp.ndarray)
    assert isinstance(yerr, jnp.ndarray)
    assert isinstance(tepoch, jnp.ndarray)

def test_read_testdata_astrometry():
    mjd, sep, e_sep, pa, e_pa, xobs, yobs = read_testdata_astrometry()
    assert mjd is not None
    assert sep is not None
    assert e_sep is not None
    assert pa is not None
    assert e_pa is not None
    assert xobs is not None
    assert yobs is not None
    assert isinstance(mjd, np.ndarray)
    assert isinstance(sep, np.ndarray)
    assert isinstance(e_sep, np.ndarray)
    assert isinstance(pa, np.ndarray)
    assert isinstance(e_pa, np.ndarray)
    assert isinstance(xobs, np.ndarray)
    assert isinstance(yobs, np.ndarray)

def test_read_testdata_transit():
    time, flux, error = read_testdata_transit()
    assert time is not None
    assert flux is not None
    assert error is not None
    assert isinstance(time, np.ndarray)
    assert isinstance(flux, np.ndarray)
    assert isinstance(error, np.ndarray)

if __name__ == "__main__":
    test_read_testdata_rv()