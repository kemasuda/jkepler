import pytest
from jkepler.rv import rv_unit_amplitude as getrv
import pkg_resources
import pandas as pd
import jax.numpy as jnp

def read_testdata_rv():
    """Read in test data for RVs

    Returns:
        _type_: pandas.DataFrame
    """
    test_rvfile = pkg_resources.resource_filename('jkepler', 'data/tests/v723mon_s12_rv.csv')
    test_rv = pd.read_csv(test_rvfile)
    return test_rv

def test_getrv():
    rvdata = read_testdata_rv()
    t, y, yerr = jnp.array(rvdata.BJD), jnp.array(rvdata.rv), jnp.array(rvdata.rv_err)
    idx = t!=2454073.62965 # bad data
    t, y, yerr = t[idx], y[idx], yerr[idx]
    tepoch = t[0]
    t -= tepoch

    
if __name__ == '__main__':
    test_getrv()
