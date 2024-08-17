import pandas as pd
import pkg_resources
import jax.numpy as jnp

def read_testdata_rv():
    """Read in test data for RVs

    Returns:
        jnp.array: time, rv, rv_err, tepoch
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

    return t, y, yerr, tepoch
