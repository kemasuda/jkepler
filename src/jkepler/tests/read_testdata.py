import pandas as pd
import pkg_resources
import jax.numpy as jnp
from astropy.time import Time

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

def read_testdata_astrometry():

    test_astfile = pkg_resources.resource_filename("jkepler", "data/tests/mann19_astrometry.txt")
    test_orbfile = pkg_resources.resource_filename("jkepler", "data/tests/mann19_orbit.txt")

    data = pd.read_csv(test_astfile, delimiter='|', comment='#')
    dates = [data.Date[i]+"T00:00:00" for i in range(len(data))]
    times = Time(dates, format='isot', scale='utc').mjd
    data['MJD'] = times
    odata = pd.read_csv(test_orbfile, delimiter='|', comment='#')
    data["Name"] = [n.strip(" ") for n in data.Name]
    odata["Name"] = [n.strip(" ") for n in odata.Name]
