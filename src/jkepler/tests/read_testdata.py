import pandas as pd
import pkg_resources
import numpy as np
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

def read_testdata_astrometry(name="GJ 1005"):
    """Read in test data for astrometry, using Mann+2019 data
    
    Args:
        name (str, optional): _description_. Defaults to "GJ 1005".

    Note:
        name can be chosen from the following list:
        'GJ 1005', 'GJ 2005', 'Gl 22', 'Gl 54', 'GJ 1038', 'Gl 65',
        'Gl 84', '2M0213+36', 'Gl 98', 'Gl 99', 'Gl 125', 'Gl 150.2',
        'Gl 190', 'GJ 1081', 'Gl 234', 'LHS 221', 'LHS 224', 'Gl 263',
        'Gl 277', '2M0736+07', 'Gl 301', 'Gl 310', 'Gl 330', 'LHS 6167',
        'Gl 340', 'Gl 352', 'Gl 381', 'Gl 416', 'Gl 469', 'Gl 473',
        'Gl 494', 'Gl 570', 'Gl 600', 'Gl 623', 'GJ 1210', 'Gl 660',
        'Gl 661', 'Gl 667', 'HIP 86707', 'Gl 695', 'Gl 747', 'Gl 748',
        'Gl 762.1', 'Gl 765.2', 'GJ 1245', 'Gl 791.2', 'Gl 804', 'Gl 831',
        'Gl 844', 'HD 239960', 'HIP 111685', 'Gl 893.4', 'Gl 900',
        'LHS 4009', 'Gl 913'

    Returns:
        float: mjd, sep, e_sep, pa, e_pa, xobs, yobs
    """
    test_astfile = pkg_resources.resource_filename("jkepler", "data/tests/mann19_astrometry.txt")

    data = pd.read_csv(test_astfile, delimiter='|', comment='#')
    dates = [data.Date[i]+"T00:00:00" for i in range(len(data))]    
    times = Time(dates, format='isot', scale='utc').mjd
    data['MJD'] = times
    data["Name"] = [n.strip(" ") for n in data.Name]
    didx = data.Name == name
    
    mjd, sep, e_sep, pa, e_pa = np.array(data[didx][["MJD", "Sep", "e_Sep", "PA", "e_PA"]]).T
    pa *= jnp.pi/180.
    e_pa *= jnp.pi/180.
    xobs, yobs = sep*np.cos(pa), sep*np.sin(pa)
    

    return mjd, sep, e_sep, pa, e_pa, xobs, yobs

if __name__ == "__main__":
    t, y, yerr, tepoch = read_testdata_rv()
    mjds, sep, e_sep, pa, e_pa, xobs, yobs = read_testdata_astrometry()

    