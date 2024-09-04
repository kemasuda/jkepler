import pandas as pd
import pkg_resources
import numpy as np
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


def read_testdata_astrometry(name="GJ 1005"):
    from astropy.time import Time

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
    test_astfile = pkg_resources.resource_filename(
        "jkepler", "data/tests/mann19_astrometry.txt"
    )

    data = pd.read_csv(test_astfile, delimiter="|", comment="#")
    dates = [data.Date[i] + "T00:00:00" for i in range(len(data))]
    times = Time(dates, format="isot", scale="utc").mjd
    data["MJD"] = times
    data["Name"] = [n.strip(" ") for n in data.Name]
    didx = data.Name == name

    mjd, sep, e_sep, pa, e_pa = np.array(
        data[didx][["MJD", "Sep", "e_Sep", "PA", "e_PA"]]
    ).T
    pa *= jnp.pi / 180.0
    e_pa *= jnp.pi / 180.0
    xobs, yobs = sep * np.cos(pa), sep * np.sin(pa)

    return mjd, sep, e_sep, pa, e_pa, xobs, yobs


def read_testdata_transit():
    """read in test data for transits

    Returns:
        float: time, flux, error
    """
    test_trfile = pkg_resources.resource_filename(
        "jkepler", "data/tests/kic11773022_long_transits.csv"
    )
    data = pd.read_csv(test_trfile)
    time, flux, error = np.array(data.time), np.array(data.flux), np.array(data.error)
    return time, flux, error


def read_testdata_transit_koiinfo():
    test_infofile = pkg_resources.resource_filename(
        "jkepler", "data/tests/kic11773022_koiinfo.csv"
    )
    dkoi = pd.read_csv(test_infofile)
    t0, period, b, depth, t0err, perr = np.array(
        dkoi[
            [
                "koi_time0bk",
                "koi_period",
                "koi_impact",
                "koi_depth",
                "koi_time0bk_err1",
                "koi_period_err1",
            ]
        ]
    ).T
    rp_over_r = np.sqrt(depth * 1e-6)
    rstar = dkoi.koi_srad[0]
    return t0, period, b, rstar, rp_over_r, t0err, perr


if __name__ == "__main__":
    t, y, yerr, tepoch = read_testdata_rv()
    mjds, sep, e_sep, pa, e_pa, xobs, yobs = read_testdata_astrometry()
    time, flux, error = read_testdata_transit()


def get_popt_refs():
    """get reference values for testdata_transit (GJ 1005)

    Returns:
        popt_refs (dict): reference values for transit fit
    """
    popt_refs = {
        "b": np.array([0.45452353, 0.99861012, 0.45813456]),
        "ecc": np.array([0.0, 0.0, 0.0]),
        "lna": -8.08475498,
        "lnc": -3.43751221,
        "lnjitter": -10.16308495,
        "meanflux": 1.37338052e-05,
        "omega": np.array([0.0, 0.0, 0.0]),
        "period": np.array([45.15524184, 85.31652727, 130.17702962]),
        "q1": 0.36,
        "q2": 0.33333333,
        "rp": np.array([0.07381892, 0.07379244, 0.10052868]),
        "rstar": 0.93461991,
        "t0": np.array([159.10729446, 295.31493438, 212.03698193]),
    }
    return popt_refs
