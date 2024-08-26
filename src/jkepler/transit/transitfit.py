
__all__ = ["TransitFit", "transitmodel", "transitmodel_ttv", "compute_prediction", "a_over_rstar_kep3", "q_to_u"]

import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
from itertools import chain
import celerite2
from celerite2.jax import terms as jax_terms
import copy
import jaxopt
from .transit import flux_loss_b


def q_to_u(q1, q2):
    """ convert q1, q2 into u1, u2

        Args:
            q1, q2: quadratic limb-darkening coefficients as parameterized in Kipping, D. M. 2013, MNRAS, 435, 2152

        Returns:
            u1, u2: quadratic limb-darkening coefficients

    """
    usum = jnp.sqrt(q1)
    u1 = 2 * q2 * usum
    u2 = usum - u1
    return u1, u2


def a_over_rstar_kep3(period, mstar, rstar):
    return 4.2083 * period**(2./3.) / rstar * mstar**(1./3.)


def supersampled_times(t, exposure_time, supersample_factor):
    supersample_num = int(supersample_factor // 2 * 2 + 1)
    dtarr = jnp.linspace(-0.5 * exposure_time, 0.5 * exposure_time, supersample_num)
    t_super = (t[:,None] + dtarr).ravel()
    return t_super, supersample_num


def transit_idx(t, t0, period):
    phase = (t - t0) / period
    tranum = np.round(phase)
    return tranum.astype(int)


def concat_dict(dicts):
    return dict(chain.from_iterable(d.items() for d in dicts))


def parameter_bounds(p_init, fit_params):
    """ return lower and upper bounds for parameters """
    p_low, p_high = copy.deepcopy(p_init), copy.deepcopy(p_init)
    for key,val in fit_params.items():
        p_low[key] = val[0]
        p_high[key] = val[1]
    return (p_low, p_high)


def transitmodel(self, p):
    u1, u2 = q_to_u(p['q1'], p['q2'])
    a = a_over_rstar_kep3(p['period'], 1., p['rstar'])
    model = self.compute_flux(p['t0'], p['period'], p['ecc'], p['omega'], p['b'], a, p['rp'], u1, u2)
    return model + p['meanflux']


def transitmodel_ttv(self, p):
    t0, period, ecc, omega, b, rstar, rp, q1, q2 = p['t0'], p['period'], p['ecc'], p['omega'], p['b'], p['rstar'], p['rp'], p['q1'], p['q2']
    mstar = 1.
    u1, u2 = q_to_u(q1, q2)
    a = a_over_rstar_kep3(period, mstar, rstar)
    ttvs = [p['ttv%d'%j] for j in range(len(t0))]
    flux_ttv = self.compute_flux_with_ttv(t0, period, ecc, omega, b, a, rp, u1, u2, ttvs)
    return flux_ttv + p['meanflux']


def compute_prediction(self, f, e, p, t_pred, fit_ttvs=False):
    if fit_ttvs:
        model = transitmodel_ttv(self, p)
    else:
        model = transitmodel(self, p)
    res = f - model
    kernel = jax_terms.Matern32Term(sigma=jnp.exp(p['lna']), rho=jnp.exp(p['lnc']))
    gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
    gp.compute(self.t, diag=e**2 + jnp.exp(2*p['lnjitter']))
    return model, gp.predict(res, t=t_pred)


class TransitFit():
    def __init__(self, t, exposure_time=None, supersample_factor=10):
        self.t = t
        if exposure_time is None:
            exposure_time = np.median(np.diff(t))
        self.exposure_time = exposure_time
        self.t_super, self.supersample_num = supersampled_times(t, exposure_time, supersample_factor)
        if self.supersample_num < 2:
            self.supersample_num = 1
            self.t_super = t

    @partial(jit, static_argnums=(0,))
    def compute_flux(self, t0, period, ecc, omega, b, a_over_r, rp_over_r, u1, u2):
        """compute total flux loss due to transiting planets
        t0, period, ..., rp_over_r should have the same shapes
        u1, u2 should be scalers
        
            Args:
                t0: times of inferior conjunction
                period: orbital periods
                ecc: eccentricities
                omega: arguments of periastron
                b: impact parameters
                a_over_r: semi-major axes divided by stellar radius
                rp_over_r: planet radii divded by stellar radius
                u1, u2: limb-darkening coefficients

            Returns:
                flux loss (same shape as self.t)
        
        """
        flux_super = jnp.sum(flux_loss_b(self.t_super[:,None], t0, period, ecc, omega, b, a_over_r, rp_over_r, u1, u2), axis=1)
        flux = jnp.mean(flux_super.reshape(len(self.t), self.supersample_num), axis=1)
        return flux

    @partial(jit, static_argnums=(0,))
    def compute_flux_with_ttv(self, t0, period, ecc, omega, b, a_over_r, rp_over_r, u1, u2, ttvs):
        flux_ttv = jnp.zeros_like(self.t_super)
        for j in range(len(t0)):
            tc_linear_j = t0[j] + self.transit_idx_set[j] * period[j]
            ttv_j = ttvs[j]
            tc_j = tc_linear_j + ttv_j
            flux_ttv += flux_loss_b(self.t_super, tc_j[self.tcidx_to_data[j]],
                                    period[j], ecc[j], omega[j], b[j], a_over_r[j], rp_over_r[j], u1, u2)
        flux_ttv = jnp.mean(flux_ttv.reshape(len(self.t), self.supersample_num), axis=1)
        return flux_ttv

    def set_ephemeris_info(self, t0, period):
        npl = len(t0)
        transit_idx_data = transit_idx(self.t_super[:,None], t0, period)
        transit_idx_set = [np.sort(list(set(np.array(transit_idx_data[:,j])))) for j in range(npl)]
        transit_nums = [len(idxset) for idxset in transit_idx_set]
        tcidx_to_data = [jnp.searchsorted(tidxset, tidxdata) for tidxset, tidxdata in zip(transit_idx_set, transit_idx_data.T)]
        self.transit_idx_set = transit_idx_set
        self.transit_nums = transit_nums
        self.tcidx_to_data = tcidx_to_data

    def optimize_transit_params(self, f, e,
                                t0, period, ecc, omega, b, rstar, rp_over_r, mstar=1.,
                                method="TNC", fit_ttvs=False, maxttv=0.2, fit_mean=True):
        """
        npl = len(dkoi)
        ones, zeros = np.ones(npl), np.zeros(npl)
        t0, period, b, depth, t0err, perr = np.array(dkoi[['koi_time0bk', 'koi_period', 'koi_impact', 'koi_depth', 'koi_time0bk_err1', 'koi_period_err1']]).T
        rp_over_r = np.sqrt(depth * 1e-6)
        u1, u2 = 0.5, 0.2
        rstar = dkoi.koi_srad[0]
        mstar = 1.
        a_over_r = 3.7528 * period**(2./3.) / rstar * mstar**(1./3.)
        ecc, omega = zeros, zeros
        """
        npl = len(t0)
        ones, zeros = np.ones(npl), np.zeros(npl)
        a_over_r = a_over_rstar_kep3(period, mstar, rstar)
        u1, u2 = 0.4, 0.2

        p_init = {
            "t0": t0, "period": period,
            "rstar": np.float64(rstar),
            "rp": rp_over_r,
            "q1": np.float64((u1+u2)**2),
            "q2": np.float64(0.5*u1/(u1+u2)),
            "b": b,
            "ecc": ecc,
            "omega": omega,
            "lna": np.float64(-5),
            "lnc": np.float64(-2),
            "lnjitter": np.float64(-13),
            "meanflux": np.float64(0.)
        }

        if fit_ttvs:
            self.set_ephemeris_info(t0, period)
            pdic_ttv = {}
            for j in range(npl):
                p_init['ttv%d'%j] = np.zeros(self.transit_nums[j])
                tcones = np.ones(self.transit_nums[j])
                pdic_ttv['ttv%d'%j] = (-maxttv*tcones, maxttv*tcones)


        eps = 1e-4
        pdic_ep = {"t0": (t0-1e-2, t0+1e-2), "period": (period-1e-3, period+1e-3)}
        pdic_rad = {"rp": (rp_over_r*0.5, rp_over_r*2.)}
        pdic_shape = {"b": (zeros+eps, ones+rp_over_r), #"a": (a_over_r*0.5, a_over_r*2.),
                      "rstar": (rstar*0.5, rstar*2.)}
        pdic_gp = {"lna": (-14+eps,-4-eps), "lnc": (-5+eps, 1-eps), "lnjitter": (-14+eps,-4-eps)}
        if fit_mean:
            pdic_mean = {"meanflux": (-1e-4*(1-eps), 1e-4*(1-eps))}
        else:
            pdic_mean = {"meanflux": (0, 0)}


        def objective(p):
            if fit_ttvs:
                model = transitmodel_ttv(self, p)
            else:
                model = transitmodel(self, p)
            res = f - model
            kernel = jax_terms.Matern32Term(sigma=jnp.exp(p['lna']), rho=jnp.exp(p['lnc']))
            gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
            gp.compute(self.t, diag=e**2 + jnp.exp(2*p['lnjitter']))
            return -2 * gp.log_likelihood(res)

        solver = jaxopt.ScipyBoundedMinimize(fun=objective, method=method)

        print ("# initial objective function: %.1f"%objective(p_init))

        print ()
        print ("# optimizing t0 and period...")
        bounds = parameter_bounds(p_init, pdic_ep)
        res = solver.run(p_init, bounds=bounds)
        p_init = res[0]
        print (res[1])

        print ()
        print ("# optimizing radius ratios and GP parameters...")
        bounds = parameter_bounds(p_init, concat_dict([pdic_rad, pdic_gp]))
        res = solver.run(p_init, bounds=bounds)
        p_init = res[0]
        print (res[1])

        print ()
        print ("# optimizing radius ratios, impact parameters, stellar radius, and GP parameters...")
        bounds = parameter_bounds(p_init, concat_dict([pdic_rad, pdic_shape, pdic_gp]))
        res = solver.run(p_init, bounds=bounds)
        p_init = res[0]
        print (res[1])

        print ()
        print ("# optimizing all parameters...")
        bounds = parameter_bounds(p_init, concat_dict([pdic_ep, pdic_rad, pdic_shape, pdic_gp]))
        res = solver.run(p_init, bounds=bounds)
        p_init = res[0]
        print (res[1])

        if not fit_ttvs:
            print ()
            print ("# optimizing all parameters with mean...")
            bounds = parameter_bounds(p_init, concat_dict([pdic_ep, pdic_rad, pdic_shape, pdic_gp, pdic_mean]))
            res = solver.run(p_init, bounds=bounds)
            p_init = res[0]
            print (res[1])
            return p_init

        print ()
        print ("# optimizing TTVs...")
        bounds = parameter_bounds(p_init, pdic_ttv)
        res = solver.run(p_init, bounds=bounds)
        p_init = res[0]
        print (res[1])

        print ()
        print ("# optimizing all parameters including TTVs...")
        bounds = parameter_bounds(p_init, concat_dict([pdic_ep, pdic_rad, pdic_shape, pdic_gp, pdic_ttv, pdic_mean]))
        res = solver.run(p_init, bounds=bounds)
        p_init = res[0]
        print (res[1])

        return p_init
