
__all__ = ["flux_loss_cosi", "flux_loss_b"]

import jax.numpy as jnp
from jax import jit
from ..kepler import get_ta, t0_to_tau
from exoplanet_core.jax import ops


def rsky_over_a_from_elements(t, tau, period, ecc, omega, cosi):
    """ compute rsky / a

        Args:
            t: times at which position and RV are evaluated
            porb: orbital period
            ecc: eccentricity
            inc: inclination (radian)
            omega: argument of periastron (radian)
            tau: time of periastron passage

        Returns:

    """
    f = get_ta(t, period, ecc, tau)
    omf = omega + f
    cosof, sinof = jnp.cos(omf), jnp.sin(omf)
    cosi2 = cosi * cosi
    r = (1. - ecc*ecc) / (1. + ecc*jnp.cos(f))
    rsky = r * jnp.sqrt(cosof * cosof + sinof * sinof * cosi2)
    z_sign = sinof
    return rsky, z_sign


def b_to_cosi(b, ecc, omega, a_over_r):
    """ convert b into cosi following Eq.7 of Winn (2010), arXiv:1001.2010

    """
    efactor = (1. - ecc**2) / (1. + ecc * jnp.sin(omega))
    return b / a_over_r / efactor


def rsky_over_a_t0(t, t0, period, ecc, omega, cosi):
    """ only t needs to be an array here """
    tau = t0_to_tau(t0, period, ecc, omega)
    rsky_over_a, z_sign = rsky_over_a_from_elements(t, tau, period, ecc, omega, cosi)
    return rsky_over_a, z_sign


def compute_flux_loss(rsky_over_rstar, rp_over_rstar, z_sign, u1, u2):
    soln = ops.quad_solution_vector(rsky_over_rstar, rp_over_rstar)
    g = jnp.array([1. - u1 - 1.5*u2, u1 + 2*u2, -0.25*u2])
    I0 = jnp.pi * (g[0] + 2 * g[1] / 3.)
    flux_loss = jnp.where(z_sign > 0, jnp.dot(soln, g) / I0 - 1., 0.)
    return flux_loss


@jit
def flux_loss_cosi(t, t0, period, ecc, omega, cosi, a_over_rstar, rp_over_rstar, u1, u2):
    rsky_over_a, z_sign = rsky_over_a_t0(t, t0, period, ecc, omega, cosi)
    rsky_over_rstar = rsky_over_a * a_over_rstar
    return compute_flux_loss(rsky_over_rstar, rp_over_rstar*jnp.ones_like(t), z_sign, u1, u2)


@jit
def flux_loss_b(t, t0, period, ecc, omega, b, a_over_rstar, rp_over_rstar, u1, u2):
    """ at least rp_over_rstar needs to be an array like t """
    cosi = b_to_cosi(b, ecc, omega, a_over_rstar)
    return flux_loss_cosi(t, t0, period, ecc, omega, cosi, a_over_rstar, rp_over_rstar, u1, u2)
