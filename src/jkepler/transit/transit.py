
__all__ = ["flux_loss_cosi", "flux_loss_b", "flux_loss_cosi_map", "flux_loss_b_map"]

import jax.numpy as jnp
from jax import jit, vmap
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


# vmap version of rsky_over_a_t0
rsky_over_a_t0_map = vmap(rsky_over_a_t0, (None,0,0,0,0,0), 0)


@jit
def flux_loss_cosi_map(t, t0, period, ecc, omega, cosi, a_over_rstar, rp_over_rstar, u1, u2):
    """compute flux loss mapping along the parameters other than time t

        Args:
            t: 1d time array (N,)
            t0: 1d array of time of inferior conjunction (Np,)
            period, ecc, ... should all be 1d arrays of the same shape (Np,)

        Returns:
            flux loss computed at time t, shape (Np, N)

    """
    rsky_over_a, z_sign = rsky_over_a_t0_map(t, t0, period, ecc, omega, cosi) # (Np, N)
    rsky_over_rstar = rsky_over_a * a_over_rstar[:,None]
    rp_over_rstar = jnp.ones_like(rsky_over_rstar) * rp_over_rstar[:,None]

    soln = ops.quad_solution_vector(rsky_over_rstar, rp_over_rstar) # (Np,N,3)
    g = jnp.array([1. - u1 - 1.5*u2, u1 + 2*u2, -0.25*u2]) # (3,Np)
    I0 = jnp.pi * (g[0] + 2 * g[1] / 3.)  # (Np,)
    flux_loss = jnp.sum(soln*g.T[:,None,:], axis=2) / I0[:,None] - 1. # (Np,N)
    flux_loss = jnp.where(z_sign > 0, flux_loss, 0)

    return flux_loss


@jit
def flux_loss_b_map(t, t0, period, ecc, omega, b, a_over_rstar, rp_over_rstar, u1, u2):
    """compute flux loss mapping along the parameters other than time t, given b instead of cosi

        Args:
            t: 1d time array (N,)
            t0: 1d array of time of inferior conjunction (Np,)
            period, ecc, ... should all be 1d arrays of the same shape (Np,)

        Returns:
            flux loss computed at time t, shape (Np, N)

    """
    cosi = b_to_cosi(b, ecc, omega, a_over_rstar)
    return flux_loss_cosi_map(t, t0, period, ecc, omega, cosi, a_over_rstar, rp_over_rstar, u1, u2)