__all__ = ["get_ta", "t0_to_tau"]

import jax.numpy as jnp
from .markley import get_E

def get_ta(t, porb, ecc, tau):
    """ compute true anomaly

        Args:
            t: times
            porb: orbital period
            ecc: eccentricity
            tau: time of periastron passage

        Returns:
            true anomaly

    """
    M = 2 * jnp.pi * (t - tau) / porb
    Mmod = M % (2*jnp.pi)
    M = jnp.where(Mmod >= jnp.pi, Mmod-2*jnp.pi, Mmod) # M in -pi to pi
    E = get_E(M, ecc)
    f = 2 * jnp.arctan(jnp.sqrt((1+ecc)/(1-ecc))*jnp.tan(0.5*E))
    return f


def t0_to_tau(t0, period, ecc, omega):
    """ compute time of periastron passage from time of inferior conjunction

        Args:
            t0: time of inferior conjunction where omega+f = pi/2
            period: orbital period
            ecc: eccentricity
            omega: argument of periastron

        Returns:
            time of periastron passage

    """
    tanw2 = jnp.tan(0.5 * omega)
    u0 = 2 * jnp.arctan( jnp.sqrt((1.-ecc)/(1.+ecc)) * (1.-tanw2)/(1.+tanw2) )
    tau = t0 - period / (2 * jnp.pi) * (u0 - ecc*jnp.sin(u0))
    return tau
