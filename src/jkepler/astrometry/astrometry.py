__all__ = ["xyzv_from_elements"]

import jax.numpy as jnp
from ..kepler import get_ta


def xyzv_from_elements(t, porb, ecc, inc, omega, lnode, tau):
    """ compute relative coordinates and RV

        Args:
            t: times at which position and RV are evaluated
            porb: orbital period
            ecc: eccentricity
            inc: inclination (radian)
            omega: argument of periastron (radian)
            lnode: longitude of ascending node (radian)
            tau: time of periastron passage

        Returns:
            x, y, z coordinates and radial velocity

    """
    f = get_ta(t, porb, ecc, tau)
    omf = omega + f
    cosO, sinO = jnp.cos(lnode), jnp.sin(lnode)
    cosof, sinof = jnp.cos(omf), jnp.sin(omf)
    cosi, sini = jnp.cos(inc), jnp.sin(inc)
    r = (1. - ecc*ecc) / (1. + ecc*jnp.cos(f))
    x = r * (cosO*cosof - sinO*sinof*cosi)
    y = r * (sinO*cosof + cosO*sinof*cosi)
    z = r * sinof * sini
    vz = cosof + ecc * jnp.cos(omega)
    return x, y, z, vz
