__all__ = ["rv_unit_amplitude", "rv_unit_amplitude_multi"]

import jax.numpy as jnp
from jax import vmap
from ..kepler import get_ta


def rv_unit_amplitude(t, porb, ecc, omega, tau):
    """ compute cos(omega+f) + e*cos(omega)

        Args:
            t: times at which RVs are computed
            porb: period
            ecc: eccentricity
            omega: argument of periastron
            tau: time of periastron passage

        Returns:
            radial velocities

    """
    f = get_ta(t, porb, ecc, tau)
    omf = omega + f
    vz = jnp.cos(omf) + ecc * jnp.cos(omega)
    return vz


rv_unit_amplitude_multi = vmap(rv_unit_amplitude, (None,0,0,0,0), 0)
