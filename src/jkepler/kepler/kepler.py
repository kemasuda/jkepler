__all__ = ['xyzrv_from_elements', 'rv_from_elements']

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
from jax import jit
from jax.lax import scan
from jax.config import config
config.update('jax_enable_x64', True)
from .markley import get_E

#%% a=1 coordinates & normalized RV
@jit
def xyzrv_from_elements(t, porb, ecc, inc, omega, lnode, tau):
    #M = 2*jnp.pi*(t-tau)/porb
    #Mmod = M % (2*jnp.pi)
    #M = jnp.where(Mmod >= jnp.pi, Mmod-2*jnp.pi, Mmod) # M in -pi to pi
    #E = get_E(M, ecc)
    #f = 2*jnp.arctan(jnp.sqrt((1+ecc)/(1-ecc))*jnp.tan(0.5*E))
    f = get_ta(t, porb, ecc, tau)
    omf = omega + f
    cosO, sinO = jnp.cos(lnode), jnp.sin(lnode)
    cosof, sinof = jnp.cos(omf), jnp.sin(omf)
    cosi, sini = jnp.cos(inc), jnp.sin(inc)
    r = (1.-ecc*ecc)/(1.+ecc*jnp.cos(f))
    x = r*(cosO*cosof - sinO*sinof*cosi)
    y = r*(sinO*cosof + cosO*sinof*cosi)
    z = r*sinof*sini
    vz = cosof + ecc*jnp.cos(omega)
    return x, y, z, vz

@jit
def rv_from_elements(t, porb, ecc, inc, omega, lnode, tau):
    f = get_ta(t, porb, ecc, tau)
    omf = omega + f
    vz = jnp.cos(omf) + ecc * jnp.cos(omega)
    return vz

#%%
@jit
def get_ta(t, porb, ecc, tau):
    M = 2*jnp.pi*(t-tau)/porb
    Mmod = M % (2*jnp.pi)
    M = jnp.where(Mmod >= jnp.pi, Mmod-2*jnp.pi, Mmod) # M in -pi to pi
    E = get_E(M, ecc)
    f = 2*jnp.arctan(jnp.sqrt((1+ecc)/(1-ecc))*jnp.tan(0.5*E))
    return f


#%%
"""
import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='sans-serif')
#sns.set_palette('colorblind')
plt.rcParams["figure.figsize"] = (18,6)
from matplotlib import rc
rc('text', usetex=True)

#%%
times = jnp.linspace(0, 10, 10000)

#%%
porb, ecc, inc, omega, lnode, tau = 1, 0.9, 90*jnp.pi/180, 0.2*jnp.pi, 0.1*jnp.pi, 1.5

#%%
%time x, y, z, vz = coords_from_elements(times, porb, ecc, inc, omega, lnode, tau)

#%%
plt.figure()
for _p, _l in zip([x, y, z], ['x', 'y', 'z']):
    plt.plot(times, _p, '-', label=_l)
plt.legend()

#%%
plt.plot(times, vz, '-')

#%%
@jit
def wrapM1(M):
    cosM, sinM = jnp.cos(M), jnp.sin(M)
    M = jnp.arctan2(sinM, cosM)
    return M

@jit
def wrapM2(M):
    M = 2*jnp.pi*(times-tau)/porb
    Mmod = M % (2*jnp.pi)
    Mpi = jnp.where(Mmod >= jnp.pi, Mmod-2*jnp.pi, Mmod)
    return Mpi

%timeit m1 = wrapM1(M)
%timeit m2 = wrapM2(M)
m1, m2 = wrapM1(M), wrapM2(M)
plt.plot(times, m1-m2)
"""
