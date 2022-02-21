# Kepler equation solver based on Markley (1995)

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
from jax import jit
from jax.lax import scan
#from jax.config import config
#config.update('jax_enable_x64', True)

#%%
"""
import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='sans-serif')
#sns.set_palette('colorblind')
plt.rcParams["figure.figsize"] = (18,6)
from matplotlib import rc
rc('text', usetex=True)
"""

#%%
"""
@jit
def get_E1_root(M, e, alpha):
    d = 3*(1.-e) + alpha*e
    sad = 6*alpha*d
    coeffs = [1, -3*M, sad*(1-e), -sad*d*M]
    z = jnp.roots(coeffs, strip_zeros=False)
    return z
#%timeit z = get_E1_root(M, e, alpha)
"""

#%%
@jit
def get_E1(M, e, alpha):
    d = 3*(1.-e) + alpha*e
    q = 2*alpha*d*(1-e) - M*M
    r = 3*alpha*d*(d-1+e)*M + M*M*M
    w = (jnp.abs(r) + jnp.sqrt(q*q*q + r*r))**(2./3.)
    z = 2*r*w/(w*w+w*q+q*q) + M
    return z/d
#%timeit z = get_E1(M, e, alpha)

#%%
@jit
def get_alpha(M, e):
    return (3*jnp.pi*jnp.pi + 1.6*jnp.pi*(jnp.pi-jnp.abs(M))/(1+e))/(jnp.pi*jnp.pi-6.)

#%%
@jit
def correct_E(E, M, e):
    ecosE, esinE = e*jnp.cos(E), e*jnp.sin(E)
    f0 = E - esinE - M
    f1 = 1 - ecosE
    f2 = esinE
    f3 = ecosE
    f4 = -esinE
    d3 = -f0 / (f1 - 0.5*f0*f2/f1)
    d4 = -f0 / (f1 + 0.5*d3*f2 + d3*d3*f3/6.)
    d5 = -f0 / (f1 + 0.5*d4*f2 + d4*d4*f3/6. + d4*d4*d4*f4/24.)
    return E + d5

#%%
@jit
def get_E(M, e):
    alpha = get_alpha(M, e)
    E1 = get_E1(M, e, alpha)
    return correct_E(E1, M, e)

#%%
"""
e = 0.1
e = 0.99
M = jnp.linspace(-1, 1, 10000)*jnp.pi

#%% 567us for 10000 M, e=0.1
%timeit E = get_E(M, e)
E = get_E(M, e)

#%%
#plt.plot(E, M)
plt.plot(E, jnp.abs(E-e*jnp.sin(E)-M), '.')

#%%
deltas = []
es = jnp.linspace(0, 1, 100)
for e in es:
    E = get_E(M, e)
    delta = jnp.abs(E-e*jnp.sin(E)-M)
    deltas.append(jnp.max(delta))
deltas = jnp.array(deltas)

#%%
plt.xlabel("$e$")
plt.ylabel("max value of $|E-e\sin E-M|$")
plt.plot(es, deltas, 'o')
plt.savefig("markley_test.png", dpi=200, bbox_inches='tight')
"""
