"""Default scipy spherical Bessel function algorithms.

The wrappers defined in this module produce a uniform interface with the
other algorithms.

"""
import numpy as np
import scipy.special

def sph_jn_inner(n, z):
    return scipy.special.sph_jn(n, z)[0][-1]

def sph_yn_inner(n, z):
    return scipy.special.sph_yn(n, z)[0][-1]

def sph_h1n(n, z):
    # No explicit support
    return sph_jn(n, z) + 1j*sph_yn(n, z)

def sph_h2n(n, z):
    # No explicit support
    return sph_jn(n, z) - 1j*sph_yn(n, z)

def sph_i1n_inner(n, z):
    return scipy.special.sph_in(n, z)[0][-1]

def sph_i2n(n, z):
    # No explicit support
    return (-1)**n*2/np.pi*sph_kn(n, z) + sph_i1n(n, z)

def sph_kn_inner(n, z):
    return scipy.special.sph_kn(n, z)[0][-1]

sph_jn  = np.vectorize(sph_jn_inner)
sph_yn  = np.vectorize(sph_yn_inner)
sph_i1n = np.vectorize(sph_i1n_inner)
sph_kn  = np.vectorize(sph_kn_inner)
