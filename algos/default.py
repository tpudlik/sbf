"""Default scipy spherical Bessel function algorithms.

The wrappers defined in this module produce a uniform interface with the
other algorithms.

"""
import numpy as np
import scipy.special

@np.vectorize
def sph_jn(n, z):
    return scipy.special.sph_jn(n, z)[0][-1]

@np.vectorize
def sph_yn(n, z):
    return scipy.special.sph_yn(n, z)[0][-1]

def sph_h1n(n, z):
    # No explicit support
    return sph_jn(n, z) + 1j*sph_yn(n, z)

def sph_h2n(n, z):
    # No explicit support
    return sph_jn(n, z) - 1j*sph_yn(n, z)

@np.vectorize
def sph_i1n(n, z):
    return scipy.special.sph_in(n, z)[0][-1]

def sph_i2n(n, z):
    # No explicit support
    return (-1)**n*2/np.pi*sph_kn(n, z) + sph_i1n(n, z)

@np.vectorize
def sph_kn(n, z):
    return scipy.special.sph_kn(n, z)[0][-1]
