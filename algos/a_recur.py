"""Ascending recurrence algorithms for spherical Bessel functions.

The recurrence relation used is http://dlmf.nist.gov/10.51.E1 .

"""
import numpy as np
from scipy.misc import factorial, factorial2


def recurrence_pattern(n, z, f0, f1):
    s0 = f0
    if n == 0:
        return s0
    s1 = f1
    if n == 1:
        return s1
    for idx in xrange(n - 1):
        sn = (2*idx + 3)/z*s1 - s0
        s0 = s1
        s1 = sn
    return sn


def v_recurrence_pattern(n, z, f0, f1):
    # This seems correct but produces seg faults.
    out = np.empty((n + z).shape)
    s0 = np.ones(shape=out.shape)*f0
    s1 = np.ones(shape=out.shape)*f1
    out[n == 0] = s0[n == 0]
    out[n == 1] = s1[n == 1]
    for idx in xrange(int(np.max(n)) - 1):
        sn = (2*idx + 3)/z*s1 - s0
        # Would an "if idx + 2 in n" speed this up?
        out[n == idx + 2] = sn[n == idx + 2]
        s0 = s1
        s1 = sn
    return out


@np.vectorize
def sph_jn(n, z):
    return recurrence_pattern(n, z,
                              np.sin(z)/z,
                              np.sin(z)/z**2 - np.cos(z)/z)

@np.vectorize
def sph_yn(n, z):
    return recurrence_pattern(n, z,
                              -np.cos(z)/z,
                              -np.cos(z)/z**2 - np.sin(z)/z)

@np.vectorize
def sph_in1(n, z):
    return recurrence_pattern(n, z,
                              np.sinh(z)/z,
                              -np.sinh(z)/z**2 + np.cosh(z)/z)

@np.vectorize
def sph_in2(n, z):
    return recurrence_pattern(n, z,
                              np.cosh(z)/z,
                              -np.cosh(z)/z**2 + np.sinh(z)/z)

@np.vectorize
def sph_kn(n, z):
    return recurrence_pattern(n, z,
                              np.pi/2*np.exp(-z)/z,
                              np.pi/2*np.exp(-z)*(1/z + 1/z**2))

def sph_h1n(n, z):
    return recurrence_pattern(n, z, -1j*np.exp(1j*z)/z, 
                              -(1j/z + 1)*np.exp(1j*z)/z)

def sph_h2n(n, z):
    return recurrence_pattern(n, z, 1j*np.exp(-1j*z)/z,
                              (1j/z - 1)*np.exp(-1j*z)/z)
