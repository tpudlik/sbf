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


def sph_jn_inner(n, z):
    return recurrence_pattern(n, z,
                              np.sin(z)/z,
                              np.sin(z)/z**2 - np.cos(z)/z)

def sph_yn_inner(n, z):
    return recurrence_pattern(n, z,
                              -np.cos(z)/z,
                              -np.cos(z)/z**2 - np.sin(z)/z)

def sph_in1_inner(n, z):
    return recurrence_pattern(n, z,
                              np.sinh(z)/z,
                              -np.sinh(z)/z**2 + np.cosh(z)/z)

def sph_in2_inner(n, z):
    return recurrence_pattern(n, z,
                              np.cosh(z)/z,
                              -np.cosh(z)/z**2 + np.sinh(z)/z)

def sph_kn_inner(n, z):
    return recurrence_pattern(n, z,
                              np.pi/2*np.exp(-z)/z,
                              np.pi/2*np.exp(-z)*(1/z + 1/z**2))

sph_jn = np.vectorize(sph_jn_inner)    
sph_yn = np.vectorize(sph_yn_inner)
sph_in1 = np.vectorize(sph_in1_inner)
sph_in2 = np.vectorize(sph_in2_inner)
sph_kn = np.vectorize(sph_kn_inner)

def sph_h1n(n, z):
    return sph_jn(n, z) + 1j*sph_yn(n, z)

def sph_h2n(n, z):
    return sph_jn(n, z) - 1j*sph_yn(n, z)

