"""Power series algorithms for spherical Bessel functions.

The power series are taken from http://dlmf.nist.gov/10.53

"""
import numpy as np
from scipy.misc import factorial, factorial2

def sph_jn(n, z, terms=100):
    s = sum((-z**2/2)**k/(factorial(k) * factorial2(2*n + 2*k + 1))
            for k in xrange(terms))
    return z**n * s

def sph_yn(n, z, terms=100):
    s1 = sum((z**2/2)**k * factorial2(2*n - 2*k - 1)/factorial(k)
             for k in xrange(n))
    s2 = sum((-z**2/2)**k/(factorial(k) * factorial2(2*k - 2*n - 1))
             for k in xrange(n + 1, terms))
    return -z**(-n - 1) * s1 + (-z)**(-n - 1) * s2

def sph_h1n(n, z, terms=100):
    return sph_jn(n, z, terms) + 1j*sph_yn(n, z, terms)

def sph_h2n(n, z, terms=100):
    return sph_jn(n, z, terms) - 1j*sph_yn(n, z, terms)

def sph_i1n(n, z, terms=100):
    s = sum((z**2/2)**k/(factorial(k) * factorial2(2*n + 2*k + 1))
            for k in xrange(terms))
    return z**n * s 

def sph_i2n(n, z, terms=100):
    s1 = sum((-z**2/2)**k * factorial2(2*n - 2*k - 1)/factorial(k)
             for k in xrange(n))
    s2 = sum((z**2/2)**k/(factorial(k) * factorial2(2*k - 2*n - 1))
             for k in xrange(n + 1, terms))
    return (-1)**n * z**(n + 1) * s1 + z**(-n - 1) * s2

def sph_kn(n, z, terms=100):
    return (-1)**(n + 1) * np.pi/2 * (sph_i1n(n, z, terms) - sph_i2n(n, z, terms))
