"""Power series algorithms for spherical Bessel functions.

The power series are taken from http://dlmf.nist.gov/10.53

"""
import numpy as np
from scipy.misc import factorial, factorial2

def sph_jn(n, z, terms=50):
    s = sum((-z**2/2)**k/(factorial(k) * factorial2(2*n + 2*k + 1))
            for k in xrange(terms))
    return z**n * s


def sph_yn(n, z, terms=50):
    n = np.asarray(n, dtype=int)
    z = np.asarray(z)

    s1 = np.zeros(np.shape(n + z), dtype=z.dtype) # complex if z complex
    for k in xrange(np.max(n + 1)):
        s = (z**2/2)**k * factorial2(2*n - 2*k - 1)/factorial(k)
        s1[k <= n] += s[k <= n]

    s2 = np.zeros(np.shape(s1), dtype=z.dtype)
    for k in xrange(np.min(n + 1), terms):
        with np.errstate(divide='ignore'):
            # Where k < n + 1, we get 0 from the factorials.  This leads to a
            # division by zero, but the corresponding entries are not used.
            s = (-z**2/2)**k/(factorial(k) * factorial2(2*k - 2*n - 1))
        s2[k >= n + 1] += s[k >= n + 1]

    return -1/z**(n + 1) * s1 + (-z)**(-n - 1) * s2


def sph_h1n(n, z, terms=50):
    return sph_jn(n, z, terms) + 1j*sph_yn(n, z, terms)


def sph_h2n(n, z, terms=50):
    return sph_jn(n, z, terms) - 1j*sph_yn(n, z, terms)


def sph_i1n(n, z, terms=50):
    s = sum((z**2/2)**k/(factorial(k) * factorial2(2*n + 2*k + 1))
            for k in xrange(terms))
    return z**n * s 


def sph_i2n(n, z, terms=50):
    n = np.asarray(n, dtype=int)
    z = np.asarray(z)

    s1 = np.zeros(np.shape(n + z))
    for k in xrange(np.max(n + 1)):
        s = (-z**2/2)**k * factorial2(2*n - 2*k - 1)/factorial(k)
        s1[k <= n] += s[k <= n]

    s2 = np.zeros(s1.shape)
    for k in xrange(np.min(n + 1), terms):
        with np.errstate(divide='ignore'):
            s = (z**2/2)**k/(factorial(k) * factorial2(2*k - 2*n - 1))
        s2[k >= n + 1] += s[k >= n + 1]

    return (-1)**n * z**(n + 1) * s1 + z**(-n - 1) * s2


def sph_kn(n, z, terms=50):
    return (-1)**(n + 1) * np.pi/2 * (sph_i1n(n, z, terms) - sph_i2n(n, z, terms))
