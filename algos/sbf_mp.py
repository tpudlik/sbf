"""Spherical Bessel function algorithms implemented using mpmath.

Two different algorithms are implemented for each spherical Bessel function:
the exact formulas of http://dlmf.nist.gov/10.49 and the expressions in terms
of the ordinary Bessel functions, http://dlmf.nist.gov/10.47.ii .

"""
from numpy import iscomplex
from mpmath import (mp, pi, mpc, mpf, factorial, fac2, sin, cos, exp, besselj,
                    bessely, besseli, besselk, sqrt, hankel1, hankel2,
                    mpmathify)

# Exact expressions #

def sph_jn_exact(n, z):
    """Return the value of j_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E1 .

    """
    zm = mpmathify(z)
    s1 = sum((-1)**k*_a(2*k, n)/zm**(2*k+1) for k in xrange(0, int(n/2) + 1))
    s2 = sum((-1)**k*_a(2*k+1, n)/zm**(2*k+2) for k in xrange(0, int((n-1)/2) + 1))
    return sin(zm - n*pi/2)*s1 + cos(zm - n*pi/2)*s2


def sph_yn_exact(n, z):
    """Return the value of y_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E4 .

    """
    zm = mpmathify(z)
    s1 = sum((-1)**k*_a(2*k, n)/zm**(2*k+1) for k in xrange(0, int(n/2) + 1))
    s2 = sum((-1)**k*_a(2*k+1, n)/zm**(2*k+2) for k in xrange(0, int((n-1)/2) + 1))
    return -cos(zm - n*pi/2)*s1 + sin(zm - n*pi/2)*s2


def sph_h1n_exact(n, z):
    """Return the value of h^{(1)}_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E6 .

    """
    zm = mpmathify(z)
    s = sum(mpc(0,1)**(k-n-1)*_a(k, n)/zm**(k+1) for k in xrange(n+1))
    return exp(mpc(0,1)*zm)*s


def sph_h2n_exact(n, z):
    """Return the value of h^{(2)}_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E7 .

    """
    zm = mpmathify(z)
    s = sum(mpc(0,-1)**(k-n-1)*_a(k, n)/zm**(k+1) for k in xrange(n+1))
    return exp(mpc(0,-1)*zm)*s


def sph_i1n_exact(n, z):
    """Return the value of i^{(1)}_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E8 .

    """
    zm = mpmathify(z)
    s1 = sum(mpc(-1,0)**k * _a(k, n)/zm**(k+1) for k in xrange(n+1))
    s2 = sum(_a(k, n)/zm**(k+1) for k in xrange(n+1))
    return exp(zm)/2 * s1 + mpc(-1,0)**(n + 1)*exp(-zm)/2 * s2


def sph_i2n_exact(n, z):
    """Return the value of i^{(2)}_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E10 .

    """
    zm = mpmathify(z)
    s1 = sum(mpc(-1,0)**k * _a(k, n)/zm**(k+1) for k in xrange(n+1))
    s2 = sum(_a(k, n)/zm**(k+1) for k in xrange(n+1))
    return exp(zm)/2 * s1 + mpc(-1,0)**n*exp(-zm)/2 * s2


def sph_kn_exact(n, z):
    """Return the value of k_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E12 .

    """
    zm = mpmathify(z)
    s = sum(_a(k, n)/zm**(k+1) for k in xrange(n+1))
    return pi*exp(-zm)/2*s


A_CACHE = {}
def _a(k, n, dps=mp.dps):
    """Return the value of the Bessel asymptotic expansion coefficient.

    Defined as in http://dlmf.nist.gov/10.49#E1 , except I use the notation
    a(k, n) for their a(k, n + 1/2).  A simple cache is used to improve
    performance, since these coefficients must be computed many times.

    """
    if (k, n, dps) in A_CACHE:
        return A_CACHE[(k, n, dps)]
    else:
        if k <= n:
            f = factorial # Abbreviation to make code more readable
            v = f(n + k)/( mpf(2)**k * f(k) * f(n - k) )
            A_CACHE[(k, n, dps)] = v
            return v
        else:
            A_CACHE[(k, n, dps)] = 0
            return 0


# Ordinary Bessel function expressions #

def sph_jn_bessel(n, z):
    out = besselj(n + mpf(1)/2, z)*sqrt(pi/(2*z))
    if mpmathify(z).imag == 0:
        return out.real # Small imaginary parts are spurious
    else:
        return out

def sph_yn_bessel(n, z):
    out = bessely(n + mpf(1)/2, z)*sqrt(pi/(2*z))
    if mpmathify(z).imag == 0:
        return out.real
    else:
        return out

def sph_h1n_bessel(n, z):
    return hankel1(n + mpf(1)/2, z)*sqrt(pi/(2*z))

def sph_h2n_bessel(n, z):
    return hankel2(n + mpf(1)/2, z)*sqrt(pi/(2*z))

def sph_i1n_bessel(n, z):
    out = besseli(n + mpf(1)/2, z)*sqrt(pi/(2*z))
    if mpmathify(z).imag == 0:
        return out.real
    else:
        return out

def sph_i2n_bessel(n, z):
    out = besseli(- n - mpf(1)/2, z)*sqrt(pi/(2*z))
    if mpmathify(z).imag == 0:
        return out.real
    else:
        return out

def sph_kn_bessel(n, z):
    out = besselk(n + mpf(1)/2, z)*sqrt(pi/(2*z))
    if mpmathify(z).imag == 0:
        return out.real
    else:
        return out


# Power series (experimental)

def sph_jn_power(n, z, terms=100):
    zm = mpmathify(z)
    s = sum((-z**2/2)**k/(factorial(k) * fac2(2*n + 2*k + 1)) for k in xrange(terms))
    return zm**n * s
