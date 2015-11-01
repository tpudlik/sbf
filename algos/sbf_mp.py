"""Spherical Bessel function algorithms implemented using mpmath.

Two different algorithms are implemented for each spherical Bessel function:
the exact formulas of http://dlmf.nist.gov/10.49 and the expressions in terms
of the ordinary Bessel functions, http://dlmf.nist.gov/10.47.ii .

"""
from mpmath import (mp, pi, mpc, factorial, sin, cos, exp, besselj, bessely,
                    besseli, besselk, sqrt, hankel1, hankel2)

# Exact expressions #

def sph_jn_exact(n, z):
    """Return the value of j_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E1 .

    """
    zm = mpc(z)
    s1 = sum((-1)**k*_a(2*k, n)/zm**(2*k+1) for k in xrange(0, int(n/2) + 1))
    s2 = sum((-1)**k*_a(2*k+1, n)/zm**(2*k+2) for k in xrange(0, int((n-1)/2) + 1))
    return sin(zm - n*pi/2)*s1 + cos(zm - n*pi/2)*s2


def sph_yn_exact(n, z):
    """Return the value of y_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E4 .

    """
    zm = mpc(z)
    s1 = sum((-1)**k*_a(2*k, n)/zm**(2*k+1) for k in xrange(0, int(n/2) + 1))
    s2 = sum((-1)**k*_a(2*k+1, n)/zm**(2*k+2) for k in xrange(0, int((n-1)/2) + 1))
    return -cos(zm - n*pi/2)*s1 + sin(zm - n*pi/2)*s2


def sph_h1n_exact(n, z):
    """Return the value of h^{(1)}_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E6 .

    """
    zm = mpc(z)
    s = sum(mpc(0,1)**(k-n-1)*_a(k, n)/zm**(k+1) for k in xrange(n+1))
    return exp(mpc(0,1)*zm)*s


def sph_h2n_exact(n, z):
    """Return the value of h^{(2)}_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E7 .

    """
    zm = mpc(z)
    s = sum(mpc(0,-1)**(k-n-1)*_a(k, n)/zm**(k+1) for k in xrange(n+1))
    return exp(mpc(0,-1)*zm)*s


def sph_i1n_exact(n, z):
    """Return the value of i^{(1)}_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E8 .

    """
    zm = mpc(z)
    s1 = sum(mpc(-1,0)**k * _a(k, n)/zm**(k+1) for k in xrange(n+1))
    s2 = sum(_a(k, n)/zm**(k+1) for k in xrange(n+1))
    return exp(zm)/2 * s1 + mpc(-1,0)**(n + 1)*exp(-zm)/2 * s2


def sph_i2n_exact(n, z):
    """Return the value of i^{(2)}_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E10 .

    """
    zm = mpc(z)
    s1 = sum(mpc(-1,0)**k * _a(k, n)/zm**(k+1) for k in xrange(n+1))
    s2 = sum(_a(k, n)/zm**(k+1) for k in xrange(n+1))
    return exp(zm)/2 * s1 + mpc(-1,0)**n*exp(-zm)/2 * s2


def sph_kn_exact(n, z):
    """Return the value of k_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E12 .

    """
    zm = mpc(z)
    s = sum(_a(k, n)/zm**(k+1) for k in xrange(n+1))
    return pi*exp(-zm)/2*s


A_CACHE = {}
def _a(k, n):
    """Return the value of the Bessel asymptotic expansion coefficient.

    Defined as in http://dlmf.nist.gov/10.49#E1 , except I use the notation
    a(k, n) for their a(k, n + 1/2).  A simple cache is used to improve
    performance, since these coefficients must be computed many times.

    """
    if (k, n) in A_CACHE:
        return A_CACHE[(k, n)]
    else:
        if k <= n:
            f = factorial # Abbreviation to make code more readable
            v = f(n + k)/( mpc(2,0)**k * f(k) * f(n - k) )
            A_CACHE[(k, n)] = v
            return v
        else:
            A_CACHE[(k, n)] = 0
            return 0


# Ordinary Bessel function expressions #

def sph_jn_bessel(n, z):
    return besselj(n + mpc(1,0)/2, z)*sqrt(pi/(2*z))

def sph_yn_bessel(n, z):
    return bessely(n + mpc(1,0)/2, z)*sqrt(pi/(2*z))

def sph_h1n_bessel(n, z):
    return hankel1(n + mpc(1,0)/2, z)*sqrt(pi/(2*z))

def sph_h2n_bessel(n, z):
    return hankel2(n + mpc(1,0)/2, z)*sqrt(pi/(2*z))

def sph_i1n_bessel(n, z):
    return besseli(n + mpc(1,0)/2, z)*sqrt(pi/(2*z))

def sph_i2n_bessel(n, z):
    return besseli(- n - mpc(1,0)/2, z)*sqrt(pi/(2*z))

def sph_kn_bessel(n, z):
    return besselk(n + mpc(1,0)/2, z)*sqrt(pi/(2*z))

