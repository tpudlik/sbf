"""Spherical Bessel function algorithms implemented using mpmath.

"""

PRECISION = 50

from mpmath import mp, pi, mpc, factorial, sin, cos
mp.dps = PRECISION

def sph_jn_exact(n, z):
    """Return the value of j_n computed using the exact formula.

    The expression used is http://dlmf.nist.gov/10.49.E1 .

    """
    s1 = mpc(0,0)
    s2 = mpc(0,0)
    zm = mpc(z)
    
    for k in xrange(0, int(n/2) + 1):
        s1 += (-1)**k*_a(2*k, n)/zm**(2k+1)
    for k in xrange(0, int((n-1)/2) + 1):
        s2 += (-1)**k*_a(2*k+1, n)/zm**(2k+2)
    
    return sin(zm - n*pi/2)*s1 + cos(zm - n*pi/2)*s2


def sph_yn_exact(n, z):
    pass


def _a(k, n):
    f = factorial # Abbreviation to make code more readable
    if k <= n:
        return f(n+k)/( mpc(2,0)**k*f(k)*f(n - k) )
    else:
        return 0