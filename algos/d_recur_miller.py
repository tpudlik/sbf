"""Miller descending recurrence algorithms for spherical Bessel function.

These are the algorithms attributed by Jablonski (1994) to Miller.  They are
simpler than the descending recurrence discussed by Cai (2011) in that the
starting order of the recurrence does not depend on the argument, and the
recurrence is always carried out all the way to n = 0 (rather than the larger
of n = 0 and n = 1).  This probably comes at the expense of accuracy, but I
would like to see that for myself.

"""
import numpy as np

ORDER = 100 # Following Jablonski (1994)

def recurrence_pattern(n, z, f0):
    if n == 0:
        return f0

    start_order = n + ORDER
    jlp1 = 0
    jl = 10**(-200)
    for idx in xrange(ORDER):
        jlm1 = (2*(start_order - idx) + 1)/z*jl - jlp1
        jlp1 = jl
        jl = jlm1
    
    out = jlm1

    for idx in xrange(n):
        jlm1 = (2*(n - idx) + 1)/z*jl - jlp1
        jlp1 = jl
        jl = jlm1

    return out*f0/jlm1

@np.vectorize
def sph_jn(n, z):
    return recurrence_pattern(n, z, np.sin(z)/z)

@np.vectorize
def sph_yn(n, z):
    return recurrence_pattern(n, z, -np.cos(z)/z)

@np.vectorize
def sph_i1n(n, z):
    return recurrence_pattern(n, z, np.sinh(z)/z)

@np.vectorize
def sph_i2n(n, z):
    return recurrence_pattern(n, z, np.cosh(z)/z)

@np.vectorize
def sph_kn(n, z):
    return recurrence_pattern(n, z, np.pi/2*np.exp(-z)/z)

def sph_h1n(n, z):
    return sph_jn(n, z) + 1j*sph_yn(n, z)

def sph_h2n(n, z):
    return sph_jn(n, z) - 1j*sph_yn(n, z)