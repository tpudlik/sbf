"""Spherical Bessel function algorithms developed by Liang-Wu Cai
(2011).

"""
import numpy as np


def recurrence_pattern(n, z, f0, f1):
    if n == 0:
        return f0
    if n == 1:
        return f1

    start_order = order(n, z)
    jlp1 = 0
    jl = 10**(-305)
    zinv = 1/z # Complex division is slower than multiplication
    for idx in xrange(start_order - n):
        jlm1 = (2*(start_order - idx) + 1)*jl*zinv - jlp1
        jlp1 = jl
        jl = jlm1
    
    out = jlm1

    for idx in xrange(n):
        jlm1 = (2*(n - idx) + 1)*jl*zinv - jlp1
        jlp1 = jl
        jl = jlm1

    if np.abs(f1) <= np.abs(f0):
        return out*(f0/jlm1)
    else:
        return out*(f1/jlp1)


def order(n, z):
    s = np.abs(np.sin(np.angle(z)))
    o_approx = np.floor((1.83 + 4.1*s**0.36)*np.abs(z)**(0.91 - 0.43*s**0.33) + 9*(1 - np.sqrt(s)))
    o_min = n + 1
    o_max = int(235 + 5*np.sqrt(np.abs(z)))
    if o_approx < o_min:
        return o_min
    if o_approx > o_max:
        return o_max
    else:
        return int(o_approx)


@np.vectorize
def sph_jn(n, z):
    return recurrence_pattern(n, z, np.sin(z)/z,
                              np.sin(z)/z**2 - np.cos(z)/z)
