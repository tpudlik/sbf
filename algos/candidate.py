"""Candidate new algorithms for use in SciPy.

These implementations are designed to be accurate, rather than performant.
But the algorithms used will be implemented more efficiently once their
correctness is verified.

"""
import numpy as np
import scipy.special
from scipy.misc import factorial, factorial2


def sph_jn(n, z):
    out = _sph_jn_bessel(n, z)
    idx = np.logical_and(np.isreal(z), np.abs(z) > n)
    if np.any(idx):
        # Ascending recurrence is more accurate for large real z
        out[idx] = _sph_jn_a_recur(n[idx], z[idx])
    return out


def sph_yn(n, z):
    idx = np.isreal(z)
    out =  _sph_yn_bessel(n, z)
    if np.any(idx):
        # Ascending recurrence is more accurate for real z
        out[idx] = _sph_yn_a_recur(n[idx], z[idx])
    if np.any(np.iscomplex(out)):
        out[np.logical_and(np.isnan(out), np.iscomplex(out))] = np.inf*(1+1j)
    return out


@np.vectorize
def _sph_jn_a_recur(n, z):
    return recurrence_pattern(n, z,
                              np.sin(z)/z,
                              np.sin(z)/z**2 - np.cos(z)/z)


@np.vectorize
def _sph_yn_a_recur(n, z):
    return recurrence_pattern(n, z,
                              -np.cos(z)/z,
                              -np.cos(z)/z**2 - np.sin(z)/z)


def _sph_jn_bessel(n, z):
    return np.sqrt(0.5*np.pi/z)*scipy.special.jv(n + 0.5, z)


def _sph_yn_bessel(n, z):
    return np.sqrt(0.5*np.pi/z)*scipy.special.yv(n + 0.5, z)


def recurrence_pattern(n, z, f0, f1):
    """Ascending recurrence for jn, yn, h1n, h2n."""
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
        if np.isinf(sn):
            # Overflow occurred already: terminate recurrence.
            return sn
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

