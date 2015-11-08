"""This module computes the reference values, i.e. values of the spherical
Bessel functions accurate to a specified tolerance at a number of points in
the complex plane.

This is done by computing the function values with two different algorithms
implemented using `mpmath`, an arbitrary-precision floating-point arithmetic
library.  The precision of the computation is increased until the outputs of
the two algorithms agree to a specified absolute and relative tolerance.

Note: computing the reference values takes a long time!

"""
import itertools
import sys
from os import path
import cPickle as pickle

import numpy as np
from mpmath import mp, mpf

# Path hack
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from algos.sbf_mp import (sph_jn_exact, sph_yn_exact, sph_h1n_exact,
                          sph_h2n_exact, sph_i1n_exact, sph_i2n_exact,
                          sph_kn_exact, sph_jn_bessel, sph_yn_bessel,
                          sph_h1n_bessel, sph_h2n_bessel, sph_i1n_bessel,
                          sph_i2n_bessel, sph_kn_bessel)
from accuracy.reference_points import reference_points
from accuracy.config import STARTING_PRECISION, MAX_PRECISION, ATOL, RTOL


def reference_value(point, order, sbf):
    """Return the value of the spherical Bessel function at a point.

    The precision of the computation is increased until the two algorithms
    used agree to within ATOL and RTOL.

    Parameters
    ----------
    point : float or complex
        The argument of the Bessel function.
    order : int
        The order of the Bessel function.
    sbf : str
        The requsted Bessel function.  One of "jn", "yn", "h1n", "h2n", "i1n",
        "i2n" or "kn".

    Returns
    -------
    out : mpf or mpc
        The value of the requested Bessel function.

    """
    f1 = choose_function(sbf, "exact")
    f2 = choose_function(sbf, "bessel")
    mp.dps = STARTING_PRECISION
    while True:
        exact = f1(point, order)
        bessel = f2(point, order)
        if mpc_close_enough(exact, bessel, ATOL, RTOL):
            if exact.imag == 0:
                return np.float64(exact.real)
            else:
                return np.complex128(exact)
        else:
            mp.dps = mp.dps + mp.dps/2
            if mp.dps > MAX_PRECISION:
                return np.nan


def choose_function(sbf, algorithm):
    if algorithm == "exact":
        if sbf == "jn": return sph_jn_exact
        if sbf == "yn": return sph_yn_exact
        if sbf == "h1n": return sph_h1n_exact
        if sbf == "h2n": return sph_h2n_exact
        if sbf == "i1n": return sph_i1n_exact
        if sbf == "i2n": return sph_i2n_exact
        if sbf == "kn": return sph_kn_exact
        raise ValueError("Unrecognized sbf: {}".format(sbf))
    elif algorithm == "bessel":
        if sbf == "jn": return sph_jn_bessel
        if sbf == "yn": return sph_yn_bessel
        if sbf == "h1n": return sph_h1n_bessel
        if sbf == "h2n": return sph_h2n_bessel
        if sbf == "i1n": return sph_i1n_bessel
        if sbf == "i2n": return sph_i2n_bessel
        if sbf == "kn": return sph_kn_bessel
        raise ValueError("Unrecognized sbf: {}".format(sbf))
    else:
        raise ValueError("Unrecognized algorithm: {}".format(algorithm))


def mpc_close_enough(a, b, atol, rtol):
    """Assert mpmath.mpc or mpmath.mpf objects a and b are within tolerance."""

    real = abs(a.real - b.real) <= (10**mpf(atol) + 10**mpf(rtol) * abs(b.real))
    imag = abs(a.imag - b.imag) <= (10**mpf(atol) + 10**mpf(rtol) * abs(b.imag))
    return (real and imag)


if __name__ == "__main__":
    sbf_list = ["jn", "yn", "h1n", "h2n", "i1n", "i2n", "kn"]
    for sbf in sbf_list:
        values = map(lambda p: reference_value(p[1], p[0], sbf),
                     reference_points())
        print "Computed {} values!".format(sbf)
        with open(sbf + ".pickle", "wb") as f:
            pickle.dump(values, f)
        np.save(sbf + ".npy", np.array(values, dtype=np.complex128))
