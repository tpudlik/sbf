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

import numpy as np
from mpmath import mp, mpf

# Path hack
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from algos.sbf_mp import (sph_jn_exact, sph_yn_exact, sph_h1n_exact,
                          sph_h2n_exact, sph_i1n_exact, sph_i2n_exact,
                          sph_kn_exact, sph_jn_bessel, sph_yn_bessel,
                          sph_h1n_bessel, sph_h2n_bessel, sph_i1n_bessel,
                          sph_i2n_bessel, sph_kn_bessel)
from accuracy.reference_points import reference_orders, reference_points
from accuracy.config import STARTING_PRECISION, ATOL, RTOL, PRECISION_STEP


def compute_values(precision, algorithm):
    """Return the values of the spherical Bessel functions, computed 
    to the given precision.

    The sizes of the returned 3D ndarrays are determined by the config
    settings.

    Parameters
    ----------
    precision : int
        The number of significant figures (mpmath.mp.dps setting)

    algorithm : str
        Either "exact" or "bessel".

    Returns
    -------
    sph_jn: ndarray
    sph_yn: ndarray
    sph_h1n: ndarray
    sph_h2n: ndarray
    sph_i1n: ndarray
    sph_i2n: ndarray
    sph_kn: ndarray

    """
    mp.dps = precision

    if algorithm == "exact":
        sph_jn  = np.vectorize(sph_jn_exact)
        sph_yn  = np.vectorize(sph_yn_exact)
        sph_h1n = np.vectorize(sph_h1n_exact)
        sph_h2n = np.vectorize(sph_h2n_exact)
        sph_i1n = np.vectorize(sph_i1n_exact)
        sph_i2n = np.vectorize(sph_i2n_exact)
        sph_kn  = np.vectorize(sph_kn_exact)
    elif algorithm == "bessel":
        sph_jn  = np.vectorize(sph_jn_bessel)
        sph_yn  = np.vectorize(sph_yn_bessel)
        sph_h1n = np.vectorize(sph_h1n_bessel)
        sph_h2n = np.vectorize(sph_h2n_bessel)
        sph_i1n = np.vectorize(sph_i1n_bessel)
        sph_i2n = np.vectorize(sph_i2n_bessel)
        sph_kn  = np.vectorize(sph_kn_bessel)
    else:
        raise ValueError("Unrecognized algorithm spec {}".format(algorithm))

    domain = reference_points()
    orders = reference_orders()

    return (sph_jn( orders, domain),
            sph_yn( orders, domain),
            sph_h1n(orders, domain),
            sph_h2n(orders, domain),
            sph_i1n(orders, domain),
            sph_i2n(orders, domain),
            sph_kn( orders, domain))


def mpc_close_enough(a, b, atol, rtol):
    """Assert mpmath.mpc objects a and b are within tolerance."""

    real = abs(a.real - b.real) <= (10**mpf(atol) + 10**mpf(rtol) * abs(b.real))
    imag = abs(a.imag - b.imag) <= (10**mpf(atol) + 10**mpf(rtol) * abs(b.imag))
    return (real and imag)


arrays_close_enough = np.vectorize(mpc_close_enough)


def all_arrays_close_enough(atuple, btuple, atol, rtol):
    """Assert all arrays in tuples a and b are within tolerance."""

    return all(np.all(arrays_close_enough(a, b, atol, rtol))
               for a, b, in itertools.izip(atuple, btuple))


if __name__ == "__main__":
    precision = STARTING_PRECISION
    done = False
    print "Starting computation with precision of {}...".format(precision)

    while not done:
        exact  = compute_values(precision, "exact")
        bessel = compute_values(precision, "bessel")
        
        if all_arrays_close_enough(exact, bessel, ATOL, RTOL):
            print "Success! Values at all points agree to target tolerance"
            np.savez("reference_values.npz",
                     sph_jn =(exact[0] + bessel[0])/2,
                     sph_yn =(exact[1] + bessel[1])/2,
                     sph_h1n=(exact[2] + bessel[2])/2,
                     sph_h2n=(exact[3] + bessel[3])/2,
                     sph_i1n=(exact[4] + bessel[4])/2,
                     sph_i2n=(exact[5] + bessel[5])/2,
                     sph_kn =(exact[6] + bessel[6])/2)
            done = True
        else:
            precision += PRECISION_STEP
            print "Incrementing precision to {} ...".format(precision)
