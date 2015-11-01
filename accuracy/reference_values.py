"""This module computes the reference values, i.e. values of the spherical
Bessel functions accurate to a specified number of significant figures at a
number of points in the complex plane.

This is done by computing the function values using two different algorithms
implemented using `mpmath`, an arbitrary-precision floating-point arithmetic
library.  The precision of the computation is increased until the outputs of
the two algorithms agree to a specified number of significant figures.

Note: computing the reference values takes a long time!

"""

# TODO: I don't like the np.array based implementation I used here and in
# reference_points.  In retrospect, I would prefer to have reference_points,
# reference_orders and compute_values to return regular iterables, rather than
# numpy arrays.  The trouble with the latter is that they end up packed with
# mpf and mpc objects anyway, and I have to constantly think which numpy
# methods will work for them, and which won't.  A good-ol iterable-based
# implementation would be easier to reason about.

import itertools
import sys
from os import path

import numpy as np
import mpmath

# Path hack
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from algos.sbf_mp import (sph_jn_exact, sph_yn_exact, sph_h1n_exact,
                          sph_h2n_exact, sph_i1n_exact, sph_i2n_exact,
                          sph_kn_exact, sph_jn_bessel, sph_yn_bessel,
                          sph_h1n_bessel, sph_h2n_bessel, sph_i1n_bessel,
                          sph_i2n_bessel, sph_kn_bessel)
from accuracy.reference_points import reference_orders, reference_points
from accuracy.config import TARGET_PRECISION


def compute_values(precision, algorithm):
    """Return the values of the spherical Bessel functions, computed 
    to the given precision.

    The sizes of the returned 2D ndarrays are determined by the config
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
    mpmath.mp.dps = precision

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


def all_arrays_close_enough(atuple, btuple, precision):
    """Assert all arrays in tuples a and b are within precision."""

    # TODO: This function is not yet correctly implemented.  The subtlety is
    # that the arrays contain complex elements, so you can't just compare them
    # to 10**(-precision).

    def iter():
        for a, b in itertools.izip(atuple, btuple)
            underestimate = a/b - 1 < 10**(-precision)
            overestimate = b/a - 1 < 10**(-precision)
            yield np.all(np.logical_or(underestimate, overestimate))
    
    return all(iter())


if __name__ == "__main__":
    precision = 15
    done = False
    print "Starting computation with precision of {}...".format(precision)

    while not done:
        exact  = compute_values(precision, "exact")
        bessel = compute_values(precision, "bessel")
        
        if all_arrays_close_enough(exact, bessel, TARGET_PRECISION):
            print "Success! Values at all points agree to {} significant figures".format(TARGET_PRECISION)
            np.savez("reference_values.npz",
                     sph_jn=exact[0],
                     sph_yn=exact[1],
                     sph_h1n=exact[2],
                     sph_h2n=exact[3],
                     sph_i1n=exact[4],
                     sph_i2n=exact[5],
                     sph_kn=exact[6])
            done = True
        else:
            print "Incrementing precision to {} ...".format(precision)
            precision += 5
