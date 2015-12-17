"""This module computes the reference values, i.e. values of the spherical
Bessel functions accurate to a specified tolerance at a number of points in
the complex plane.

The script creates two sets of files: .npy files with double-precision
values, and .pickle files with arbitrary precision (mpf, mpc) values.

Note: computing the reference values takes a long time!

"""
import itertools, sys, logging
from os import path
import cPickle as pickle

import numpy as np
from mpmath import mp, mpf, isnan

# Path hack
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from algos.sbf_mp import (sph_jn_bessel, sph_yn_bessel,
                          sph_h1n_bessel, sph_h2n_bessel, sph_i1n_bessel,
                          sph_i2n_bessel, sph_kn_bessel)
from reference.reference_points import reference_points
from reference.config import (STARTING_PRECISION, MAX_PRECISION, ATOL, RTOL,
                              PRECISION_STEP)


def reference_value(point, order, sbf):
    """Return the value of the spherical Bessel function at a point.

    The precision of the computation is increased until the obtained value
    stabilizes to within ATOL and RTOL.

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
    f = choose_function(sbf)
    mp.dps = STARTING_PRECISION
    while True:
        lower  = f(order, point)
        mp.dps = mp.dps + PRECISION_STEP
        higher = f(order, point)
        if mpc_close_enough(lower, higher, ATOL, RTOL):
            return higher
        else:
            mp.dps = mp.dps - PRECISION_STEP + mp.dps/2
            if mp.dps > MAX_PRECISION:
                logging.warning("No convergence at max precision for {} of order {} at {}".format(sbf, order, point))
                return mpmath.nan


def mpc_to_np(x):
    if isnan(x):
        return np.nan
    if x.imag == 0:
        return np.float64(x.real)
    
    return np.complex128(x)


def choose_function(sbf):
    if sbf == "jn": return sph_jn_bessel
    if sbf == "yn": return sph_yn_bessel
    if sbf == "h1n": return sph_h1n_bessel
    if sbf == "h2n": return sph_h2n_bessel
    if sbf == "i1n": return sph_i1n_bessel
    if sbf == "i2n": return sph_i2n_bessel
    if sbf == "kn": return sph_kn_bessel
    raise ValueError("Unrecognized sbf: {}".format(sbf))


def mpc_close_enough(a, b, atol, rtol):
    """Assert mpmath.mpc or mpmath.mpf objects a and b are within tolerance."""

    real = abs(a.real - b.real) <= (10**mpf(atol) + 10**mpf(rtol) * abs(b.real))
    imag = abs(a.imag - b.imag) <= (10**mpf(atol) + 10**mpf(rtol) * abs(b.imag))
    return (real and imag)


if __name__ == "__main__":
    logging.basicConfig(filename='reference_values.log', level=logging.DEBUG)
    sbf_list = ["jn", "yn", "h1n", "h2n", "i1n", "i2n", "kn"]
    for sbf in sbf_list:
        values = map(lambda p: reference_value(p[0], p[1], sbf),
                     reference_points())
        with open(sbf + ".pickle", "wb") as f:
            pickle.dump(values, f)
        np.save(sbf + ".npy", np.array(map(mpc_to_np, values),
                                       dtype=np.complex128))
        msg = "Computed {} values!".format(sbf)
        print msg
        logging.info(msg)
