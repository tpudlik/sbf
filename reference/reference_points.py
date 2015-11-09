"""Generate (z, n) values.

Real arguments are reated separately to prevent small imaginary parts in the
input from leading to difficult-to-quash small imaginary parts of the output.

The output is saved as two one-dimensional numpy arrays:

    reference_points_real : [('z', 'f8'), ('n', 'u8')]
        Contains RADIAL_POINTS * (MAX_ORDER + 1) elements, with the 
        MAX_ORDER + 1 values of n iterated over first.
    reference_points_complex : [('z', 'c16'), ('n', 'u8')]
        Contains RADIAL_POINTS * (MAX_ORDER + 1) * ANGULAR_POINTS elements,
        with the MAX_ORDER + 1 values of n iterated over first, then the
        RADIAL_POINTS values of |z|, and finally the ANGULAR_POINTS values
        of arg z.

"""

import itertools, sys
from os import path
import cPickle as pickle

import numpy as np

# Path hack
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from reference.config import (INNER_RADIUS, OUTER_RADIUS, RADIAL_POINTS,
                              ANGULAR_POINTS, MAX_ORDER)

def reference_points():
    """Returns an iterable of (point, order) pairs.

    These points in C x Z will be used for computing reference values of the
    spherical Bessel functions.

    Returns
    -------
    iterator : Yields tuples of (z, order).  Real values occur before any
        imaginary ones.  The total number of values returned is
        RADIAL_POINTS * (MAX_ORDER + 1) * (ANGULAR_POINTS + 1).

    """
    return itertools.chain(reference_points_real(),reference_points_complex())


def reference_points_real():
    return itertools.product(iter(radius_array()), xrange(MAX_ORDER + 1))


def reference_points_complex():
    radii = radius_array()
    phases = np.exp(1j*np.arange(1, ANGULAR_POINTS + 1, dtype=np.complex128)
                    *2*np.pi/(ANGULAR_POINTS + 1))
    complex_points = iter(np.outer(radii, phases).transpose().flatten())
    # The transpose above guarantees that radii are iterated over first,
    # before phases.
    orders = xrange(MAX_ORDER + 1)
    return itertools.product(complex_points, orders)


def radius_array():
    return 10**np.linspace(INNER_RADIUS, OUTER_RADIUS, RADIAL_POINTS,
                           dtype=np.float64)


if __name__ == "__main__":
    np.save("reference_points_real.npy",
            np.array(list(reference_points_real()),
                     dtype=[('z', 'f8'), ('n', 'u8')]))
    np.save("reference_points_complex.npy",
            np.array(list(reference_points_complex()),
                     dtype=[('z', 'c16'), ('n', 'u8')]))
