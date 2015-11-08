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
    iterator : Yields tuples of (z, order)

    """
    radii = 10**np.linspace(INNER_RADIUS, OUTER_RADIUS, RADIAL_POINTS,
                            dtype=np.float64)
    phases = np.exp(1j*np.arange(1, ANGULAR_POINTS + 1, dtype=np.float64)
                    *2*np.pi/(ANGULAR_POINTS + 1))
    
    real_points = iter(radii) # Treated separately to prevent small imaginary
                              # parts in the input from leading to difficult-
                              # to-quash small imaginary parts of the output.
    complex_points = iter(np.outer(radii, phases).flatten())
    orders = xrange(MAX_ORDER + 1)
    return itertools.chain(itertools.product(real_points, orders),
                           itertools.product(complex_points, orders))


if __name__ == "__main__":
    with open("reference_points.pickle", "wb") as f:
        pickle.dump(reference_points(), f)
    np.save("reference_points.npy", np.array(list(reference_points()),
                                             dtype=np.complex128))
