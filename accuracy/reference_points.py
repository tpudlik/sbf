import numpy as np
from accuracy.config import (INNER_RADIUS, OUTER_RADIUS, RADIAL_POINTS,
                                 ANGULAR_POINTS, MAX_ORDER)

def reference_points():
    """Returns an ndarray of points at which the spherical Bessel function
    algorithms will be compared.

    """
    radii = 10**np.linspace(INNER_RADIUS, OUTER_RADIUS, RADIAL_POINTS)
    angles = np.linspace(0, 2*np.pi, ANGULAR_POINTS, endpoint=False)
    return np.outer(radii, np.exp(1j*angles))[:, :, np.newaxis]

def reference_orders():
    return np.arange(MAX_ORDER)[np.newaxis, np.newaxis, :]
