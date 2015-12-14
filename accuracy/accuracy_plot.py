import sys
from os import path
from itertools import izip

import numpy as np
from matplotlib import pyplot as plt

# Path hack
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from reference.config import (INNER_RADIUS, OUTER_RADIUS, RADIAL_POINTS,
                              ANGULAR_POINTS, MAX_ORDER)
REFERENCE_DIR = path.join(path.dirname(path.dirname(path.abspath(__file__))),
                          "reference")


def accuracy_plot(f, sbf, atol, rtol):
    """Generates plots illustrating the accuracy of f in approximating the
    named sbf.

    """
    if sbf not in ("jn", "yn", "h1n", "h2n", "i1n", "i2n", "kn"):
        raise ValueError("Unrecorgnized sbf value {}".format(sbf))

    real_points = np.load(path.join(REFERENCE_DIR,
                                    "reference_points_real.npy"))
    complex_points = np.split(np.load(path.join(REFERENCE_DIR,
                                                "reference_points_complex.npy")),
                              ANGULAR_POINTS)

    real_ref_values, complex_ref_values = get_ref_values(sbf)

    real_values = f(real_points['n'], real_points['z'])
    complex_values = [f(x['n'], x['z']) for x in complex_points]

    make_accuracy_plot(real_points, real_values, real_ref_values,
                       atol, rtol, "{}_real.png".format(sbf))
    for point, value, ref_value, idx in izip(complex_points, complex_values,
                                             complex_ref_values,
                                             xrange(ANGULAR_POINTS)):
        make_accuracy_plot(point, value, ref_value, atol, rtol,
                           "{}_complex_{}.png".format(sbf, idx))


def make_accuracy_plot(point, value, reference, atol, rtol, filename):
    z = np.reshape(point['z'], (RADIAL_POINTS, MAX_ORDER + 1))
    n = np.reshape(point['n'], (RADIAL_POINTS, MAX_ORDER + 1))
    error_1D = (np.abs(value - reference) - atol)/(np.abs(reference)*rtol)
    error = np.reshape(np.clip(error_1D, 0, np.inf),
                       (RADIAL_POINTS, MAX_ORDER + 1))
    log_error = np.log10(np.clip(error, 1, np.inf))
    imdata = np.ma.masked_invalid(log_error)

    cmap = plt.cm.Greys
    cmap.set_bad('r', 1)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(np.log10(np.abs(z.transpose())), n.transpose(),
                       imdata.transpose(),
                       cmap=cmap, vmin=0, vmax=5)
    plt.colorbar(im)
    ax.set_xlim((INNER_RADIUS, OUTER_RADIUS))
    ax.set_ylim((0, imdata.shape[1]))
    plt.savefig(filename)
    plt.close(fig)


def get_ref_values(sbf):
    """Return arrays of reference values for sbf at real and complex args."""

    filename = path.join(REFERENCE_DIR, "{}.npy".format(sbf))
    values = np.split(np.load(filename), ANGULAR_POINTS + 1)
    return values[0], values[1:]
