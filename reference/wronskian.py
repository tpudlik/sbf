"""Assess the accuracy of the reference values using the Wronskian relations.

This follows the methodology of Liang-Wu Cai (2011), eqs. (9) and (10). It
should give me more confidence in the reference values.

At the moment, I only implement a single Wronskian relation (between jn and
yn), but once the code is vetted I should implement more.

"""
import cPickle as pickle
from itertools import izip, imap

import numpy as np
import mpmath
from matplotlib import pyplot as plt

from reference_points import reference_points
from reference_values import mpc_to_np
from config import (MAX_ORDER, RADIAL_POINTS, ANGULAR_POINTS, INNER_RADIUS,
                    OUTER_RADIUS, MAX_PRECISION)

def sig_figs_jy(jn, jn1, yn, yn1, z):
    """Check relation http://dlmf.nist.gov/10.50 .

    Parameters
    ----------
        jn : mpf
            The value of j_n(x).
        jn1 : mpf
            The value of j_{n + 1}(x).
        yn : mpf
            The value of y_n(x).
        yn1 : mpf
            The value of y_{n + 1}(x).

    Returns
    -------
    The estimated number of significant digits to which the computation of
    the passed Bessel functions is correct.

    """
    w = mpmath.fabs(z**2*(jn1*yn - jn*yn1) - 1)
    if w > 0:
        return 1 - mpmath.log10(w)
    else:
        # w == 0 to (at least) current working precision.
        return mpmath.mp.dps


def reference_data():
    with open("jn.pickle", "rb") as f:
        jn = pickle.load(f)
    with open("yn.pickle", "rb") as f:
        yn = pickle.load(f)

    return jn, yn


def reference_sig_figs(points, jn, yn):
    """Yield significant figures of jn, yn at point."""
    for idx, p in enumerate(points):
        z, order = p
        if order < MAX_ORDER:
            yield sig_figs_jy(jn[idx], jn[idx+1], yn[idx], yn[idx+1], z)
        else:
            # Can't compute Wronskian, but need to yield something to stay
            # in step with the points generator.
            yield mpmath.nan


def plots_from_generators(pointgen, valgen, title):
    p = point_arrays_from_generator(pointgen)
    v = data_arrays_from_generator(valgen)

    make_plot(p[0], v[0], "{}_real.png".format(title), title)

    for point, value, idx in izip(p[1:], v[1:], xrange(ANGULAR_POINTS)):
        make_plot(point, value, "{}_complex_{}.png".format(title, idx),
                  r"{}, $\exp(2\pi\imath*{}/{})$ line".format(title, idx + 1, ANGULAR_POINTS + 1))


def point_arrays_from_generator(g):
    return np.split(np.array(list(g), dtype=[('z', 'c16'), ('n', 'u8')]), ANGULAR_POINTS + 1)


def data_arrays_from_generator(g):
    return np.split(np.array(list(imap(mpc_to_np, g))), ANGULAR_POINTS + 1)


def make_plot(point, value, filename, title):
    z = np.reshape(point['z'], (RADIAL_POINTS, MAX_ORDER + 1))
    n = np.reshape(point['n'], (RADIAL_POINTS, MAX_ORDER + 1))
    v = np.reshape(value,      (RADIAL_POINTS, MAX_ORDER + 1))
    imdata = np.ma.masked_invalid(v)

    cmap = plt.cm.Greys
    cmap.set_bad('r', 1)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(np.log10(np.abs(z.transpose())), n.transpose(),
                       imdata.transpose(), cmap=cmap, vmin=0, vmax=15)
    plt.colorbar(im)
    ax.set_xlim((INNER_RADIUS, OUTER_RADIUS))
    ax.set_ylim((0, imdata.shape[1]))

    ax.set_xlabel(r"$\log_{10}(|z|)$")
    ax.set_ylabel("order")
    if title:
        ax.set_title(title)
    
    plt.savefig(filename)
    plt.close(fig)


if __name__ == '__main__':
    mpmath.mp.dps = MAX_PRECISION
    jn, yn = reference_data()
    sig_figs = reference_sig_figs(reference_points(), jn, yn)
    plots_from_generators(reference_points(), sig_figs, "jnyn")
