import argparse, cProfile, importlib, itertools, pstats, sys
from os import path

import numpy as np

# Path hack
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )


def profile(f, args):
    """Profile the sbf function f."""
    z = 10**np.linspace(-3, 4, 1000)
    n = np.arange(200)
    zz, nn = np.meshgrid(z, n)
    fnames = get_input_filenames(args)

    cProfile.runctx("f(nn, zz)", globals(), locals(), fnames[0])

    phases = np.exp(2*np.pi*np.random.rand(zz.shape[0], zz.shape[1]))
    zz = zz*phases
    fname = "{}_{}_complex.pstats".format(args.sbf, args.algo)
    cProfile.runctx("f(nn, zz)", globals(), locals(), fnames[1])


def get_input_filenames(args):
    return ("{}_{}_real.pstats".format(args.sbf, args.algo),
            "{}_{}_complex.pstats".format(args.sbf, args.algo))


def get_output_filenames(args):
    return ("{}_{}_real.txt".format(args.sbf, args.algo),
            "{}_{}_complex.txt".format(args.sbf, args.algo))


def print_stats(args):
    f_ins = get_input_filenames(args)
    f_outs = get_output_filenames(args)

    for f_in, f_out in itertools.izip(f_ins, f_outs):
        with open(f_out, "w") as f:
            p = pstats.Stats(f_in, stream=f)
            p.strip_dirs().sort_stats("cumulative").print_stats(50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sbf",
                        help="The spherical Bessel function to profile.",
                        choices=["jn", "yn", "h1n", "h2n", "i1n", "i2n", "kn"])
    parser.add_argument("algo",
                        help="The implementation to profile.",
                        choices=["default", "bessel", "a_recur", "cai",
                                 "power_series", "d_recur_miller",
                                 "candidate"])
    args = parser.parse_args()

    m = importlib.import_module("algos.{}".format(args.algo))
    f = getattr(m, "sph_{}".format(args.sbf))

    profile(f, args)
    print_stats(args)
