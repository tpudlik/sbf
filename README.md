# Comparison of spherical Bessel function algorithms #

A number of algorithms exist for computing the values of
[spherical Bessel functions][1] with fixed-precision arithmetic.  This
repository contains implementations of these algorithms, as well as code for
comparing their accuracy over a wide range of parameters.  It's an extension
of the work of [Jabłoński (1994)][2] to complex-valued arguments.

The ultimate objective of this work is to design a new implementation
of spherical Bessel functions for [SciPy][3].

[1]: http://dlmf.nist.gov/10.47
[2]: http://linkinghub.elsevier.com/retrieve/pii/S0021999184710606
[3]: https://github.com/scipy/scipy


## Notation ##

*   Throughout, `x` denotes a real argument and `z` a complex argument.
*   `j_n(z)`: spherical Bessel function of the first kind.
*   `y_n(z)`: spherical Bessel function of the second kind.

## Algorithms ##

I compare the following implementations:

1.  The current SciPy implementation, which uses the algorithms described by
    Zhang and Jin in *Computation of Special Functions* (Wiley, 1996). The
    algorithm uses backward recurrence for `j_n(x)` and `j_n(z)`, a forward
    recurrence for `y_n(x)`, and a recurrence based on the Wronskian relation
    between `j_n(z)` and `y_n(z)` for computing `y_n(z)`. The algorithm for
    `j_n(x)` is reported to be unstable for moderately large values of the
    argument (see [these][4] [issues][5]).  This problem motivates the
    investigation reported here.
2.  An implementation using the SciPy implementations of the Bessel functions,
    based on their connection to the spherical Bessel functions
    ([DLMF 10.47.ii][6]).
3.  An algorithm recently proposed by [Liang-Wu Cai (2011)][7].
4.  The explicit or analytical formulas in terms of trigonometric functions
    ([DLMF 10.49][8]).
5.  Ascending recurrence.
6.  Descending recurrence.
7.  Power series expansion about `z = 0`.

[4]: https://github.com/scipy/scipy/issues/2165
[5]: https://github.com/scipy/scipy/issues/1641
[6]: http://dlmf.nist.gov/10.47.ii
[7]: http://dx.doi.org/10.1016/j.cpc.2010.11.019
[8]: http://dlmf.nist.gov/10.49


## Accuracy tests ##

All of the tested algorithms are used to compute the spherical Bessel
functions at a range of order and argument values, and compared to the output
of variants of algorithms 2 and 4 implemented using the `mpmath` arbitrary-
precision arithmetic library.  The goal is to identify regions of order and
argument space in which each of the fixed-precision algorithms of the previous
section is correct.  Correctness is defined (somewhat arbitrarily, but
following Jabłoński) as matching the first 10 significant figures of the exact
result.

## Unit tests ##

To run the algorithms' unit tests, run `nosetests` in the `sbf` directory.
