"""Unit tests of the sbf_mp module.

These tests verify that the expressions for the Bessel functions were
correctly implemented, not their accuracy.

"""

import unittest
import numpy as np
from mpmath import mpc, pi
from numpy.testing import assert_allclose
from sbf.algos.sbf_mp import (sph_jn_exact, sph_yn_exact, sph_h1n_exact,
                              sph_h2n_exact, sph_i1n_exact, sph_i2n_exact,
                              sph_kn_exact, sph_jn_bessel, sph_yn_bessel,
                              sph_h1n_bessel, sph_h2n_bessel, sph_i1n_bessel,
                              sph_i2n_bessel, sph_kn_bessel)


class TestExactSmallOrder(unittest.TestCase):
    """Test the exact small-order expressions http://dlmf.nist.gov/10.49.i """

    def test_sph_j0_exact(self):
        x = 52
        desired = np.sin(x)/x
        actual = np.complex(sph_jn_exact(0, x))
        assert_allclose(actual, desired)

    def test_sph_j1_exact(self):
        x = 1.3
        desired = np.sin(x)/x**2 - np.cos(x)/x
        actual = np.complex(sph_jn_exact(1, x))
        assert_allclose(actual, desired)

    def test_sph_j2_exact(self):
        x = 24.5
        desired = (-1.0/x + 3/x**3)*np.sin(x) - 3/x**2*np.cos(x)
        actual = np.complex(sph_jn_exact(2, x))
        assert_allclose(actual, desired)

    def test_sph_y0_exact(self):
        x = 4.3
        desired = -np.cos(x)/x
        actual = np.complex(sph_yn_exact(0, x))
        assert_allclose(actual, desired)

    def test_sph_y1_exact(self):
        x = 29.2
        desired = -np.cos(x)/x**2 - np.sin(x)/x
        actual = np.complex(sph_yn_exact(1, x))
        assert_allclose(actual, desired)

    def test_sph_y2_exact(self):
        x = 0.321
        desired = (1.0/x - 3/x**3)*np.cos(x) - 3/x**2*np.sin(x)
        actual = np.complex(sph_yn_exact(2, x))
        assert_allclose(actual, desired)

    def test_sph_i10_exact(self):
        x = 62.3
        desired = np.sinh(x)/x
        actual = np.complex(sph_i1n_exact(0, x))
        assert_allclose(actual, desired)

    def test_sph_i11_exact(self):
        x = 6.802805007381322
        desired = -np.sinh(x)/x**2 + np.cosh(x)/x
        actual = np.complex(sph_i1n_exact(1, x))
        assert_allclose(actual, desired)

    def test_sph_i12_exact(self):
        x = 9.810801919951345
        desired = (1/x + 3/x**3)*np.sinh(x) - 3/x**2*np.cosh(x)
        actual = np.complex(sph_i1n_exact(2, x))
        assert_allclose(actual, desired)

    def test_sph_i20_exact(self):
        x = 5.807556326907072
        desired = np.cosh(x)/x
        actual = np.complex(sph_i2n_exact(0, x))
        assert_allclose(actual, desired)

    def test_sph_i21_exact(self):
        x = 11.070885074673603
        desired = -np.cosh(x)/x**2 + np.sinh(x)/x
        actual = np.complex(sph_i2n_exact(1, x))
        assert_allclose(actual, desired)

    def test_sph_i22_exact(self):
        x = 6.916886515975276
        desired = (1/x + 3/x**3)*np.cosh(x) - 3/x**2*np.sinh(x)
        actual = np.complex(sph_i2n_exact(2, x))
        assert_allclose(actual, desired)

    def test_sph_k0_exact(self):
        x = 7.547480927635526
        desired = np.pi*np.exp(-x)/(2*x)
        actual = np.complex(sph_kn_exact(0, x))
        assert_allclose(actual, desired)

    def test_sph_k1_exact(self):
        x = 8.853399925289201
        desired = (1/x + 1/x**2)*np.exp(-x)*np.pi/2
        actual = np.complex(sph_kn_exact(1, x))
        assert_allclose(actual, desired)

    def test_sph_k2_exact(self):
        x = 12.806013305135048
        desired = (1/x + 3/x**2 + 3/x**3)*np.exp(-x)*np.pi/2
        actual = np.complex(sph_kn_exact(2, x))
        assert_allclose(actual, desired)


class TestBesselSmallOrder(unittest.TestCase):
    """Test the exact small-order expressions http://dlmf.nist.gov/10.49.i """

    def test_sph_j0_bessel(self):
        x = 52
        desired = np.sin(x)/x
        actual = np.complex(sph_jn_bessel(0, x))
        assert_allclose(actual, desired)

    def test_sph_j1_bessel(self):
        x = 1.3
        desired = np.sin(x)/x**2 - np.cos(x)/x
        actual = np.complex(sph_jn_bessel(1, x))
        assert_allclose(actual, desired)

    def test_sph_j2_bessel(self):
        x = 24.5
        desired = (-1.0/x + 3/x**3)*np.sin(x) - 3/x**2*np.cos(x)
        actual = np.complex(sph_jn_bessel(2, x))
        assert_allclose(actual, desired)

    def test_sph_y0_bessel(self):
        x = 4.3
        desired = -np.cos(x)/x
        actual = np.complex(sph_yn_bessel(0, x))
        assert_allclose(actual, desired)

    def test_sph_y1_bessel(self):
        x = 29.2
        desired = -np.cos(x)/x**2 - np.sin(x)/x
        actual = np.complex(sph_yn_bessel(1, x))
        assert_allclose(actual, desired)

    def test_sph_y2_bessel(self):
        x = 0.321
        desired = (1.0/x - 3/x**3)*np.cos(x) - 3/x**2*np.sin(x)
        actual = np.complex(sph_yn_bessel(2, x))
        assert_allclose(actual, desired)

    def test_sph_i10_bessel(self):
        x = 62.3
        desired = np.sinh(x)/x
        actual = np.complex(sph_i1n_bessel(0, x))
        assert_allclose(actual, desired)

    def test_sph_i11_bessel(self):
        x = 6.802805007381322
        desired = -np.sinh(x)/x**2 + np.cosh(x)/x
        actual = np.complex(sph_i1n_bessel(1, x))
        assert_allclose(actual, desired)

    def test_sph_i12_bessel(self):
        x = 9.810801919951345
        desired = (1/x + 3/x**3)*np.sinh(x) - 3/x**2*np.cosh(x)
        actual = np.complex(sph_i1n_bessel(2, x))
        assert_allclose(actual, desired)

    def test_sph_i20_bessel(self):
        x = 5.807556326907072
        desired = np.cosh(x)/x
        actual = np.complex(sph_i2n_bessel(0, x))
        assert_allclose(actual, desired)

    def test_sph_i21_bessel(self):
        x = 11.070885074673603
        desired = -np.cosh(x)/x**2 + np.sinh(x)/x
        actual = np.complex(sph_i2n_bessel(1, x))
        assert_allclose(actual, desired)

    def test_sph_i22_bessel(self):
        x = 6.916886515975276
        desired = (1/x + 3/x**3)*np.cosh(x) - 3/x**2*np.sinh(x)
        actual = np.complex(sph_i2n_bessel(2, x))
        assert_allclose(actual, desired)

    def test_sph_k0_bessel(self):
        x = 7.547480927635526
        desired = np.pi*np.exp(-x)/(2*x)
        actual = np.complex(sph_kn_bessel(0, x))
        assert_allclose(actual, desired)

    def test_sph_k1_bessel(self):
        x = 8.853399925289201
        desired = (1/x + 1/x**2)*np.exp(-x)*np.pi/2
        actual = np.complex(sph_kn_bessel(1, x))
        assert_allclose(actual, desired)

    def test_sph_k2_bessel(self):
        x = 12.806013305135048
        desired = (1/x + 3/x**2 + 3/x**3)*np.exp(-x)*np.pi/2
        actual = np.complex(sph_kn_bessel(2, x))
        assert_allclose(actual, desired)


class TestExactInterrelations(unittest.TestCase):
    """Test the relations http://dlmf.nist.gov/10.47.iv ."""

    def test_interrelations_h1_exact(self):
        n = 6
        x = 8.160875740148962
        left = np.complex(sph_h1n_exact(n, x))
        right = np.complex(sph_jn_exact(n, x) + 1j*sph_yn_exact(n, x))
        assert_allclose(left, right)

    def test_interrelations_h2_exact(self):
        n = 5
        x = 2.826012588052626
        left = np.complex(sph_h2n_exact(n, x))
        right = np.complex(sph_jn_exact(n, x) - 1j*sph_yn_exact(n, x))
        assert_allclose(left, right)

    def test_interrelations_k_exact(self):
        n = 7
        x = 5.746161870059113
        left = np.complex(sph_kn_exact(n, x))
        right = np.complex((-1)**(n + 1)*pi/2*(
                            sph_i1n_exact(n, x) - sph_i2n_exact(n, x)))
        assert_allclose(left, right)

    def test_interrelations_i1_exact(self):
        n = 3
        x = 1.9513017144556937
        left = np.complex(sph_i1n_exact(n, x))
        right = np.complex(mpc(0,1)**(-n)*sph_jn_exact(n, 1j*x))
        assert_allclose(left, right)

    def test_interrelations_i2_exact(self):
        n = 9
        x = 6.083765036260925
        left = np.complex(sph_i2n_exact(n, x))
        right = np.complex(mpc(0,1)**(-n-1)*sph_yn_exact(n, 1j*x))
        assert_allclose(left, right)

    def test_interrelations_kh1_exact(self):
        n = 4
        x = 4.403387573037941
        left = np.complex(sph_kn_exact(n, x))
        right = np.complex(-pi/2*mpc(0,1)**n*sph_h1n_exact(n, 1j*x))
        assert_allclose(left, right)

    def test_interrelations_kh2_exact(self):
        n = 8
        x = 1.464478579484278
        left = np.complex(sph_kn_exact(n, x))
        right = np.complex(-pi/2*mpc(0,1)**(-n)*sph_h2n_exact(n, -1j*x))
        assert_allclose(left, right)


class TestBesselInterrelations(unittest.TestCase):
    """Test the relations http://dlmf.nist.gov/10.47.iv ."""

    def test_interrelations_h1_bessel(self):
        n = 6
        x = 8.160875740148962
        left = np.complex(sph_h1n_bessel(n, x))
        right = np.complex(sph_jn_bessel(n, x) + 1j*sph_yn_bessel(n, x))
        assert_allclose(left, right)

    def test_interrelations_h2_bessel(self):
        n = 5
        x = 2.826012588052626
        left = np.complex(sph_h2n_bessel(n, x))
        right = np.complex(sph_jn_bessel(n, x) - 1j*sph_yn_bessel(n, x))
        assert_allclose(left, right)

    def test_interrelations_k_bessel(self):
        n = 7
        x = 5.746161870059113
        left = np.complex(sph_kn_bessel(n, x))
        right = np.complex((-1)**(n + 1)*pi/2*(
                            sph_i1n_bessel(n, x) - sph_i2n_bessel(n, x)))
        assert_allclose(left, right)

    def test_interrelations_i1_bessel(self):
        n = 3
        x = 1.9513017144556937
        left = np.complex(sph_i1n_bessel(n, x))
        right = np.complex(mpc(0,1)**(-n)*sph_jn_bessel(n, 1j*x))
        assert_allclose(left, right)

    def test_interrelations_i2_bessel(self):
        n = 9
        x = 6.083765036260925
        left = np.complex(sph_i2n_bessel(n, x))
        right = np.complex(mpc(0,1)**(-n-1)*sph_yn_bessel(n, 1j*x))
        assert_allclose(left, right)

    def test_interrelations_kh1_bessel(self):
        n = 4
        x = 4.403387573037941
        left = np.complex(sph_kn_bessel(n, x))
        right = np.complex(-pi/2*mpc(0,1)**n*sph_h1n_bessel(n, 1j*x))
        assert_allclose(left, right)

    def test_interrelations_kh2_bessel(self):
        n = 8
        x = 1.464478579484278
        left = np.complex(sph_kn_bessel(n, x))
        right = np.complex(-pi/2*mpc(0,1)**(-n)*sph_h2n_bessel(n, -1j*x))
        assert_allclose(left, right)


