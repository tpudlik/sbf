"""Spherical Bessel function algorithms based on ordinary Bessel functions.

These algorithms compute the spherical Bessel functions by calling the
scipy Bessel function routines, exploiting the identities
http://dlmf.nist.gov/10.47.ii .

"""
import numpy as np
import scipy.special

def sph_jn(n, z):
    return np.sqrt(0.5*np.pi/z)*scipy.special.jv(n + 0.5, z)

def sph_yn(n, z):
    return np.sqrt(0.5*np.pi/z)*scipy.special.yv(n + 0.5, z)

def sph_h1n(n, z):
    return np.sqrt(0.5*np.pi/z)*scipy.special.hankel1(n + 0.5, z)

def sph_h2n(n, z):
    return np.sqrt(0.5*np.pi/z)*scipy.special.hankel2(n + 0.5, z)

def sph_i1n(n, z):
    return np.sqrt(0.5*np.pi/z)*scipy.special.iv(n + 0.5, z)

def sph_i2n(n, z):
    return np.sqrt(0.5*np.pi/z)*scipy.special.iv(-n - 0.5, z)

def sph_kn(n, z):
    return np.sqrt(0.5*np.pi/z)*scipy.special.kv(n + 0.5, z)
