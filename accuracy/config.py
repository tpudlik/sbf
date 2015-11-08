"""Config file for setting the dimensions of the region on which the
spherical Bessel function values will be compared.

This region is an annulus in the complex plane, uniformly sampled with 
respect to angle and the *logarithm* of the radius.

"""

# Decimal logarithm of the inner radius; default -3
INNER_RADIUS = -3

# Decimal logarithm of the outer radius; default 4
OUTER_RADIUS = 4

# Number of points to sample in the radial direction; default 100
RADIAL_POINTS = 10**1

# Number of points to sample in the angular direction; default 5
ANGULAR_POINTS = 2

# Largest order to consider; default 200
MAX_ORDER = 200

# Initial setting of mpmath.mp.dps for computing reference values; default 256
STARTING_PRECISION = 256

# Maximum precision to attempt before giving up; default 2000
MAX_PRECISION = 2000

# The base-10 logarithm of the absolute precision to which the two mpmath
# algorithms must agree for a value to be declared correct. Default -300.
ATOL = -50

# The base-10 logarith of the relative precision to which the two mpmath
# algorithms must agree for a value to be declared correct. Default -15.
RTOL = -10
