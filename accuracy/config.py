"""Config file for setting the dimensions of the region on which the
spherical Bessel function values will be compared.

This region is an annulus in the complex plane, uniformly sampled with 
respect to angle and the *logarithm* of the radius.

"""

# Decimal logarithm of the inner radius
INNER_RADIUS = -3

# Decimal logarithm of the outer radius
OUTER_RADIUS = 4

# Number of points to sample in the radial direction
RADIAL_POINTS = 10**1

# Number of points to sample in the angular direction
ANGULAR_POINTS = 3

# Largest order to consider
MAX_ORDER = 10

# The number of significant figures that should agree between the two
# mpmath algorithms for a value to be declared correct.  Note that the
# precision of a double is not guaranteed to be greater than 15.
TARGET_PRECISION = 13