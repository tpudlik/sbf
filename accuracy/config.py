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
RADIAL_POINTS = 10**2

# Number of points to sample in the angular direction
ANGULAR_POINTS = 7

# Largest order to consider
MAX_ORDER = 200

# Initial setting of mpmath.mp.dps for computing reference values.
STARTING_PRECISION = 250

# Step in which precision values are increased if tolerances not met.
PRECISION_STEP = 20

# The base-10 logarithm of the absolute precision to which the two mpmath
# algorithms must agree for a value to be declared correct.
ATOL = -300

# The base-10 logarith of the relative precision to which the two mpmath
# algorithms must agree for a value to be declared correct.
RTOL = -15
