import numpy as np


# -------------------------------------------------
# Sparse 4 by 4 Joint PMF
# -------------------------------------------------

# Joint PMF table
pmf_table = np.array([
    [0.10, 0.05, 0.00, 0.00],
    [0.15, 0.20, 0.05, 0.00],
    [0.00, 0.10, 0.15, 0.05],
    [0.00, 0.00, 0.05, 0.10]
])


def joint_pmf(x, y):
    """
    Joint PMF table:

             y=0   y=1   y=2   y=3
    x=0      0.10  0.05  0.00  0.00
    x=1      0.15  0.20  0.05  0.00
    x=2      0.00  0.10  0.15  0.05
    x=3      0.00  0.00  0.05  0.10
    """

    # Return 0 for invalid indices
    if x < 0 or x > 3 or y < 0 or y > 3:
        return 0.0

    return pmf_table[x][y]


def marginal_px(x):
    """
    Compute PX(x) by summing joint_pmf(x, y) over y = 0,1,2,3.
    """
    total = 0

    for y in range(4):
        total += joint_pmf(x, y)

    return total


def marginal_py(y):
    """
    Compute PY(y) by summing joint_pmf(x, y) over x = 0,1,2,3.
    """
    total = 0

    for x in range(4):
        total += joint_pmf(x, y)

    return total


def conditional_pmf_x_given_y(x, y):
    """
    Compute P(X=x given Y=y).

    P(X=x given Y=y) = joint_pmf(x,y) / PY(y)

    If PY(y) is zero, return 0.
    """
    py = marginal_py(y)

    if py == 0:
        return 0

    return joint_pmf(x, y) / py


def conditional_distribution_x_given_y(y):
    """
    Return conditional distribution of X given Y=y
    as dictionary.
    """
    distribution = {}

    for x in range(4):
        distribution[x] = conditional_pmf_x_given_y(x, y)

    return distribution


def probability_sum_greater_than_3():
    """
    Compute P(X + Y > 3).
    """
    probability = 0

    for x in range(4):
        for y in range(4):
            if x + y > 3:
                probability += joint_pmf(x, y)

    return probability


def independence_check():
    """
    Return True if X and Y are independent.
    """
    for x in range(4):
        for y in range(4):

            left = joint_pmf(x, y)
            right = marginal_px(x) * marginal_py(y)

            if not np.isclose(left, right):
                return False

    return True


# -------------------------------------------------
# Expectation, Covariance, and Correlation
# -------------------------------------------------

def expected_x():
    """
    Compute E[X].
    """
    expectation = 0

    for x in range(4):
        expectation += x * marginal_px(x)

    return expectation


def expected_y():
    """
    Compute E[Y].
    """
    expectation = 0

    for y in range(4):
        expectation += y * marginal_py(y)

    return expectation


def expected_xy():
    """
    Compute E[XY].
    """
    expectation = 0

    for x in range(4):
        for y in range(4):
            expectation += x * y * joint_pmf(x, y)

    return expectation


def variance_x():
    """
    Compute Var(X).
    """
    ex = expected_x()

    ex2 = 0

    for x in range(4):
        ex2 += (x ** 2) * marginal_px(x)

    return ex2 - (ex ** 2)


def variance_y():
    """
    Compute Var(Y).
    """
    ey = expected_y()

    ey2 = 0

    for y in range(4):
        ey2 += (y ** 2) * marginal_py(y)

    return ey2 - (ey ** 2)


def covariance_xy():
    """
    Compute Cov(X,Y).

    Cov(X,Y) = E[XY] - E[X]*E[Y]
    """
    return expected_xy() - (expected_x() * expected_y())


def correlation_xy():
    """
    Compute correlation coefficient:

    rho_XY = Cov(X,Y) / sqrt( Var(X) * Var(Y) )
    """
    cov = covariance_xy()
    varx = variance_x()
    vary = variance_y()

    return cov / np.sqrt(varx * vary)


def variance_sum():
    """
    Compute Var(X+Y).
    """
    expected_sum = 0
    expected_sum_sq = 0

    for x in range(4):
        for y in range(4):

            value = x + y

            expected_sum += value * joint_pmf(x, y)
            expected_sum_sq += (value ** 2) * joint_pmf(x, y)

    return expected_sum_sq - (expected_sum ** 2)


def variance_identity_check():
    """
    Verify:

    Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y)

    Return True if the identity holds, else False.
    """

    left = variance_sum()

    right = (
        variance_x()
        + variance_y()
        + 2 * covariance_xy()
    )

    # Convert numpy bool to normal Python bool
    return bool(np.isclose(left, right))

