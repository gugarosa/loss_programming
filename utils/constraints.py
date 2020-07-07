def sum_equals_one(x):
    """Defines a constraint where variables sum should equal one.

    Args:
        x (np.array): Array of input values.

    Returns:
        Whether constraint is valid or not.

    """

    return x[0] + x[1] == 1