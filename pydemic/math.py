def numeric_derivative(func, epsilon=1e-6):
    """
    Return a function that computes the numeric first-order derivative of func.
    """

    def diff(x):
        return (func(x + epsilon) - func(x)) / epsilon

    return diff
