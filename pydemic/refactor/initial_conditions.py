class IC:
    """
    Basic class that represents initial conditions to models
    """

    def __init__(self, state_variables):
        self.state_variables = tuple(state_variables)
