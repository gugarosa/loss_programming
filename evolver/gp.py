from opytimizer.optimizers.evolutionary.gp import GP


class LossGP(GP):
    """LossGP implements a loss-based version of the Genetic Programming.

    """

    def __init__(self, algorithm='LossGP', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving arguments
        super(LossGP, self).__init__(algorithm, hyperparams)