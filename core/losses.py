from torch import nn
import opytimizer.math.random as r


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """
    """

    def __init__(self):
        """
        """

        super(CrossEntropyLoss, self).__init__()

    def __str__(self):
        """
        """

        return 'CE'

class ConstantLoss:
    """
    """

    def __init__(self):
        """
        """

        self.value = r.generate_uniform_random_number(0, 1)[0]

    def __call__(self, *args):
        """
        """

        return self.value

    def __str__(self):
        """
        """

        return str(round(self.value, 4))
