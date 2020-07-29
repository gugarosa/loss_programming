import opytimizer.math.random as r
from torch import nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss wrapper around the PyTorch's loss function.

    """

    def __init__(self):
        """Initialization method.

        """

        # Overrides the parent class
        super(CrossEntropyLoss, self).__init__()

    def __str__(self):
        """String representation.

        """

        return 'CE'

class ConstantLoss:
    """ConstantLoss used to defined constant values that may appear in the GP's trees.

    """

    def __init__(self):
        """Initialization method.
        
        """

        # Defines a random uniform value between `0` and `1`
        self.value = r.generate_uniform_random_number(0, 1)[0]

    def __call__(self, *args):
        """Callable whenever this class is called.

        """

        return self.value

    def __str__(self):
        """String representation.
        
        """

        return str(round(self.value, 4))
