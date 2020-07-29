import torch
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


class ConstantLoss(nn.Module):
    """ConstantLoss used to defined constant values that may appear in the GP's trees.

    """

    def __init__(self):
        """Initialization method.

        """

        # Overrides the parent class
        super(ConstantLoss, self).__init__()

        # Defines a random uniform value between `0` and `1`
        self.value = torch.tensor(torch.rand(1), requires_grad=True)

    def forward(self, y_preds, y_true):
        """Forward pass.

        Args:
            y_preds (torch.Tensor): Predictions.
            y_true (torch.Tensor): True labels.

        Returns:
            The constant value between `0` and `1`.

        """
        
        return self.value
        

    def __str__(self):
        """String representation.

        """

        return str(round(self.value.item(), 4))
