import torch
from torch import nn


class WeightedLossAccuracy:
    """
    """

    def __init__(self):
        """
        """

        #
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, y_true, y_pred, weights):
        #
        loss = self.loss(y_pred, y_true)

        #
        acc = torch.mean((torch.sum(torch.argmax(y_pred, dim=1) == y_true).float()) / y_true.size(0))

        return loss