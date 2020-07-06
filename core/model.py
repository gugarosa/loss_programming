import torch
from torch import nn, optim
from tqdm import tqdm

from core.losses import WeightedLossAccuracy


class Model(torch.nn.Module):
    """A Model class is responsible for customly implementing neural network architectures.

    One can configure, if necessary, different properties or methods that
    can be used throughout all childs.

    """

    def __init__(self, init_weights=None, device='cpu'):
        """Initialization method.

        Args:
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        # Override its parent class
        super(Model, self).__init__()

        # Creates the initialization weights property
        self.init_weights = init_weights

        # Creates a cpu-based device property
        self.device = device

        # Setting default tensor type to float
        torch.set_default_tensor_type(torch.FloatTensor)

    def _compile(self, lr=0.001):
        """Compiles the network by setting its optimizer, loss function and additional properties.

        Args:
            lr (float): Learning rate.

        """

        # Defining an optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Defines the loss as usual
        self.loss = WeightedLossAccuracy()

        # Check if there is a tuple for the weights initialization
        if self.init_weights:
            # Iterate over all possible parameters
            for _, p in self.named_parameters():
                # Initializes with a uniform distributed value
                nn.init.uniform_(p.data, init_weights[0], init_weights[1])

        # Checks if GPU is avaliable
        if torch.cuda.is_available() and self.device == 'cuda':
            # Uses CUDA in the whole class
            self.cuda()

    def step(self, batch, weights, is_training=True):
        """Performs a single batch optimization step.

        Args:
            batch (tuple): Tuple containing the batches input (x) and target (y).
            is_training (bool): Whether it is a training step or not.

        Returns:
            

        """

        # Gathers the batch's input and target
        x, y = batch[0], batch[1]

        # Calculate the predictions based on inputs
        preds = self(x)

        # Calculates the batch's loss
        batch_loss = self.loss(y, preds, weights)

        # Checks if it is a training batch
        if is_training:
            # Propagates the gradients backward
            batch_loss.backward()

            # Perform the parameeters updates
            self.optimizer.step()

        return batch_loss.item()

    def fit(self, train_iterator, weights, epochs=10):
        """Trains the model.

        Args:
            train_iterator (torchtext.data.Iterator): Training data iterator.
            
            epochs (int): The maximum number of training epochs.

        """

        print('Fitting model ...')

        # Iterate through all epochs
        for e in range(epochs):
            print(f'Epoch {e+1}/{epochs}')

            # Setting the training flag
            self.train()

            # Initializes the loss as zero
            mean_loss = 0.0

            # For every batch in the iterator
            for batch in train_iterator:
                # Resetting the gradients
                self.optimizer.zero_grad()

                # Calculates the batch's loss
                loss = self.step(batch, weights)

                # Summing up batch's loss
                mean_loss += loss

            # Gets the mean loss across all batches
            mean_loss /= len(train_iterator)

            print(f'train_loss: {mean_loss}')

    def evaluate(self, iterator, weights):
        """Evaluates the model.

        Args:
            iterator (torchtext.data.Iterator): Validation or testing data iterator.

        """

        print(f'Evaluating model ...')

        # Setting the evalution flag
        self.eval()

        # Initializes the loss as zero
        mean_loss = 0.0

        # Inhibits the gradient from updating the parameters
        with torch.no_grad():
            # For every batch in the iterator
            for batch in iterator:
                # Calculates the batch's loss
                loss = self.step(batch, weights, is_training=False)

                # Summing up batch's loss
                mean_loss += loss

        # Gets the mean loss across all batches
        mean_loss /= len(iterator)

        print(f'eval_loss: {mean_loss}')

        return mean_loss