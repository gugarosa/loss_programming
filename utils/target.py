import torch


def _step(model, batch, w):
    """Performs a single batch optimization step.

    Args:
        batch (tuple): Tuple containing the batches input (x) and target (y).

    Returns:
        The loss and accuracy accross the batch.

    """

    # Gathers the batch's input and target
    x, y = batch[0], batch[1]

    # Calculate the predictions based on inputs
    preds = model(x)

    # Calculates the batch's loss
    loss = model.loss(preds, y)

    #
    acc = torch.mean((torch.sum(torch.argmax(preds, dim=1) == y).float()) / x.size(0))

    #
    batch_loss = w[0][0] * loss + w[1][0] * (1 - acc)

    # Propagates the gradients backward
    batch_loss.backward()

    # Perform the parameeters updates
    model.optimizer.step()

    return batch_loss.item()


def create_loss_function(model, train_iterator, epochs):
    """

    Args:

    """

    def f(w):
        """

        Args:
            w (float): Array of variables.

        Returns:
            

        """

        print('Fitting model ...')

        # Iterate through all epochs
        for e in range(epochs):
            print(f'Epoch {e+1}/{epochs}')

            # Setting the training flag
            model.train()

            # Initializes the loss as zero
            mean_loss = 0.0

            # For every batch in the iterator
            for batch in train_iterator:
                # Resetting the gradients
                model.optimizer.zero_grad()

                # Calculates the batch's loss
                loss = _step(model, batch, w)

                # Summing up batch's loss
                mean_loss += loss

            # Gets the mean loss across all batches
            mean_loss /= len(train_iterator)

            print(f'loss: {mean_loss}')

        return mean_loss

    return f
