from models.mlp import MLP


def validate_losses(train_iterator, val_iterator, model_obj, n_input, n_hidden, n_classes, lr, epochs, device):
    """Trains a model using the provided loss function and validates it.

    Args:
        train_iterator (torchtext.data.Iterator): Training data iterator.
        val_iterator (torchtext.data.Iterator): Validation data iterator.
        model_obj (Model): A Model-child instance.
        n_input (int): Number of input units.
        n_hidden (int): Number of hidden units.
        n_classes (int): Number of output units.
        lr (float): Learning rate.
        epochs (int): Number of training epochs.
        device (str): Device used to train the model ('cpu' or 'cuda').

    """

    def f(loss):
        """Initializes the model, applies the loss, fits the training data
        and evaluates the validation data.

        Args:
            loss (LossNode): Tree composed of LossNodes.

        Returns:
            1 - validation accuracy.
            
        """

        # Initializing the model
        model = model_obj(n_input=n_input, n_hidden=n_hidden, n_classes=n_classes, lr=lr, init_weights=None, device=device)

        # Gathers the loss function
        model.loss = loss

        # Fits the model
        model.fit(train_iterator, epochs)

        # Evaluates the model
        _, acc = model.evaluate(val_iterator)

        return 1 - acc

    return f
