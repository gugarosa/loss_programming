from models.mlp import MLP


def validate_losses(train_iterator, val_iterator, epochs=10):
    """Trains a model using the provided loss function and validates it.

    Args:
        train_iterator (torchtext.data.Iterator): Training data iterator.
        val_iterator (torchtext.data.Iterator): Validation data iterator.
        epochs (int): Number of training epochs.

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
        model = MLP()

        # Gathers the loss function
        # model.loss = loss

        # Fits the model
        model.fit(train_iterator, epochs)

        # Evaluates the model
        eval_loss = model.evaluate(val_iterator)

        return eval_loss

    return f
