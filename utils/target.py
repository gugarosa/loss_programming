from models.mlp import MLP


def loss_function(train_iterator, val_iterator, epochs=10):
    """

    Args:

    """

    def f(w):
        """

        Args:
            w (float): Array of variables.

        Returns:
            

        """

        # Initializing the model
        model = MLP()

        #
        model.fit(train_iterator, w, epochs)

        #
        eval_loss = model.evaluate(val_iterator, w)

        return eval_loss

    return f
