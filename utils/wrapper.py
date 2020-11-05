from opytimizer import Opytimizer
from opytimizer.core.function import Function

from core.linker.gp import LossGP
from core.linker.space import LossTreeSpace


def run(target, n_trees, n_terminals, n_iterations,
        min_depth, max_depth, functions, hyperparams=None):
    """Abstracts Opytimizer's loss-based Genetic Programming into a single method.

    Args:
        target (callable): The method to be optimized.
        n_trees (int): Number of agents.
        n_terminals (int): Number of terminals
        n_iterations (int): Number of iterations.
        min_depth (int): Minimum depth of trees.
        max_depth (int): Maximum depth of trees.
        functions (list): Functions' nodes.
        hyperparams (dict): Dictionary of hyperparameters.

    Returns:
        A History object containing all optimization's information.

    """

    # Creating a loss-based TreeSpace
    space = LossTreeSpace(n_trees=n_trees, n_terminals=n_terminals, n_iterations=n_iterations,
                          min_depth=min_depth, max_depth=max_depth, functions=functions)

    # Creating a loss-based GP optimizer
    optimizer = LossGP(hyperparams=hyperparams)

    # Creating the Function
    function = Function(pointer=target)

    # Creating the optimization task
    task = Opytimizer(space=space, optimizer=optimizer, function=function)

    # Initializing task
    history = task.start()

    return history
