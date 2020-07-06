from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.optimizers.evolutionary.gp import GP
from opytimizer.spaces.tree import TreeSpace


def optimize(target, n_trees, n_terminals, n_variables, n_iterations,
             min_depth, max_depth, functions, lb, ub, hyperparams):
    """Abstracts Opytimizer's Genetic Programming into a single method.

    Args:
        target (callable): The method to be optimized.
        n_trees (int): Number of agents.
        n_terminals (int): Number of terminals
        n_variables (int): Number of variables.
        n_iterations (int): Number of iterations.
        min_depth (int): Minimum depth of trees.
        max_depth (int): Maximum depth of trees.
        functions (list): Functions' nodes.
        lb (list): List of lower bounds.
        ub (list): List of upper bounds.
        hyperparams (dict): Dictionary of hyperparameters.

    Returns:
        A History object containing all optimization's information.

    """

    # Creating the TreeSpace
    space = TreeSpace(n_trees=n_trees, n_terminals=n_terminals, n_variables=n_variables,
                      n_iterations=n_iterations, min_depth=min_depth, max_depth=max_depth,
                      functions=functions, lower_bound=lb, upper_bound=ub)

    # Creating GP's optimizer
    optimizer = GP(hyperparams=hyperparams)

    # Creating the Function
    function = Function(pointer=target)

    # Creating the optimization task
    task = Opytimizer(space=space, optimizer=optimizer, function=function)

    # Initializing task
    history = task.start(store_best_only=True)

    return history