from opytimizer.spaces.tree import TreeSpace


class LossTreeSpace(TreeSpace):
    """LossTreeSpace implements a loss-based version of the tree search space.

    """

    def __init__(self, n_trees=1, n_terminals=1, n_variables=1, n_iterations=10,
                 min_depth=1, max_depth=3, functions=None, lower_bound=(0,), upper_bound=(1,)):
        """Initialization method.

        Args:
            n_trees (int): Number of trees.
            n_terminals (int): Number of terminal nodes.
            n_variables (int): Number of decision variables.
            n_iterations (int): Number of iterations.
            min_depth (int): Minimum depth of the trees.
            max_depth (int): Maximum depth of the trees.
            functions (tuple): Functions nodes.
            lower_bound (tuple): Lower bound tuple with the minimum possible values.
            upper_bound (tuple): Upper bound tuple with the maximum possible values.

        """

        # Override its parent class with the receiving arguments
        super(LossTreeSpace, self).__init__(n_trees, n_terminals, n_variables, n_iterations,
                                            min_depth, max_depth, functions, lower_bound, upper_bound)
