import copy

import numpy as np
import opytimizer.math.general as g
import opytimizer.utils.decorator as d
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
import torch
from opytimizer.optimizers.evolutionary.gp import GP
from tqdm import tqdm

logger = l.get_logger(__name__)


class LossGP(GP):
    """LossGP implements a loss-based version of the Genetic Programming.

    """

    def __init__(self, algorithm='LossGP', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving arguments
        super(LossGP, self).__init__(algorithm, hyperparams)

    def _reproduction(self, space):
        """Reproducts a number of individuals pre-selected through a tournament procedure (p. 99).

        Args:
            space (TreeSpace): A TreeSpace object.

        """

        # Calculates a list of current trees' fitness
        fitness = [fit for fit in space.fits]

        # Number of individuals to be reproducted
        n_individuals = int(space.n_trees * self.p_reproduction)

        # Gathers a list of selected individuals to be replaced
        selected = g.tournament_selection(fitness, n_individuals)

        # For every selected individual
        for s in selected:
            # Gathers the worst individual index
            worst = np.argmax(fitness)

            # Replace the individual by performing a deep copy on selected tree
            space.trees[worst] = copy.deepcopy(space.trees[s])

            # Replaces the worst individual fitness with a minimum value
            fitness[worst] = 0

    def _mutation(self, space):
        """Mutates a number of individuals pre-selected through a tournament procedure.

        Args:
            space (TreeSpace): A TreeSpace object.

        """

        # Calculates a list of current trees' fitness
        fitness = [fit for fit in space.fits]

        # Number of individuals to be mutated
        n_individuals = int(space.n_trees * self.p_mutation)

        # Gathers a list of selected individuals to be replaced
        selected = g.tournament_selection(fitness, n_individuals)

        # For every selected individual
        for s in selected:
            # Gathers individual number of nodes
            n_nodes = space.trees[s].n_nodes

            # Checks if the tree has more than one node
            if n_nodes > 1:
                # Prunes the amount of maximum nodes
                max_nodes = self._prune_nodes(n_nodes)

                # Mutatets the individual
                space.trees[s] = self._mutate(space, space.trees[s], max_nodes)

            # If there is only one node
            else:
                # Re-create it with a random tree
                space.trees[s] = space.grow(space.min_depth, space.max_depth)

    def _crossover(self, space):
        """Crossover a number of individuals pre-selected through a tournament procedure (p. 101).

        Args:
            space (TreeSpace): A TreeSpace object.
            agents (list): Current iteration agents.
            trees (list): Current iteration trees.

        """

        # Calculates a list of current trees' fitness
        fitness = [fit for fit in space.fits]

        # Number of individuals to be crossovered
        n_individuals = int(space.n_trees * self.p_crossover)

        # Checks if `n_individuals` is an odd number
        if n_individuals % 2 != 0:
            # If it is, increase it by one
            n_individuals += 1

        # Gathers a list of selected individuals to be replaced
        selected = g.tournament_selection(fitness, n_individuals)

        # For every pair in selected individuals
        for s in g.n_wise(selected, 2):
            # Calculates the amount of father nodes
            father_nodes = space.trees[s[0]].n_nodes

            # Calculate the amount of mother nodes
            mother_nodes = space.trees[s[1]].n_nodes

            # Checks if both trees have more than one node
            if (father_nodes > 1) and (mother_nodes > 1):
                # Prunning father nodes
                max_f_nodes = self._prune_nodes(father_nodes)

                # Prunning mother nodes
                max_m_nodes = self._prune_nodes(mother_nodes)

                # Apply the crossover operation
                space.trees[s[0]], space.trees[s[1]] = self._cross(
                    space.trees[s[0]], space.trees[s[1]], max_f_nodes, max_m_nodes)

    @d.pre_evaluation
    def _evaluate(self, space, function):
        """Evaluates the search space according to the objective function.

        Args:
            space (TreeSpace): A TreeSpace object.
            function (Function): A Function object that will be used as the objective function.

        """

        # For every possible tree
        for i, tree in enumerate(space.trees):
            # Logs out tree for debugging purpose
            logger.debug(tree)

            # Evaluates the tree
            fit = function(tree)

            # If the fitness is a tensor
            if isinstance(fit, torch.Tensor):
                # Gathers as a float number
                fit = fit.item()

            # Replaces in the space's list of fitness
            space.fits[i] = fit

            # If current tree's fitness is smaller than best fitness
            if space.fits[i] < space.best_fit:
                # Makes a deep copy of current tree
                space.best_tree = copy.deepcopy(tree)

                # Makes a deep copy of the fitness
                space.best_fit = copy.deepcopy(space.fits[i])

    def run(self, space, function, store_best_only=False, pre_evaluation=None):
        """Runs the optimization pipeline.

        Args:
            space (TreeSpace): A TreeSpace object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluation (callable): This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Initial tree space evaluation
        self._evaluate(space, function, hook=pre_evaluation)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating trees with designed operators
                self._update(space)

                # After the update, we need to re-evaluate the tree space
                self._evaluate(space, function, hook=pre_evaluation)

                # Every iteration, we need the best tree and its fitness
                history.dump(best_tree=space.best_tree,
                             best_fit=space.best_fit)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_fit)
                b.update()

                logger.file(f'Fitness: {space.best_fit}')

        return history
