import torch
import copy
from opytimizer.optimizers.evolutionary.gp import GP
import opytimizer.utils.decorator as d
import opytimizer.utils.history as h


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

    @d.pre_evaluation
    def _evaluate(self, space, function):
        """Evaluates the search space according to the objective function.

        Args:
            space (TreeSpace): A TreeSpace object.
            function (Function): A Function object that will be used as the objective function.

        """

        #
        for i, tree in enumerate(space.trees):
            #
            fit = function(tree)

            #
            if isinstance(fit, torch.Tensor):
                #
                fit = fit.item()

            #
            space.fits[i] = fit

            #
            if space.fits[i] < space.best_fit:
                # Makes a deep copy of current tree
                space.best_tree = copy.deepcopy(tree)

                # Makes a deep copy of the fitness
                space.best_fit = copy.deepcopy(space.fits[i])
        

        # # Iterates through all (trees, agents)
        # for (tree, agent) in zip(space.trees, space.agents):
        #     # Runs through the tree and returns a position array
        #     agent.position = copy.deepcopy(tree.position)

        #     # Checks the agent limits
        #     agent.clip_limits()

        #     # Calculates the fitness value of the agent
        #     agent.fit = function(agent.position)

        #     # If tree's fitness is better than global fitness
        #     if agent.fit < space.best_agent.fit:
        #         # Makes a deep copy of current tree
        #         space.best_tree = copy.deepcopy(tree)

        #         # Makes a deep copy of agent's position to the best agent
        #         space.best_agent.position = copy.deepcopy(agent.position)

        #         # Also, copies its fitness from agent's fitness
        #         space.best_agent.fit = copy.deepcopy(agent.fit)

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

        # # Initializing a progress bar
        # with tqdm(total=space.n_iterations) as b:
        #     # These are the number of iterations to converge
        #     for t in range(space.n_iterations):
        #         logger.file(f'Iteration {t+1}/{space.n_iterations}')

        #         # Updating trees with designed operators
        #         self._update(space)

        #         # After the update, we need to re-evaluate the tree space
        #         self._evaluate(space, function, hook=pre_evaluation)

        #         # Every iteration, we need to dump agents and best agent
        #         history.dump(agents=space.agents,
        #                      best_agent=space.best_agent,
        #                      best_tree=space.best_tree)

        #         # Updates the `tqdm` status
        #         b.set_postfix(fitness=space.best_agent.fit)
        #         b.update()

        #         logger.file(f'Fitness: {space.best_agent.fit}')
        #         logger.file(f'Position: {space.best_agent.position}')

        return history