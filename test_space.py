import torch
from evolver.space import LossTreeSpace
from opytimizer.spaces.tree import TreeSpace

torch.manual_seed(0)

space = LossTreeSpace(n_trees=5, n_terminals=2, min_depth=1, max_depth=5, functions=['SUM', 'SUB', 'MUL', 'DIV'])

x = torch.randn(3, 5)
y = torch.empty(3, dtype=torch.long).random_(5)

print(space.trees[0])
print(space.trees[0].evaluate(x, y))
