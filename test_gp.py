import torch
from opytimizer import Opytimizer
from opytimizer.core.function import Function

from assembler.gp import LossGP
from assembler.space import LossTreeSpace

torch.manual_seed(0)

x = torch.randn(3, 5)
y = torch.empty(3, dtype=torch.long).random_(5)

def func(t):
    fitness = t.evaluate(x, y)
    
    return fitness

# Creating a loss-based TreeSpace
space = LossTreeSpace(n_trees=5, n_terminals=2, min_depth=1, max_depth=5, functions=['SUM', 'SUB', 'MUL', 'DIV'])

# Creating a loss-based GP optimizer
optimizer = LossGP()

# Creating the Function
function = Function(pointer=func)

# Creating the optimization task
task = Opytimizer(space=space, optimizer=optimizer, function=function)

# Initializing task
history = task.start(store_best_only=True)

print(space.best_tree)
print(space.best_fit)