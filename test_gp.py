import numpy as np
import torch
from opytimizer import Opytimizer
from opytimizer.core.function import Function
from torch.utils.data import DataLoader

import utils.loader as l
import utils.target as t
import utils.wrapper as w
from assembler.gp import LossGP
from assembler.space import LossTreeSpace

# Loads the data
train, val, _ = l.load_dataset(name='mnist')

# Creates the iterators
train_iterator = DataLoader(train, batch_size=128, shuffle=False)
val_iterator = DataLoader(val, batch_size=128, shuffle=False)

# Defining the torch seed
torch.manual_seed(0)

# Defining the optimization function
opt_fn = t.validate_losses(train_iterator, val_iterator, epochs=1)

# Running the optimization task
history = w.run(opt_fn, n_trees=5, n_terminals=2, n_iterations=5,
                min_depth=1, max_depth=5, functions=['SUM', 'SUB', 'MUL', 'DIV'])
