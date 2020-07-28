import torch
import numpy as np
from evolver.node import LossNode

torch.manual_seed(0)

# Creating two new Nodes
n1 = LossNode(name='0', node_type='TERMINAL', value=torch.nn.CrossEntropyLoss())
n2 = LossNode(name='1', node_type='TERMINAL', value=torch.nn.CrossEntropyLoss())

# #
x = torch.randn(3, 5)
y = torch.empty(3, dtype=torch.long).random_(5)

print(n1.position(x, y), n2.position(x, y))

# # Additionally, one can stack nodes to create a tree
t = LossNode(name='SUM', node_type='FUNCTION', left=n1, right=n2)

# # Defining `n1` and `n2` parent as `t`
n1.parent = t
n2.parent = t

# # Outputting information about the tree
print(t.position(x, y))
# print(f'Post Order: {t.post_order} | Size: {t.n_nodes} | Minimum Depth: {t.min_depth} | Maximum Depth: {t.max_depth}.')