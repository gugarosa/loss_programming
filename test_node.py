import numpy as np
import torch

from assembler.node import LossNode

torch.manual_seed(0)

# Creating two new Nodes
n1 = LossNode(name='0', node_type='TERMINAL', value=torch.nn.CrossEntropyLoss())
n2 = LossNode(name='1', node_type='TERMINAL', value=torch.nn.CrossEntropyLoss())

# Creating two new Nodes
n3 = LossNode(name='2', node_type='TERMINAL', value=torch.nn.CrossEntropyLoss())
n4 = LossNode(name='3', node_type='TERMINAL', value=torch.nn.CrossEntropyLoss())

# #
x = torch.randn(3, 5)
y = torch.empty(3, dtype=torch.long).random_(5)

print(n1.evaluate(x, y), n2.evaluate(x, y))

# # Additionally, one can stack nodes to create a tree
t1 = LossNode(name='MUL', node_type='FUNCTION', left=n1, right=n2)
t2 = LossNode(name='MUL', node_type='FUNCTION', left=n3, right=n4)
t3 = LossNode(name='DIV', node_type='FUNCTION', left=t1, right=t2)

# # Defining `n1` and `n2` parent as `t`
n1.parent = t1
n2.parent = t1

n3.parent = t2
n4.parent = t2

t1.parent = t3
t2.parent = t3

# # Outputting information about the tree
print(t3.evaluate(x, y))
print(t3)
# print(f'Post Order: {t.post_order} | Size: {t.n_nodes} | Minimum Depth: {t.min_depth} | Maximum Depth: {t.max_depth}.')
