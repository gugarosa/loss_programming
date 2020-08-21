# Defining the dataset to be used
DATA="mnist"

# Batch size
BATCH_SIZE=128

# Number of input units
N_INPUT=784

# Number of hidden units
N_HIDDEN=128

# Number of classes
N_CLASSES=10

# Learning rate
LR=0.001

# Epochs
EPOCHS=3

# Number of agents
N_AGENTS=5

# Number of iterations
N_ITER=10

# Minimum depth of trees
MIN_DEPTH=1

# Maximum depth of trees
MAX_DEPTH=5

# Number of runnings
N_RUNS=25

# Creating a loop
for i in $(seq 1 $N_RUNS); do
    # Running the optimization
    python model_optimization.py $DATA -batch_size $BATCH_SIZE -n_input $N_INPUT -n_hidden $N_HIDDEN -n_classes $N_CLASSES -lr $LR -epochs $EPOCHS -n_agents $N_AGENTS -n_iter $N_ITER -min_depth $MIN_DEPTH -max_depth $MAX_DEPTH -seed $i

    # Running the final evaluation
    python model_evaluation.py $DATA -batch_size $BATCH_SIZE -n_input $N_INPUT -n_hidden $N_HIDDEN -n_classes $N_CLASSES -lr $LR -epochs $EPOCHS -seed $i
done