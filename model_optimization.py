import argparse

import torch
from torch.utils.data import DataLoader

import utils.loader as l
import utils.target as t
import utils.wrapper as w


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Optimizes a Machine Learning model with GP-based losses.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist', 'fmnist', 'kmnist', 'barrett-miccai', 'barrett-augsburg'])

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-n_input', help='Number of input units', type=int, default=784)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-n_classes', help='Number of classes', type=int, default=10)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=3)

    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=5)

    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=10)

    parser.add_argument('-min_depth', help='Minimum depth of trees', type=int, default=1)

    parser.add_argument('-max_depth', help='Maximum depth of trees', type=int, default=5)

    parser.add_argument('-shuffle', help='Whether data should be shuffled or not', type=bool, default=True)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'])

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    batch_size = args.batch_size
    n_input = args.n_input
    n_hidden = args.n_hidden
    n_classes = args.n_classes
    epochs = args.epochs
    lr = args.lr
    n_agents = args.n_agents
    n_iterations = args.n_iter
    min_depth = args.min_depth
    max_depth = args.max_depth
    shuffle = args.shuffle
    device = args.device
    seed = args.seed

    # Loads the data
    train, val, _ = l.load_tv_dataset(name=dataset, seed=seed)

    # Creates the iterators
    train_iterator = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    val_iterator = DataLoader(val, batch_size=batch_size, shuffle=shuffle)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Defining the optimization task
    opt_fn = t.validate_losses(train_iterator, val_iterator, n_input, n_hidden, n_classes, lr, epochs, device)

    # Running the optimization task
    history = w.run(opt_fn, n_trees=n_agents, n_terminals=2, n_iterations=n_iterations,
                    min_depth=min_depth, max_depth=max_depth, functions=['SUM', 'SUB', 'MUL', 'DIV'])

    # Saving optimization history
    history.save(f'outputs/{dataset}_{seed}.pkl')
