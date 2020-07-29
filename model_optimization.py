import argparse

import torch
from torch.utils.data import DataLoader

import optimization.wrapper as w
import utils.constraints as c
import utils.loader as l
import utils.target as t


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Trains and evaluates a machine learning model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist', 'fmnist', 'kmnist'])

    parser.add_argument('-shuffle', help='Whether data should be shuffled or not', type=bool, default=True)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'])

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    shuffle = args.shuffle
    seed = args.seed

    # Loads the data
    train, val, _ = l.load_dataset(name=dataset)

    # Creates the iterators
    train_iterator = DataLoader(train, batch_size=128, shuffle=shuffle)
    val_iterator = DataLoader(val, batch_size=128, shuffle=shuffle)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Defining the optimization task
    opt_fn = t.loss_function(train_iterator, val_iterator, epochs=1)

    # Running the optimization task
    history = w.run(opt_fn, [c.sum_equals_one], 25, 2, 2, 3, 2, 5, ['SUM', 'SUB', 'MUL', 'DIV'], [0, 0], [1, 1], dict())

    print(history.best_agent[-1])
