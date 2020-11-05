import argparse

import numpy as np
import torch
from opytimizer.utils.history import History
from torch.utils.data import DataLoader

import utils.loader as l
import utils.object as o
import utils.target as t
import utils.wrapper as w
from core.losses import CrossEntropyLoss
from models.mlp import MLP


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Evaluates a Machine Learning model with GP-based best loss or traditional loss.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist', 'fmnist', 'kmnist',
                                                                       'barrett-miccai', 'barrett-augsburg',
                                                                       'exudate'])

    parser.add_argument('model', help='Model identifier', choices=['mlp', 'resnet'])

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-n_input', help='Number of input units', type=int, default=784)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-n_classes', help='Number of classes', type=int, default=10)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=3)

    parser.add_argument('-shuffle', help='Whether data should be shuffled or not', type=bool, default=True)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'], default='cpu')

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Transforms arguments into variables
    dataset = args.dataset
    name = args.model
    batch_size = args.batch_size
    n_input = args.n_input
    n_hidden = args.n_hidden
    n_classes = args.n_classes
    lr = args.lr
    epochs = args.epochs
    shuffle = args.shuffle
    device = args.device
    seed = args.seed

    # Loads the optimization history
    h = History()
    #h.load(f'outputs/{dataset}_{seed}.pkl')

    # Loads the data
    train, _, test = l.load_dataset(name=dataset, seed=seed)

    # Creates the iterators
    train_iterator = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_iterator = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    # Defines the torch seed
    torch.manual_seed(seed)

    # Initializing the model
    model_obj = o.get_model(name).obj
    model = model_obj(n_input=n_input, n_hidden=n_hidden, n_classes=n_classes, lr=lr, init_weights=None, device=device)

    # Gathers the loss function (replace line below for changing to a standard loss)
    #model.loss = h.best_tree[-1]
    model.loss = CrossEntropyLoss()

    # Fits the model
    model.fit(train_iterator, epochs)

    # Evaluates the model
    _, acc = model.evaluate(test_iterator)

    # Saving accuracy to output
    with open(f'outputs/{dataset}_{seed}.txt', 'w') as f:
        f.write(f'{acc}')
