import torch
import torchvision as tv

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'mnist': tv.datasets.MNIST,
    'fmnist': tv.datasets.FashionMNIST,
    'kmnist': tv.datasets.KMNIST
}


def load_dataset_from_folder(folder, val_split):
    """Loads a dataset from a folder of images.

    Args:
        folder (str): Dataset's folder path.
        val_split (float): Percentage of split for the validation set.

    Returns:
        Training, validation and testing sets of loaded dataset.

    """

    # Creating the dataset
    dataset = tv.datasets.ImageFolder(root=folder,
                                      transform=tv.transforms.Compose([
                                          tv.transforms.Resize((128, 128)),
                                          tv.transforms.ToTensor(),
                                          tv.transforms.Normalize(
                                              (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ]))

    # Splitting the training data into training/validation
    train, val, test = torch.utils.data.random_split(
        dataset, [int(len(dataset) * (1 - (2 * val_split))), int(len(dataset) * val_split), int(len(dataset) * val_split)])

    return train, val, test


def load_dataset(name='mnist', val_split=0.25, seed=0):
    """Loads a dataset.

    Args:
        name (str): Name of dataset to be loaded.
        val_split (float): Percentage of split for the validation set.
        seed (int): Randomness seed.

    Returns:
        Training, validation and testing sets of loaded dataset.

    """

    # Defining the torch seed
    torch.manual_seed(seed)

    # Checks if it is supposed to load custom datasets
    if name == 'barrett-miccai':
        return load_dataset_from_folder('data/MICCAI', val_split)
    elif name == 'barrett-augsburg':
        return load_dataset_from_folder('data/AUGSBURG', val_split)
    elif name =='exudate':
        return load_dataset_from_folder('data/EXUDATE', val_split)

    # Loads the training data
    train = DATASETS[name](root='./data', train=True, download=True,
                           transform=tv.transforms.ToTensor())

    # Splitting the training data into training/validation
    train, val = torch.utils.data.random_split(
        train, [int(len(train) * (1 - val_split)), int(len(train) * val_split)])

    # Loads the testing data
    test = DATASETS[name](root='./data', train=False, download=True,
                          transform=tv.transforms.ToTensor())

    return train, val, test
