import re
import json
from datetime import datetime
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from dataset import load_pickle_data, train_test_indices_split
from utils import centralized_args_parser, print_binned_counts
from model import Net, train_cnn, eval_cnn, save_cnn


# Parse arguments for hyperparameters
args = centralized_args_parser()
print(f"hyperparam args: {args}")
learning_rate = args.learning_rate
weight_decay = args.weight_decay
batch_divisor = args.batch_divisor
epochs = args.epochs
seed = args.seed
test_frac = args.test_fraction
accuracy_tolerance = args.accuracy_tolerance
data_partitions_file = args.data_partitions_file
n_models = args.n_models

torch.manual_seed(seed)

# Importing the model building data, one hot encoding and creating a label set
ohe, vcf, pheno = load_pickle_data()
print(f"data (tt + ho) shape: {vcf.shape}")
print(f"labels (tt + ho) shape: {pheno.shape}")
combined_dataset = np.concatenate((vcf, pheno), axis=1)

# Check if data partitions file is provided and exists
data_partitions = None
if data_partitions_file and Path(data_partitions_file).exists():
    data_partitions = np.load(data_partitions_file)
    print(f"Loaded data partitions from {data_partitions_file}")
else:
    print(
        f"Data partitions file not found at {data_partitions_file}. "
        f"Using n_models = {n_models}"
    )


def train_partition(
    model_id: int, combined_dataset: np.ndarray, indices: np.ndarray
):
    train_indices, test_indices = train_test_indices_split(
        combined_dataset, indices, test_frac, seed
    )

    # Split into train and test based on the indices
    train_dataset = combined_dataset[train_indices]
    test_dataset = combined_dataset[test_indices]

    print('Train dataset binned label counts')
    print_binned_counts(combined_dataset, train_indices)
    print('Test dataset binned label counts')
    print_binned_counts(combined_dataset, test_indices)

    # create dataloaders
    batch_size = max(1, vcf.shape[0] // batch_divisor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # create model
    model = Net(vcf.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adamax(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # train, evaluate and save model
    train_cnn(
        model_id,
        train_loader,
        epochs,
        model,
        optimizer,
        criterion,
        accuracy_tolerance,
    )
    train_acc, test_acc, train_loss, test_loss, train_mse, test_mse = eval_cnn(
        model_id,
        train_loader,
        test_loader,
        model,
        criterion,
        accuracy_tolerance,
    )
    partitions_path = (
        Path(data_partitions_file).name if data_partitions else 'none'
    )
    metadata = {
        'created on': str(datetime.now()),
        'model id': int(model_id),
        'partitions file': partitions_path,
        'train accuracy': float(train_acc),
        'test accuracy': float(test_acc),
        'train mean squared error': float(train_mse),
        'test mean squared error': float(test_mse),
        'train loss': float(train_loss),
        'test loss': float(test_loss),
        'train indices': train_indices,
        'test indices': test_indices,
        'hyperparameters': {
            'learning rate': float(learning_rate),
            'weight decay': float(weight_decay),
            'batch divisor': int(batch_divisor),
            'epochs': int(epochs),
            'seed': int(seed),
            'test fraction': float(test_frac),
            'accuracy tolerance': float(accuracy_tolerance),
        },
    }

    metadata['hyperparameters'] = json.dumps(metadata['hyperparameters'])
    save_cnn(model, metadata, f'cnn_oil_{model_id}')


def train_custom_partitions():
    """Create models for each given data partitions"""
    data_partition_ids = sorted(
        [k for k in data_partitions.keys() if re.match(r'client_\d+', k)]
    )
    # Train a model for each given data partitions
    for partition_id in data_partition_ids:
        model_id = partition_id.split('_')[-1]
        print('Model id:', model_id)
        partition_indices = data_partitions[partition_id]
        print(
            f"Training model {model_id} "
            f"with {len(partition_indices)} samples"
        )
        train_partition(model_id, combined_dataset, partition_indices)


def train_random_partitions():
    """Create models for randomly partitioned data."""
    X, y = combined_dataset[:, :-1], combined_dataset[:, -1]
    dataset_indices = np.arange(len(X))
    print()
    if n_models == 1:
        # Training only single model, so partition is required
        indices = dataset_indices  # Use all indices
        print(f"Training model 1/1 with {len(indices)} samples")
        train_partition(1, combined_dataset, indices)
    else:
        # Partition data using StratifiedKFold where n_splits = n_models
        skf = StratifiedKFold(
            n_splits=n_models, shuffle=True, random_state=seed
        )

        # Create stratified dataset splits
        splits = list(skf.split(dataset_indices, y))

        # Create a model for each dataset split
        for i, (_, indices) in enumerate(splits):
            print(
                f"Training model {i + 1}/{n_models} "
                f"with {len(indices)} samples"
            )
            train_partition(i, combined_dataset, indices)
            print()


# Check if data partitions are provided
if data_partitions is not None:
    # if data partitions available, train a model for each data partition
    train_custom_partitions()
else:
    # Partition data into n_models randomly to train N client models
    train_random_partitions()
