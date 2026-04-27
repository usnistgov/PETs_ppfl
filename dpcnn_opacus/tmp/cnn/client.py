import sys
from typing import List, Tuple
from collections import OrderedDict
from pathlib import Path
import re
import json
from datetime import datetime
from collections import Counter

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import (
    instantiate_partitioner,
    train_test_partition_split,
    load_pickle_data,
    train_test_indices_split,
)
from model import Net, eval_cnn, save_cnn, train_cnn
from utils import get_device


# warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = get_device()
print('DEVICE: ', DEVICE)
# Parse arguments for client parameters
args = client_args_parser()
print(f"client args: {args}")
partitioner_type = args.partitioner_type
client_id = args.client_id
partition_id = args.partition_id
num_partitions = args.num_partitions
batch_divisor = args.batch_divisor
learning_rate = args.learning_rate
weight_decay = args.weight_decay
seed = args.seed
test_fraction = args.test_fraction
epochs = args.epochs
data_partitions_file = args.data_partitions_file
n_models = args.n_models

ohe, tt_vcf, tt_pheno = load_pickle_data()
combined_dataset = np.concatenate((tt_vcf, tt_pheno), axis=1)

# Check if data partitions file is provided and exists
data_partitions = None
if data_partitions_file and Path(data_partitions_file).exists():
    data_partitions = np.load(data_partitions_file)
    print(f"Loaded data partitions from {data_partitions_file}")
else:
    print(
        f"Data partitions file not found at {data_partitions_file}. "
        f"Using a random partition for client {client_id}"
    )


def load_data(
    data_partition_id: int,
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    """
    Load data using flower dataset partitioner
    """
    # initialize and get data partition
    partitioner = instantiate_partitioner(
        partitioner_type=partitioner_type, num_partitions=num_partitions
    )
    partition = partitioner.load_partition(data_partition_id)
    train_indices, test_indices, num_train, num_test = (
        train_test_partition_split(
            partition, test_fraction=test_fraction, seed=seed
        )
    )

    # Instantiating the dataset partition requires Pandas,
    # but Torch loaders expect Numpy
    train_indices = train_indices.to_pandas().to_numpy().flatten()
    test_indices = test_indices.to_pandas().to_numpy().flatten()

    # Split into train and test based on the indices
    train_data = combined_dataset[train_indices]
    test_data = combined_dataset[test_indices]

    # count labels in train and test set
    train_class_counts = Counter(train_data[:, -1])
    test_class_counts = Counter(test_data[:, -1])
    print(
        f"Train class counts: {train_class_counts}, "
        f"Test class counts: {test_class_counts}"
    )

    batch_size = max(1, tt_vcf.shape[0] // batch_divisor)

    # create data loaders
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_data_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    return train_data_loader, test_data_loader, train_indices, test_indices


def load_custom_partitions(
    data_partition_id: int,
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    # Check if data partition id is available in the data partitions
    data_partition_ids = sorted(
        [k for k in data_partitions.keys() if re.match(r'client_\d+', k)]
    )
    numeric_id = [int(ci.split('_')[-1]) for ci in data_partition_ids]
    if data_partition_id not in numeric_id:
        print(
            f"Cannot use client {client_id} for training because it was "
            f"not found in the data partitions."
        )
        sys.exit(1)

    # get data partition indices and create train test split
    data_partition_str_id = data_partition_ids[data_partition_id]
    partition_indices = data_partitions[data_partition_str_id]

    train_indices, test_indices = train_test_indices_split(
        combined_dataset, partition_indices, test_fraction, seed
    )
    # Split into train and test based on the indices
    train_dataset = combined_dataset[train_indices]
    test_dataset = combined_dataset[test_indices]

    # count labels in train and test set
    train_class_counts = Counter(train_dataset[:, -1])
    test_class_counts = Counter(test_dataset[:, -1])
    print(
        f"Train class counts: {train_class_counts}, "
        f"Test class counts: {test_class_counts}"
    )

    # create dataloaders
    batch_size = max(1, tt_vcf.shape[0] // batch_divisor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, train_indices, test_indices


if data_partitions is not None:
    # if data partitions available, train a model for each data partition
    train_loader, test_loader, train_indices, test_indices = (
        load_custom_partitions(client_id)
    )
else:
    # Partition data into n_models randomly to train N client models
    train_loader, test_loader, train_indices, test_indices = load_data(
        partition_id
    )


# Load model and data
model = Net(tt_vcf.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)


def save_client(
    federated_round,
    train_acc,
    test_acc,
    train_loss,
    test_loss,
    train_acc_per_epoch,
    test_acc_per_epoch,
    losses_per_epoch,
    predicted_labels,
):
    partitions_path = (
        Path(data_partitions_file).name if data_partitions else 'none'
    )
    metadata = {
        'created on': str(datetime.now()),
        'model id': int(client_id),
        'partitions file': partitions_path,
        'train accuracy': float(train_acc),
        'test accuracy': float(test_acc),
        'train loss': float(train_loss),
        'test loss': float(test_loss),
        'train indices': train_indices,
        'test indices': test_indices,
        "train accuracy per epoch": np.array(train_acc_per_epoch),
        "test accuracy per epoch": np.array(test_acc_per_epoch),
        "losses per epoch": np.array(losses_per_epoch),
        "predicted labels": json.dumps(predicted_labels),
        'hyperparameters': {
            'learning rate': float(learning_rate),
            'weight decay': float(weight_decay),
            'batch divisor': int(batch_divisor),
            'epochs': int(epochs),
            'seed': int(seed),
            'test fraction': float(test_fraction),
        },
    }

    metadata['hyperparameters'] = json.dumps(metadata['hyperparameters'])
    save_cnn(model, metadata, f'flcnn_{client_id}', federated_round)


def train_partition():
    # train model
    model.to(DEVICE)
    train_acc, test_acc, losses = train_cnn(
        train_loader, test_loader, epochs, model, optimizer, criterion, DEVICE
    )
    return train_acc, test_acc, losses


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        model.eval()
        return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).float().to(DEVICE)
                                  for k, v in params_dict})
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.current_round = config.get('server_round', 1) - 1
        self.train_acc_per_epoch, self.test_acc_per_epoch, self.losses = (
            train_partition()
        )
        return (
            self.get_parameters(config=config),
            len(train_loader.dataset),
            {},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        train_acc, test_acc, train_loss, test_loss, predicted_labels = (
            eval_cnn(train_loader, test_loader, model, criterion)
        )
        save_client(
            self.current_round,
            train_acc,
            test_acc,
            train_loss,
            test_loss,
            self.train_acc_per_epoch,
            self.test_acc_per_epoch,
            self.losses,
            predicted_labels,
        )
        # convert loss to float to avoid Flower-Numpy type error
        return (
            float(test_loss),
            len(test_loader.dataset),
            {"accuracy": test_acc},
        )


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
)
