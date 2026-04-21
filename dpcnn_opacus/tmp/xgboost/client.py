import sys
import re
from typing import Tuple, List
import warnings
import numpy as np
from pathlib import Path
from collections import Counter
import flwr as fl
from xgboost.core import DMatrix

from client_utils import XgbClient
from dataset import (
    instantiate_partitioner,
    train_test_partition_split,
    train_test_indices_split,
    load_pickle_data,
)
from utils import (
    client_args_parser,
    get_best_params,
    NUM_LOCAL_ROUND,
)


warnings.filterwarnings("ignore", category=UserWarning)


# Parse arguments for experimental settings
args = client_args_parser()

# Train method (bagging or cyclic)
train_method = args.train_method
partitioner_type = args.partitioner_type
client_id = args.client_id
partition_id = args.partition_id
num_partitions = args.num_partitions
test_frac = args.test_fraction
seed = args.seed
data_partitions_file = args.data_partitions_file

ohe, tt_vcf, tt_pheno = load_pickle_data()
combined_dataset = np.concatenate((tt_vcf, tt_pheno), axis=1)

# Check if data partitions file is provided and exists
data_partitions = None
if data_partitions_file and Path(data_partitions_file).exists():
    data_partitions = np.load(data_partitions_file)
    print(f"Loaded data partitions from {data_partitions_file}")
else:
    print(
        f"Data partitions file not found at {data_partitions_file}."
        f"Data will be partitioned using the partitioner type "
        f"{partitioner_type}"
    )


def load_data(
    data_partition_id: int,
) -> Tuple[DMatrix, DMatrix, List[int], List[int]]:
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
            partition, test_fraction=test_frac, seed=seed
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

    train_dmatrix = DMatrix(data=train_data[:, :-1], label=train_data[:, -1])
    test_dmatrix = DMatrix(data=test_data[:, :-1], label=test_data[:, -1])

    return train_dmatrix, test_dmatrix, train_indices, test_indices


def load_custom_partitions(
    data_partition_id: int,
) -> Tuple[DMatrix, DMatrix, List[int], List[int]]:
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
        combined_dataset, partition_indices, test_frac, seed
    )
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
    train_dmatrix = DMatrix(data=train_data[:, :-1], label=train_data[:, -1])
    test_dmatrix = DMatrix(data=test_data[:, :-1], label=test_data[:, -1])

    return train_dmatrix, test_dmatrix, train_indices, test_indices


if data_partitions is not None:
    # if data partitions available, train a model for each data partition
    train_dmatrix, test_dmatrix, train_indices, test_indices = (
        load_custom_partitions(client_id)
    )
else:
    # Partition data into n_models randomly to train N client models
    train_dmatrix, test_dmatrix, train_indices, test_indices = load_data(
        partition_id
    )


# Hyper-parameters for xgboost training
num_local_round = NUM_LOCAL_ROUND
params = get_best_params()
params = {**params, 'random_state': seed}
# Setup learning rate
if args.train_method == "bagging" and args.scaled_lr:
    new_lr = params["eta"] / args.num_partitions
    params.update({"eta": new_lr})

# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=XgbClient(
        client_id,
        train_dmatrix,
        test_dmatrix,
        train_indices,
        test_indices,
        num_local_round,
        params,
        train_method,
        data_partitions_file,
        test_frac,
    ),
)
