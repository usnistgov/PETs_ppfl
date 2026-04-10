from typing import List, Tuple
import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit

from torch.utils.data import DataLoader
from flwr_datasets.partitioner import (
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)
from utils import print_binned_counts


CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}


def load_pickle_data(data_path="../data/Oil_binned5"):
    cur_path = os.path.dirname(__file__)
    file_patterns = ["_ohe.dat", "_tt_vcf.dat", "_tt_pheno.dat", "_ho_vcf.dat", "_ho_pheno.dat"]

    """
    ORIGINAL CODE:

        dir_path = os.path.join(cur_path, "../../..", "genetic_plant_data")

    RECOMMENDED CODE:
    """

    dir_path = os.path.join(cur_path, data_path)

    """
    INTENDED ACTION: Modify

    JUSTIFICAITON: Fixing the reference
    """
    def load_by_pattern(pattern):
            matches = [f for f in os.listdir(dir_path) if f.endswith(pattern)]
            if not matches:
                raise FileNotFoundError(f"No file found in {dir_path} matching {pattern}")
            if len(matches) > 1:
                raise ValueError(f"Multiple files found in {dir_path} matching {pattern}: {matches}")

            file_path = os.path.relpath(os.path.join(dir_path, matches[0]))
            with open(file_path, "rb") as f:
                return pickle.load(f)

    ohe, tt_vcf, tt_pheno, ho_vcf, ho_pheno = [
        load_by_pattern(pattern) for pattern in file_patterns
    ]

    vcf = np.concatenate((tt_vcf, ho_vcf), axis=0)
    pheno = np.concatenate((tt_pheno, ho_pheno), axis=0)

    return ohe, vcf, pheno


def instantiate_partitioner(partitioner_type: str, num_partitions: int):
    """Initialise partitioner based on selected partitioner type
    and number of partitions"""
    _, vcf, pheno = load_pickle_data()

    concat_dataset = np.concatenate((vcf, pheno), axis=1)
    indices = np.arange(len(concat_dataset))

    # Dataset class works with Pandas dataframes, but not Numpy arrays
    partitioner = CORRELATION_TO_PARTITIONER[partitioner_type](
        num_partitions=num_partitions
    )
    partitioner.dataset = Dataset.from_pandas(pd.DataFrame(indices))
    return partitioner


def train_test_partition_split(
    partition: Dataset, test_fraction: float, seed: int
):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def train_test_indices_split(
    dataset: np.ndarray, indices: np.ndarray, test_frac: float, seed: int
) -> Tuple[List[int], List[int]]:
    """
    Split dataset indices into train and test sets for regression tasks.
    Continuous labels are split into train and test sets using stratified
    sampling based on quantile bins. If any bin has less than 2 samples,
    they are added to the training set, and the sufficient indices are
    stratified split.
    """
    # Get labels from the dataset
    labels = dataset[indices, -1]

    # Create quantile bins for stratification
    num_bins = min(10, len(np.unique(labels)))  # Adjust number of bins
    bins = np.linspace(np.min(labels), np.max(labels), num_bins + 1)
    binned_labels = np.digitize(labels, bins) - 1

    # Count occurrences of each bin
    bin_counts = Counter(binned_labels)

    # Find bins with less than 2 samples
    insufficient_bins = [b for b, count in bin_counts.items() if count < 2]

    # Split indices into insufficient and sufficient bins groups
    insufficient_indices = [
        i for i, b in zip(indices, binned_labels) if b in insufficient_bins
    ]
    sufficient_indices = [
        i for i, b in zip(indices, binned_labels) if b not in insufficient_bins
    ]
    sufficient_binned_labels = [
        b for i, b in zip(indices, binned_labels) if b not in insufficient_bins
    ]

    # Perform stratified splitting on sufficient indices
    print(
        'Stratified split: ',
        f'insufficient indices: {len(insufficient_indices)}, '
        f'sufficient indices: {len(sufficient_indices)}, {bin_counts}',
    )
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed
    )
    train_sufficient, test_sufficient = next(
        sss.split(sufficient_indices, sufficient_binned_labels)
    )

    # Map back to original indices
    train_indices = np.array(sufficient_indices)[train_sufficient].tolist()
    test_indices = np.array(sufficient_indices)[test_sufficient].tolist()

    # Add insufficient indices to the training set
    train_indices += insufficient_indices

    return train_indices, test_indices


def load_random_partitions(
    data_partition_id: int,
    combined_dataset: np.ndarray,
    batch_size: int,
    test_fraction: float,
    seed: int,
    num_partitions: int,
    partitioner_type: str,
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
    train_indices = train_indices.to_pandas().to_numpy().flatten().tolist()
    test_indices = test_indices.to_pandas().to_numpy().flatten().tolist()

    # Split into train and test based on the indices
    train_data = combined_dataset[train_indices]
    test_data = combined_dataset[test_indices]

    # count labels in train and test set
    print('Train dataset binned label counts')
    print_binned_counts(combined_dataset, train_indices)
    print('Test dataset binned label counts')
    print_binned_counts(combined_dataset, test_indices)

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
    combined_dataset: np.ndarray,
    data_partitions: dict,
    batch_size: int,
    test_fraction: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    # Check if data partition id is available in the data partitions
    data_partition_ids = sorted(
        [k for k in data_partitions.keys() if re.match(r'client_\d+', k)]
    )
    numeric_id = [int(ci.split('_')[-1]) for ci in data_partition_ids]
    if data_partition_id not in numeric_id:
        print(
            f"Cannot use client {data_partition_id} for training because "
            f"it was not found in the data partitions."
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
    print('Train dataset binned label counts')
    print_binned_counts(combined_dataset, train_indices)
    print('Test dataset binned label counts')
    print_binned_counts(combined_dataset, test_indices)

    # create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, train_indices, test_indices
