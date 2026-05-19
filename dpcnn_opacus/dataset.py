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
from pathlib import Path

from torch.utils.data import DataLoader
from flwr_datasets.partitioner import (
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)
from utils import print_binned_counts
from convert_dat_to_npy import convert_dat_to_npy
import torch
from torch.utils.data import Dataset as TorchDataset

NPY_SUFFIXES = ("_tt_vcf", "_tt_pheno", "_ho_vcf", "_ho_pheno")

class IndexedArrayDataset(TorchDataset):
    """Lazy row lookup over the old logical [tt; ho] dataset.

    The backing arrays stay separate and mmap-backed. Global row indices keep
    their historical meaning: first tt rows, then ho rows.
    """

    def __init__(self, tt_features, tt_labels, ho_features, ho_labels, indices):
        self.tt_features = tt_features
        self.tt_labels = tt_labels.reshape(-1)
        self.ho_features = ho_features
        self.ho_labels = ho_labels.reshape(-1)
        self.tt_len = len(tt_features)
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row_idx = int(self.indices[idx])

        if row_idx < self.tt_len:
            return self.tt_features[row_idx], self.tt_labels[row_idx]

        ho_idx = row_idx - self.tt_len
        return self.ho_features[ho_idx], self.ho_labels[ho_idx]

CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}


def load_pickle_data(data_path):
    """Legacy pickle loader.

    Kept for older scripts such as centralized_train.py. The Flower simulation
    path should use load_npy_feature_label_data to avoid full in-memory copies.
    """
    cur_path = os.path.dirname(__file__)
    file_patterns = ["_ohe.dat", "_tt_vcf.dat", "_tt_pheno.dat", "_ho_vcf.dat", "_ho_pheno.dat"]
    dir_path = os.path.join(cur_path, data_path)
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


def resolve_data_dir(data_path) -> Path:
    data_dir = Path(data_path)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent / data_dir
    return data_dir.resolve()

def find_single_npy(data_dir: Path, suffix: str) -> Path | None:
    matches = list(data_dir.glob(f"*{suffix}.npy"))
    if len(matches) > 1:
        raise ValueError(f"Multiple .npy files found for {suffix}: {matches}")
    return matches[0] if matches else None

def ensure_npy_feature_label_files(data_path) -> Path:
    data_dir = resolve_data_dir(data_path)
    missing = [suffix for suffix in NPY_SUFFIXES if find_single_npy(data_dir, suffix) is None]

    if missing:
        print(
            "Missing required .npy data files "
            f"({', '.join(missing)}). Converting .dat files in {data_dir}."
        )
        convert_dat_to_npy(data_dir)

    still_missing = [
        suffix for suffix in NPY_SUFFIXES if find_single_npy(data_dir, suffix) is None
    ]
    if still_missing:
        raise FileNotFoundError(
            "Missing required .npy files after conversion: "
            f"{', '.join(still_missing)} in {data_dir}"
        )

    return data_dir

def get_split_labels(tt_labels, ho_labels, indices):
    indices = np.asarray(indices)
    tt_len = len(tt_labels)
    total_len = tt_len + len(ho_labels)
    if len(indices) and (indices.min() < 0 or indices.max() >= total_len):
        raise IndexError(
            f"Partition index range [{indices.min()}, {indices.max()}] "
            f"is outside logical dataset size {total_len}."
        )

    out = np.empty(len(indices), dtype=np.asarray(tt_labels).dtype)
    tt_mask = indices < tt_len

    out[tt_mask] = tt_labels[indices[tt_mask]]
    out[~tt_mask] = ho_labels[indices[~tt_mask] - tt_len]

    return out

def train_test_split_backed_indices(tt_labels, ho_labels, indices, test_frac, seed):
    selected_labels = get_split_labels(tt_labels, ho_labels, indices)
    local_indices = np.arange(len(selected_labels))

    local_train, local_test = train_test_indices_split(
        selected_labels.reshape(-1, 1),
        local_indices,
        test_frac,
        seed,
    )

    indices = np.asarray(indices)
    return indices[local_train].tolist(), indices[local_test].tolist()

def load_npy_feature_label_data(data_path):
    """Load tt/ho feature and label arrays as read-only mmap-backed arrays."""
    data_dir = resolve_data_dir(data_path)

    def load_one(suffix):
        path = find_single_npy(data_dir, suffix)
        if path is None:
            raise FileNotFoundError(f"No .npy file found in {data_dir} matching *{suffix}.npy")
        return np.load(path, mmap_mode="r")

    tt_vcf = load_one(NPY_SUFFIXES[0])
    tt_pheno = load_one(NPY_SUFFIXES[1])
    ho_vcf = load_one(NPY_SUFFIXES[2])
    ho_pheno = load_one(NPY_SUFFIXES[3])

    return tt_vcf, tt_pheno.reshape(-1), ho_vcf, ho_pheno.reshape(-1)

def instantiate_partitioner(partitioner_type, num_partitions, data_dir, num_rows=None):
    if num_rows is None:
        tt_features, _, ho_features, _ = load_npy_feature_label_data(data_dir)
        num_rows = len(tt_features) + len(ho_features)

    indices = np.arange(num_rows)
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
    data_partition_id,
    tt_features,
    tt_labels,
    ho_features,
    ho_labels,
    batch_size,
    test_fraction,
    seed,
    num_partitions,
    partitioner_type,
    data_directory,
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    """
    Load data using flower dataset partitioner
    """
    # initialize and get data partition
    partitioner = instantiate_partitioner(
        partitioner_type=partitioner_type,
        num_partitions=num_partitions,
        data_dir=data_directory,
        num_rows=len(tt_features) + len(ho_features),
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
    train_data = IndexedArrayDataset(
        tt_features, tt_labels, ho_features, ho_labels, train_indices
    )
    test_data = IndexedArrayDataset(
        tt_features, tt_labels, ho_features, ho_labels, test_indices
    )

    # count labels in train and test set
    train_labels = get_split_labels(tt_labels, ho_labels, train_indices)
    test_labels = get_split_labels(tt_labels, ho_labels, test_indices)

    print('Train dataset binned label counts')
    print_binned_counts(train_labels.reshape(-1, 1), np.arange(len(train_labels)))
    print('Test dataset binned label counts')
    print_binned_counts(test_labels.reshape(-1, 1), np.arange(len(test_labels)))

    # create data loaders
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_data_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    return train_data_loader, test_data_loader, train_indices, test_indices


def load_custom_partitions(
    data_partition_id,
    tt_features,
    tt_labels,
    ho_features,
    ho_labels,
    data_partitions,
    batch_size,
    test_fraction,
    seed,
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

    train_indices, test_indices = train_test_split_backed_indices(
        tt_labels,
        ho_labels,
        partition_indices,
        test_fraction,
        seed,
    )

    train_data = IndexedArrayDataset(
        tt_features, tt_labels, ho_features, ho_labels, train_indices
    )
    test_data = IndexedArrayDataset(
        tt_features, tt_labels, ho_features, ho_labels, test_indices
    )

    # count labels in train and test set
    train_labels = get_split_labels(tt_labels, ho_labels, train_indices)
    test_labels = get_split_labels(tt_labels, ho_labels, test_indices)

    print('Train dataset binned label counts')
    print_binned_counts(train_labels.reshape(-1, 1), np.arange(len(train_labels)))
    print('Test dataset binned label counts')
    print_binned_counts(test_labels.reshape(-1, 1), np.arange(len(test_labels)))

    # create dataloaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, train_indices, test_indices