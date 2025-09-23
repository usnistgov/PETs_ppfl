from typing import List, Tuple
import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from datasets import Dataset
from torch.utils.data.dataset import Dataset as TorchDataset
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader
from flwr_datasets.partitioner import (
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)


CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}


class IndexedDataset(TorchDataset):
    def __init__(self, data, indices):
        self._data = data  # torch.Tensor or np.ndarray
        self._indices = indices  # list of original indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        orig_idx = self._indices[idx]
        sample = self._data[idx]
        return sample, orig_idx


def load_pickle_data():
    cur_path = os.path.dirname(__file__)
    dir_path = os.path.join(
        cur_path, "../../..", "genetic_plant_data/gpd_oil_binned5"
    )
    bin_ranges = pickle.load(
        open(
            os.path.relpath(os.path.join(dir_path, "Oil_QTL_pheno_bins.dat")),
            "rb",
        )
    )
    tt_vcf = pickle.load(
        open(
            os.path.relpath(os.path.join(dir_path, "Oil_QTL_tt_vcf.dat")), "rb"
        )
    )
    tt_pheno = pickle.load(
        open(
            os.path.relpath(os.path.join(dir_path, "Oil_QTL_tt_pheno.dat")),
            "rb",
        )
    )
    ho_vcf = pickle.load(
        open(
            os.path.relpath(os.path.join(dir_path, "Oil_QTL_ho_vcf.dat")), "rb"
        )
    )
    ho_pheno = pickle.load(
        open(
            os.path.relpath(os.path.join(dir_path, "Oil_QTL_ho_pheno.dat")),
            "rb",
        )
    )
    # combine traintest and ho data
    vcf = np.concatenate((tt_vcf, ho_vcf), axis=0)
    pheno = np.concatenate((tt_pheno, ho_pheno), axis=0)
    print('ALL ROWS', vcf.shape)
    return vcf, pheno, bin_ranges


def instantiate_partitioner(partitioner_type: str, num_partitions: int):
    """Initialise partitioner based on selected partitioner type
    and number of partitions"""
    vcf, pheno, _ = load_pickle_data()

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


def print_bin_ranges(data, bins_range):
    labels_count = Counter(data[:, -1])
    for bin_id in sorted(labels_count):
        low = bins_range[int(bin_id)]
        high = bins_range[int(bin_id) + 1]
        print(
            f"Bin {int(bin_id)}: {labels_count[bin_id]} "
            f"samples, range [{low:.2f}, {high:.2f})"
        )


def print_binned_counts(train_class_counts, test_class_counts):
    print('Training Samples Distribution')
    for bin_id in sorted(train_class_counts):
        print(f"Bin {int(bin_id)}: {train_class_counts[bin_id]} samples")
    print('Testing Samples Distribution')
    for bin_id in sorted(test_class_counts):
        print(f"Bin {int(bin_id)}: {test_class_counts[bin_id]} samples")
    print()


def train_test_indices_split(
    dataset: np.ndarray, indices: np.ndarray, test_frac: float, seed: int
) -> Tuple[List[int], List[int]]:
    """
    Split dataset indices into train and test sets using stratified sampling.
    If any label has less than 2 samples, it will split labels with more
    than 2 samples using random sampling and test split is combined with the
    samples that has the label with than 2 samples.
    """
    # Get labels/classes from the combined dataset
    labels = dataset[indices, -1]
    # Count occurrences of each label
    label_counts = Counter(labels)

    # Find labels with less than 2 samples
    insufficient_labels = [
        cls for cls, count in label_counts.items() if count < 2
    ]

    # Split data samples into insufficient and sufficient labels groups.
    # insufficient indices are those that has label count less than 2
    insufficient_indices = [
        i for i in indices if dataset[i, -1] in insufficient_labels
    ]
    sufficient_indices = [
        i for i in indices if dataset[i, -1] not in insufficient_labels
    ]
    sufficient_labels = dataset[sufficient_indices, -1]

    # if there are any labels with less than 2 samples
    if len(insufficient_labels):
        print(
            'Random split: ',
            f'insufficient indices: {len(insufficient_indices)}, '
            f'sufficient indices: {len(sufficient_indices)}, {label_counts}',
        )
        # Randomly split sufficient indices into train and test sets
        train_indices, test_indices = train_test_split(
            sufficient_indices, test_size=test_frac, random_state=seed
        )
        train_indices += insufficient_indices
    else:  # if all labels have more than 2 samples
        print(
            'Stratified split: ',
            f'insufficient indices: {len(insufficient_indices)}, '
            f'sufficient indices: {len(sufficient_indices)}, {label_counts}',
        )
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_frac, random_state=seed
        )
        train_sufficient, test_sufficient = next(
            sss.split(sufficient_indices, sufficient_labels)
        )

        # Map back to original indices
        train_indices = np.array(sufficient_indices)[train_sufficient].tolist()
        test_indices = np.array(sufficient_indices)[test_sufficient].tolist()
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

    # shuffle train_indices
    rng = np.random.default_rng()
    train_indices = list(rng.permutation(train_indices))

    # Split into train and test based on the indices
    train_data = combined_dataset[train_indices]
    test_data = combined_dataset[test_indices]

    # count labels in train and test set
    train_class_counts = Counter(train_data[:, -1])
    test_class_counts = Counter(test_data[:, -1])

    print_binned_counts(train_class_counts, test_class_counts)

    train_idxd_dataset = IndexedDataset(train_data, train_indices)
    test_idxd_dataset = IndexedDataset(test_data, test_indices)

    # create data loaders
    train_loader = DataLoader(
        train_idxd_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_idxd_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, train_indices, test_indices


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
            f"Cannot use client {data_partition_id} for "
            f"training because it was "
            f"not found in the data partitions."
        )
        sys.exit(1)

    # get data partition indices and create train test split
    data_partition_str_id = data_partition_ids[data_partition_id]
    partition_indices = data_partitions[data_partition_str_id]

    train_indices, test_indices = train_test_indices_split(
        combined_dataset, partition_indices, test_fraction, seed
    )

    # shuffle train_indices
    rng = np.random.default_rng()
    train_indices = list(rng.permutation(train_indices))

    # Split into train and test based on the indices
    train_dataset = combined_dataset[train_indices]
    test_dataset = combined_dataset[test_indices]

    # count labels in train and test set
    train_class_counts = Counter(train_dataset[:, -1])
    test_class_counts = Counter(test_dataset[:, -1])

    # count labels in train and test set
    print_binned_counts(train_class_counts, test_class_counts)

    train_idxd_dataset = IndexedDataset(train_dataset, train_indices)
    test_idxd_dataset = IndexedDataset(test_dataset, test_indices)

    # create dataloaders
    train_loader = DataLoader(
        train_idxd_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_idxd_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, train_indices, test_indices
