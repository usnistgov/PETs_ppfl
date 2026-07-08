# This Software (PETs Testbed) is being made available as a public service by the
# National Institute of Standards and Technology (NIST), an Agency of the United
# States Department of Commerce. This software was developed in part by employees of
# NIST and in part by NIST contractors. Copyright in portions of this software that
# were developed by NIST contractors has been licensed or assigned to NIST. Pursuant
# to Title 17 United States Code Section 105, works of NIST employees are not
# subject to copyright protection in the United States. However, NIST may hold
# international copyright in software created by its employees and domestic
# copyright (or licensing rights) in portions of software that were assigned or
# licensed to NIST. To the extent that NIST holds copyright in this software, it is
# being made available under the Creative Commons Attribution 4.0 International
# license (CC BY 4.0). The disclaimers of the CC BY 4.0 license apply to all parts
# of the software developed or licensed by NIST.
#
# ACCESS THE FULL CC BY 4.0 LICENSE HERE:
# https://creativecommons.org/licenses/by/4.0/legalcode

from typing import List, Tuple
import re
import sys
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
import bisect

from torch.utils.data import DataLoader
from flwr_datasets.partitioner import (
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)
from utils import print_binned_counts
from torch.utils.data import Dataset as TorchDataset
import warnings

NPY_SUFFIXES = ("_tt_vcf", "_tt_pheno", "_ho_vcf", "_ho_pheno", "_pub_vcf", "_pub_pheno")

class IndexedArrayDataset(TorchDataset):
    """Lazy row lookup over a logical concatenation of multiple datasets.

    The backing arrays stay separate and mmap-backed. Global row indices follow
    the order of `features_list` / `labels_list`.
    """

    def __init__(self, features_list, labels_list, indices, label_to_index=None):
        if len(features_list) != len(labels_list):
            raise ValueError("features_list and labels_list must have the same length")

        self.features_list = list(features_list)
        self.labels_list = [labels.reshape(-1) for labels in labels_list]

        lengths = [len(features) for features in self.features_list]

        for i, (features, labels) in enumerate(zip(self.features_list, self.labels_list)):
            if len(features) != len(labels):
                raise ValueError(f"features_list[{i}] and labels_list[{i}] have different lengths")

        self.offsets = np.cumsum(lengths)
        self.indices = np.asarray(indices)
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row_idx = int(self.indices[idx])

        array_idx = bisect.bisect_right(self.offsets, row_idx)
        start = 0 if array_idx == 0 else self.offsets[array_idx - 1]
        local_idx = row_idx - start

        label = self.labels_list[array_idx][local_idx]
        return self.features_list[array_idx][local_idx], self.encode_label(label)

    def encode_label(self, label):
        if self.label_to_index is None:
            return label
        key = normalize_label(label)
        if key not in self.label_to_index:
            raise ValueError(f"Label {key!r} is not present in class_labels")
        return self.label_to_index[key]

CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}


def normalize_label(label):
    return label.item() if hasattr(label, "item") else label


def build_label_to_index(class_labels):
    if class_labels is None:
        return None
    return {normalize_label(label): i for i, label in enumerate(class_labels)}

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
        warnings.warn(
            "Missing .npy files after conversion: "
            f"{', '.join(still_missing)} in {data_dir}"
        )

    return data_dir

"""
This file converts .dat files into .npy files for more efficient loading into shared memory. This will not overwrite the .dat files.
Instead, this will create new .npy files with the same file prefixes as the .dat files. 
"""
def convert_dat_to_npy(data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_dir}")

    dat_files = sorted(data_dir.glob("*.dat"))
    npy_files = sorted(data_dir.glob("*.npy"))

    if not dat_files and not npy_files:
        print(f"No .dat files or .npy files found in {data_dir}. Please add in .dat data for conversions.")
        return

    for dat_path in dat_files:
        out_path = dat_path.with_suffix(".npy")

        if out_path.exists():
            print(f"Already exists: {out_path.name}")
            continue

        with dat_path.open("rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, np.ndarray):
            print(f"Skipping {dat_path.name}: not a NumPy array ({type(obj)})")
            continue

        arr = obj.astype(np.float32, copy=False) if obj.dtype == np.float64 else obj

        np.save(out_path, arr)
        print(f"Converted: {dat_path.name} -> {out_path.name} {arr.shape} {arr.dtype}")

def get_split_labels(labels, indices):
    indices = np.asarray(indices)
    label_arrays = [np.asarray(label).reshape(-1) for label in labels]
    lengths = [len(label) for label in label_arrays]
    total_len = sum(len(label) for label in labels)
    if len(indices) and (indices.min() < 0 or indices.max() >= total_len):
        raise IndexError(
            f"Partition index range [{indices.min()}, {indices.max()}] "
            f"is outside logical dataset size {total_len}."
        )

    offsets = np.cumsum(lengths)
    out = np.empty(len(indices), dtype=label_arrays[0].dtype)

    for array_idx, label_array in enumerate(label_arrays):
        start = 0 if array_idx == 0 else offsets[array_idx - 1]
        stop = offsets[array_idx]

        mask = (indices >= start) & (indices < stop)
        out[mask] = label_array[indices[mask] - start]

    return out

def train_test_split_backed_indices(labels, indices, test_frac, seed, problem_type):
    selected_labels = get_split_labels(labels, indices)
    local_indices = np.arange(len(selected_labels))

    local_train, local_test = train_test_indices_split(
        selected_labels.reshape(-1, 1),
        local_indices,
        test_frac,
        seed,
        problem_type,
    )

    indices = np.asarray(indices)
    return indices[local_train].tolist(), indices[local_test].tolist()

def load_npy_feature_label_data(data_path):
    """Load tt/ho feature and label arrays as read-only mmap-backed arrays."""
    data_dir = resolve_data_dir(data_path)

    def load_one(suffix):
        path = find_single_npy(data_dir, suffix)
        if path is None:
            warnings.warn(f"No .npy file found in {data_dir} matching *{suffix}.npy")
            return np.array([])
        return np.load(path, mmap_mode="r")

    tt_vcf = load_one("tt_vcf")
    tt_pheno = load_one("tt_pheno")
    ho_vcf = load_one("ho_vcf")
    ho_pheno = load_one("ho_pheno")
    pub_vcf = load_one("pub_vcf")
    pub_pheno = load_one("pub_pheno")

    return tt_vcf, tt_pheno.reshape(-1), ho_vcf, ho_pheno.reshape(-1), pub_vcf, pub_pheno.reshape(-1)

def instantiate_partitioner(partitioner_type, num_partitions, data_dir, num_rows=None):
    if num_rows is None:
        tt_features, _, ho_features, _, pub_features, _ = load_npy_feature_label_data(data_dir)
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


def train_test_indices_split(dataset: np.ndarray, indices: np.ndarray, test_frac: float, seed: int, problem_type: str) -> Tuple[List[int], List[int]]:
    """
    Split dataset indices into train and test sets for regression tasks.
    Continuous labels are split into train and test sets using stratified
    sampling based on quantile bins. If any bin has less than 2 samples,
    they are added to the training set, and the sufficient indices are
    stratified split.
    """
    # Get labels from the dataset
    labels = dataset[indices, -1]

    if problem_type == "classification":
        binned_labels = [int(x) for x in np.asarray(labels)]
    else:
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
    features,
    labels,
    batch_size,
    test_fraction,
    seed,
    num_partitions,
    partitioner_type,
    data_directory,
    problem_type,
    class_labels=None,
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    """
    Load data using flower dataset partitioner
    """
    # initialize and get data partition
    partitioner = instantiate_partitioner(
        partitioner_type=partitioner_type,
        num_partitions=num_partitions,
        data_dir=data_directory,
        num_rows=sum(len(sublist) for sublist in features),
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
    label_to_index = (
        build_label_to_index(class_labels)
        if problem_type == "classification"
        else None
    )
    train_data = IndexedArrayDataset(
        features, labels, train_indices, label_to_index
    )
    test_data = IndexedArrayDataset(
        features, labels, test_indices, label_to_index
    )

    # count labels in train and test set
    train_labels = get_split_labels(labels, train_indices)
    test_labels = get_split_labels(labels, test_indices)
    if problem_type == "classification":
        print(f"Client {data_partition_id}: Train dataset label counts")
        train_counts = Counter(int(x) for x in np.asarray(train_labels))
        for cls, count in sorted(train_counts.items()):
            print(f"class {cls}: {count} records")
        print(f"Client {data_partition_id}: Test dataset label counts")
        test_counts = Counter(int(x) for x in np.asarray(test_labels))
        for cls, count in sorted(test_counts.items()):
            print(f"class {cls}: {count} records")
    else:
        print(f'Client {data_partition_id}: Train dataset binned label counts')
        print_binned_counts(train_labels.reshape(-1, 1), np.arange(len(train_labels)))
        print(f'Client {data_partition_id}: Test dataset binned label counts')
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
    features,
    labels,
    data_partitions,
    batch_size,
    test_fraction,
    seed,
    problem_type,
    class_labels=None,
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
        labels,
        partition_indices,
        test_fraction,
        seed,
        problem_type,
    )

    label_to_index = (
        build_label_to_index(class_labels)
        if problem_type == "classification"
        else None
    )
    train_data = IndexedArrayDataset(
        features, labels, train_indices, label_to_index
    )
    test_data = IndexedArrayDataset(
        features, labels, test_indices, label_to_index
    )

    # count labels in train and test set
    train_labels = get_split_labels(labels, train_indices)
    test_labels = get_split_labels(labels, test_indices)
    if problem_type == "classification":
        print(f"Client {data_partition_id}: Train dataset label counts")
        train_counts = Counter(int(x) for x in np.asarray(train_labels))
        for cls, count in sorted(train_counts.items()):
            print(f"class {cls}: {count} records")
        print(f"Client {data_partition_id}: Test dataset label counts")
        test_counts = Counter(int(x) for x in np.asarray(test_labels))
        for cls, count in sorted(test_counts.items()):
            print(f"class {cls}: {count} records")
    else:
        print(f'Client {data_partition_id}: Train dataset binned label counts')
        print_binned_counts(train_labels.reshape(-1, 1), np.arange(len(train_labels)))
        print(f'Client {data_partition_id}: Test dataset binned label counts')
        print_binned_counts(test_labels.reshape(-1, 1), np.arange(len(test_labels)))

    # create dataloaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, train_indices, test_indices
