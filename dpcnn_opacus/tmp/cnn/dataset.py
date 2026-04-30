from typing import List, Tuple
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
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


def pickle_params(best_params, filename):
    pickle.dump(best_params, open(f"./cnn/{filename}.dat", "wb"))


def load_pickle_data(data_path):
    cur_path = os.path.dirname(__file__).split("/tmp/")[0]
    file_patterns = ["_ohe.dat", "_tt_vcf.dat", "_tt_pheno.dat", "_ho_vcf.dat", "_ho_pheno.dat"]

    dir_path = os.path.join(cur_path, data_path)

    def load_by_pattern(pattern):
        matches = [f for f in os.listdir(dir_path) if f.endswith(pattern)]
        if not matches:
            raise FileNotFoundError(
                f"No file found in {dir_path} matching {pattern}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple files found in {dir_path} matching {pattern}: "
                f"{matches}"
            )
        with open(os.path.relpath(os.path.join(dir_path, matches[0])), "rb") as f:
            return pickle.load(f)

    ohe, tt_vcf, tt_pheno, ho_vcf, ho_pheno = [
        load_by_pattern(pattern) for pattern in file_patterns
    ]

    vcf = np.concatenate((tt_vcf, ho_vcf), axis=0)
    pheno = np.concatenate((tt_pheno, ho_pheno), axis=0)
    return ohe, vcf, pheno


def instantiate_partitioner(partitioner_type: str, num_partitions: int, data_dir=None):
    """Initialise partitioner based on selected partitioner type
    and number of partitions"""
    _, vcf, pheno = load_pickle_data(data_dir)

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
