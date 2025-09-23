from typing import List, Tuple
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
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


def load_pickle_data():
    cur_path = os.path.dirname(__file__)
    dir_path = os.path.join(cur_path, "../../..", "genetic_plant_data")
    ohe = pickle.load(
        open(os.path.relpath(os.path.join(dir_path, "Oil_QTL_ohe.dat")), "rb")
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
    return ohe, vcf, pheno


def instantiate_partitioner(partitioner_type: str, num_partitions: int):
    """Initialise partitioner based on selected partitioner type
    and number of partitions"""
    _, tt_vcf, tt_pheno = load_pickle_data()

    concat_dataset = np.concatenate((tt_vcf, tt_pheno), axis=1)
    # Dataset class works with Pandas dataframes, but not Numpy arrays
    concat_dataset_df = pd.DataFrame(concat_dataset)
    flwr_dataset = Dataset.from_pandas(concat_dataset_df)
    partitioner = CORRELATION_TO_PARTITIONER[partitioner_type](
        num_partitions=num_partitions
    )
    partitioner.dataset = flwr_dataset

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
