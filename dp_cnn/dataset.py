from typing import List, Tuple
import os
import pickle
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def load_pickle_data():
    cur_path = os.path.dirname(__file__)
    dir_path = os.path.join(cur_path, "..", "genetic_plant_data")
    ohe = pickle.load(
        open(os.path.relpath(os.path.join(dir_path, "FC_QTL_ohe.dat")), "rb")
    )
    tt_vcf = pickle.load(
        open(
            os.path.relpath(os.path.join(dir_path, "FC_QTL_tt_vcf.dat")), "rb"
        )
    )
    tt_pheno = pickle.load(
        open(
            os.path.relpath(os.path.join(dir_path, "FC_QTL_tt_pheno.dat")),
            "rb",
        )
    )
    ho_vcf = pickle.load(
        open(
            os.path.relpath(os.path.join(dir_path, "FC_QTL_ho_vcf.dat")), "rb"
        )
    )
    ho_pheno = pickle.load(
        open(
            os.path.relpath(os.path.join(dir_path, "FC_QTL_ho_pheno.dat")),
            "rb",
        )
    )
    # combine traintest and ho data
    vcf = np.concatenate((tt_vcf, ho_vcf), axis=0)
    pheno = np.concatenate((tt_pheno, ho_pheno), axis=0)
    return ohe, vcf, pheno


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
