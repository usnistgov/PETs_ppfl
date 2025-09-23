import pickle
import argparse
from typing import List
import numpy as np
from collections import Counter


def pickle_params(best_params, filename):
    pickle.dump(best_params, open(f"./cnn/{filename}.dat", "wb"))


def binned_counter(
    dataset: np.ndarray, indices: List[int], num_bins: int = 10
):
    """
    Count the occurrences of binned labels for a given indices in the dataset
    and print the counts with the bin ranges

    Args:
        dataset: np.ndarray
            The dataset to use for counting binned labels.
        indices: List[int]
            The indices to use for counting binned labels.
        num_bins: int
            The number of bins to use for binning the labels.
    Returns:
        None
    """
    # Get labels from the dataset for given indices
    labels = dataset[indices, -1]
    # Create bins for the selected labels
    bins = np.linspace(np.min(labels), np.max(labels), num_bins + 1)
    binned_labels = np.digitize(labels, bins) - 1
    # Count occurrences of each bin
    binned_counts = Counter(binned_labels)
    # Print binned label counts with ranges
    for bin_idx, count in sorted(binned_counts.items()):
        if bin_idx < len(bins) - 1:
            print(
                f"{bins[bin_idx]:.2f} - {bins[bin_idx + 1]:.2f}: "
                f"{count} records"
            )


def centralized_args_parser():
    """
    Parse arguments to define hyperparameter settings for centralized training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed used for train/test splitting (default = 42).",
    )
    parser.add_argument(
        "--test-fraction",
        default=0.2,
        type=float,
        help="Test fraction for train/test splitting (default = 0.2).",
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        help="Number of training epochs (default = 30).",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.005,
        type=float,
        help="Learning rate (default = 0.005).",
    )
    parser.add_argument(
        "--batch-divisor",
        default=30,
        type=int,
        help="Divisor to determine batch size (default = 30).",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.0001,
        type=float,
        help="Weight decay constant (default = 0.0001).",
    )
    parser.add_argument(
        "--accuracy-tolerance",
        default=0.15,
        type=float,
        help="Error tolerance to declare prediction as correct "
        "(default = 0.1). This is used for computing the "
        "accuracy of the model.",
    )
    parser.add_argument(
        "--optimizer",
        default="sgd",
        choices=["sgd", "adamax"],
        type=str,
        help="Optimizer to use sgd or adamax (default = sgd)."
    )
    parser.add_argument(
        "--opacus-secure-mode",
        default=False,
        type=bool,
        help="Use Opacus secure mode. It is set to false by default for "
             "faster experimentation (default = False)."
    )
    parser.add_argument(
        "--epsilon",
        default=1.0,
        type=float,
        help="Privacy parameter: epsilon (default = 1.0)."
    )
    parser.add_argument(
        "--delta",
        default=1e-5,
        type=float,
        help="Privacy parameter: delta (default = 1e-5)."
    )
    parser.add_argument(
        "--max-grad-norm",
        default=5.0,
        type=float,
        help="Privacy parameter: max grad norm (default = 5.0)."
             "This clips the gradients to be under this value before "
             "applying noise."
    )
    parser.add_argument(
        "--data-partitions-file",
        default=None,
        type=str,
        help=f"Path to the data partitions file (default = {None}).\n"
        "If not used, then the data is split into n_models equal parts."
        "If data-partitions file is provided, "
        "then the number of models (n_models) "
        "to train is equal to the number of "
        "data partitions available in the file. In the data partitions, "
        "each key is a client id and value for each key is a list of "
        "dataset indices to be used for that client.",
    )
    parser.add_argument(
        "--n-models",
        default=1,
        type=int,
        help="Number of models to train (default = 1). "
        "Data is split into n_models equal parts."
        "This is used when the clustered-indices file is not found.",
    )
    args = parser.parse_args()
    return args
