from typing import Tuple, List
import warnings
import numpy as np
from pathlib import Path
import pickle
import flwr as fl
from flwr.common import Context
from xgboost.core import DMatrix

from dataset import (
    load_random_partitions,
    load_custom_partitions,
    load_npy_feature_label_data,
)
from client_utils import XgbClient

# Hyper-parameters for xgboost training
NUM_LOCAL_ROUND = 1
BST_PARAMS = {
    "objective": "reg:squarederror",
    "eta": 0.1,  # Learning rate
    "max_depth": 8,
    "eval_metric": "rmse",
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}

warnings.filterwarnings("ignore", category=UserWarning)


def get_best_params():
    """Load centralized XGBoost params when available."""
    best_params = BST_PARAMS.copy()
    params_path = Path(__file__).with_name("centralized_params.dat")
    try:
        saved_params = pickle.load(open(params_path, "rb"))
    except (OSError, IOError) as e:
        print(f"no centralized_params file: {e}")
        saved_params = {}
    for key in saved_params:
        if key in best_params:
            best_params[key] = saved_params[key]
    return best_params


def loader_to_dmatrix(loader):
    features = []
    labels = []
    for batch_features, batch_labels in loader:
        features.append(np.asarray(batch_features))
        labels.append(np.asarray(batch_labels).reshape(-1))
    return DMatrix(data=np.concatenate(features), label=np.concatenate(labels))

def configure_objective(params, labels):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    integer_labels = np.all(np.equal(labels, labels.astype(int)))
    nonnegative_labels = np.all(labels >= 0)

    if len(unique_labels) <= 2 and set(unique_labels.astype(int)) <= {0, 1}:
        params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})
    elif integer_labels and nonnegative_labels:
        params.update(
            {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "num_class": int(np.max(labels)) + 1,
            }
        )
    else:
        params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})

    return params

def load_data(
    client_id: int,
    data_partitions_file,
    data_directory,
    num_partitions,
    partitioner_type,
    test_fraction,
    seed,
    batch_divisor,
):
    tt_vcf, tt_pheno, ho_vcf, ho_pheno = load_npy_feature_label_data(data_directory)
    num_data_features = tt_vcf.shape[1]
    total_rows = len(tt_vcf) + len(ho_vcf)
    batch_size = max(1, total_rows // batch_divisor)

    """
    Load data using flower dataset partitioner
    """
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

    if data_partitions is not None:
        # if data partitions available, train a model for each data partition
        train_loader, test_loader, train_indices, test_indices = (
            load_custom_partitions(
                client_id,
                tt_vcf,
                tt_pheno,
                ho_vcf,
                ho_pheno,
                data_partitions,
                batch_size,
                test_fraction,
                seed,
            )
        )
    else:
        # Partition data into n_models randomly to train N client models
        train_loader, test_loader, train_indices, test_indices = (
            load_random_partitions(
                client_id,
                tt_vcf,
                tt_pheno,
                ho_vcf,
                ho_pheno,
                batch_size,
                test_fraction,
                seed,
                num_partitions,
                partitioner_type,
                data_directory,
            )
        )

    partitions_path = (
        Path(data_partitions_file).name if data_partitions is not None else 'none'
    )

    train_dmatrix = loader_to_dmatrix(train_loader)
    test_dmatrix = loader_to_dmatrix(test_loader)

    return train_dmatrix, test_dmatrix, train_indices, test_indices


class FlowerClient:
    def __init__(self, context: Context, client_id: int, params):
        self.client_id = client_id
        self.params = params

    def to_client(self):
        seed = self.params.get('seed', 42)
        train_method = self.params.get('train_method', 'bagging')
        num_partitions = self.params.get('num_partitions', 1)
        train_dmatrix, test_dmatrix, train_indices, test_indices = load_data(
            self.client_id,
            self.params.get('data_partitions_file', None),
            self.params.get('data_dir', None),
            num_partitions,
            self.params.get('partitions_type', 'uniform'),
            self.params.get('test_fraction', 0.2),
            seed,
            self.params.get('batch_divisor', 5),
        )

        xgb_params = {**get_best_params(), 'random_state': seed}
        if train_method == "bagging" and self.params.get('scaled_lr', False):
            xgb_params["eta"] = xgb_params["eta"] / num_partitions

        return XgbClient(
            self.client_id,
            train_dmatrix,
            test_dmatrix,
            train_indices,
            test_indices,
            NUM_LOCAL_ROUND,
            xgb_params,
            train_method,
            self.params.get('data_partitions_file', None),
            self.params.get('test_fraction', 0.2),
        )
