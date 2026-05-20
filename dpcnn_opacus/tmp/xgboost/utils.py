import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, Union
import numpy as np
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer
from xgboost import XGBClassifier, Booster

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

# Setting Space to Optimize
default_space = {
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
    'max_depth': Integer(0, 50),
    'max_delta_step': Integer(0, 20),
    'subsample': Real(0.01, 1.0, 'uniform'),
    'colsample_bytree': Real(0.01, 1.0, 'uniform'),
    'colsample_bylevel': Real(0.01, 1.0, 'uniform'),
    'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
    'reg_alpha': Real(1e-9, 1.0, 'log-uniform'),
    'gamma': Real(1e-9, 0.5, 'log-uniform'),
    'min_child_weight': Integer(0, 5),
    'n_estimators': Integer(50, 200),
    'scale_pos_weight': Real(1e-6, 500, 'log-uniform'),
}


def pickle_params(best_params, filename):
    pickle.dump(best_params, open(f"./xgboost/{filename}.dat", "wb"))


def load_pickle_data():
    cur_path = os.path.dirname(__file__)
    dir_path = os.path.relpath("./genetic_plant_data", cur_path)

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

    return tt_vcf, tt_pheno


def get_best_params():
    """if available, get params saved from centralized training,
    otherwise use default_space"""
    best_params = BST_PARAMS
    try:
        saved_params = pickle.load(open("./centralized_params.dat", "rb"))
    except (OSError, IOError) as e:
        print(f"no centralized_params file: {e}")
        saved_params = {}
    for key in saved_params:
        if key in best_params:
            best_params[key] = saved_params[key]

    return best_params


def get_tt_data(seed):
    tt_vcf, tt_pheno = load_pickle_data()

    return train_test_split(tt_vcf, tt_pheno, test_size=0.2, random_state=seed)


def save_xgb(
    model: Union[XGBClassifier, Booster],
    metadata: Dict[str, any],
    name: str,
    round_number: int | None = None,
):
    round_number = f"round_{round_number}" if round_number is not None else ''
    parent_path = Path(__file__).parent
    if round_number:
        model_path = Path(parent_path, f"{name}_{round_number}.json")
    else:
        model_path = Path(parent_path, f"{name}.json")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    metadata_path_name = f"{name}_meta.npz"
    metadata_path = Path(parent_path, metadata_path_name)
    if not metadata_path.exists():
        np.savez(metadata_path, **metadata)
        print(f"Model metadata saved to {metadata_path}")


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
        "--data-partitions-file",
        default=None,
        type=str,
        help=f"Path to the data partitions file (default = {None}).\n"
        "If not used, then the data is split into n_models equal parts."
        "If data partitions file is provided, "
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
        "This is used when the data partitions file is not found.",
    )

    args = parser.parse_args()
    return args


def server_args_parser():
    """Parse arguments to define experimental settings on server side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-method",
        default="bagging",
        type=str,
        choices=["bagging", "cyclic"],
        help="Training methods selected from bagging aggregation "
        "or cyclic training (default = 'bagging').",
    )
    parser.add_argument(
        "--pool-size",
        default=1,
        type=int,
        help="Number of total clients (default = 1).",
    )
    parser.add_argument(
        "--num-rounds",
        default=10,
        type=int,
        help="Number of FL rounds (default = 10).",
    )
    parser.add_argument(
        "--num-clients-per-round",
        default=1,
        type=int,
        help="Number of clients participate in "
        "training each round (default = 1).",
    )
    parser.add_argument(
        "--num-evaluate-clients",
        default=1,
        type=int,
        help="Number of clients selected for evaluation (default = 1).",
    )
    parser.add_argument(
        "--centralised-eval",
        action="store_true",
        help="Conduct centralised evaluation (True), "
        "or client evaluation on hold-out data (False).",
    )

    args = parser.parse_args()
    return args


def client_args_parser():
    """Parse arguments to define experimental settings on client side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-method",
        default="bagging",
        type=str,
        choices=["bagging", "cyclic"],
        help="Training methods selected from bagging aggregation "
        "or cyclic training (default = 'bagging').",
    )
    parser.add_argument(
        "--num-partitions",
        default=1,
        type=int,
        help="Number of partitions (default = 1).",
    )
    parser.add_argument(
        "--partitioner-type",
        default="uniform",
        type=str,
        choices=["uniform", "linear", "square", "exponential"],
        help="Partitioner types (default = 'uniform').",
    )
    parser.add_argument(
        "--client-id",
        default=0,
        type=int,
        help="client ID used for the current client (default = 0).",
    )
    parser.add_argument(
        "--partition-id",
        default=0,
        type=int,
        help="Partition ID used for the current client (default = 0).",
    )
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
        "--centralised-eval",
        action="store_true",
        help="Conduct evaluation on centralized test set (True), "
        "or on hold-out data (False).",
    )
    parser.add_argument(
        "--scaled-lr",
        action="store_true",
        help="Perform scaled learning rate "
        "based on the number of clients (True).",
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

    args = parser.parse_args()
    return args
