import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, Union
import numpy as np
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer
from xgboost import XGBClassifier, Booster
from report import Report

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
    output_dir: str | Path | None = None,
):
    out_name = (
        f"{name}_round_{round_number}" if round_number is not None else name
    )
    parent_path = Path(output_dir).absolute() if output_dir is not None else Path(__file__).parent
    parent_path.mkdir(parents=True, exist_ok=True)
    metadata_path_name = f"{out_name}_meta.npz"
    metadata_path = Path(parent_path, metadata_path_name)
    np.savez(metadata_path, **metadata, allow_pickle=True)
    report = Report(metadata)
    report.save_to_file(Path(parent_path, f"{out_name}.json"))
    print(f"Model metadata saved to {metadata_path}")
