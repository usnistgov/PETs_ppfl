import re
import json
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from skopt.searchcv import BayesSearchCV
import xgboost as xgb

from utils import default_space, centralized_args_parser, save_xgb

from dataset import load_pickle_data, train_test_indices_split

# Parser arguments for hyperparameters
args = centralized_args_parser()
print(f"Commandline arguments: {args}")
seed = args.seed
test_frac = args.test_fraction
data_partitions_file = args.data_partitions_file
n_models = args.n_models

# Setting Space to Optimise
space = default_space

ohe, vcf, pheno = load_pickle_data()
print(f"data (tt + ho) shape: {vcf.shape}")
print(f"labels (tt + ho) shape: {pheno.shape}")
combined_dataset = np.concatenate((vcf, pheno), axis=1)

# Check if data partitions file is provided and exists
data_partitions = None
if data_partitions_file and Path(data_partitions_file).exists():
    data_partitions = np.load(data_partitions_file)
    print(f"Loaded data partitions from {data_partitions_file}")
else:
    print(
        f"Data partitions file not found at {data_partitions_file}. "
        f"Using n_models = {n_models}"
    )


def train_partition(
    model_id: int, combined_dataset: np.ndarray, indices: np.ndarray
):
    train_indices, test_indices = train_test_indices_split(
        combined_dataset, indices, test_frac, seed
    )
    # Split into train and test based on the indices
    train_dataset = combined_dataset[train_indices]
    test_dataset = combined_dataset[test_indices]

    # count labels in train and test set
    train_class_counts = Counter(train_dataset[:, -1])
    test_class_counts = Counter(test_dataset[:, -1])
    print(
        f"Train class counts: {train_class_counts}, "
        f"Test class counts: {test_class_counts}"
    )

    # Look for optimum parameters from the defined space and print best params
    xgbcl = xgb.XGBClassifier(seed=seed)
    bayes_params = {
        "n_iter": 32,
        "scoring": None,
        "n_jobs": 1,
        "cv": 5,
        "verbose": 3,
        "random_state": seed,
        "n_points": 12,
        "refit": True,
    }

    xgb_bayes_search = BayesSearchCV(
        xgbcl,
        space,
        n_iter=bayes_params["n_iter"],
        scoring=bayes_params["scoring"],
        n_jobs=bayes_params["n_jobs"],
        cv=bayes_params["cv"],
        verbose=bayes_params["verbose"],
        random_state=bayes_params["random_state"],
        n_points=bayes_params["n_points"],
        refit=bayes_params["refit"],
    )

    # search for best params
    xgb_bayes_search.fit(train_dataset[:, :-1], train_dataset[:, -1])

    # declare best model
    model = xgb.XGBClassifier(
        **{**xgb_bayes_search.best_params_, "seed": seed}
    )
    model.fit(train_dataset[:, :-1], train_dataset[:, -1], verbose=True)

    # Use model to find train and test data accuracy
    y_train_pred = model.predict(train_dataset[:, :-1])
    predictions = [round(value) for value in y_train_pred]
    train_acc = accuracy_score(train_dataset[:, -1], predictions)

    y_test_pred = model.predict(test_dataset[:, :-1])
    predictions = [round(value) for value in y_test_pred]
    test_acc = accuracy_score(test_dataset[:, -1], predictions)

    print(f"Train accuracy: {train_acc * 100:.2f}%")
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    partitions_path = (
        Path(data_partitions_file).name if data_partitions else 'none'
    )
    metadata = {
        'created on': str(datetime.now()),
        'model id': int(model_id),
        'partitions file': partitions_path,
        'train accuracy': float(train_acc),
        'test accuracy': float(test_acc),
        'train loss': 0,
        'test loss': 0,
        'train indices': train_indices,
        'test indices': test_indices,
        'hyperparameters': {
            **xgb_bayes_search.best_params_,
            "seed": seed,
            "test_fraction": test_frac,
        },
    }

    metadata['hyperparameters'] = json.dumps(metadata['hyperparameters'])
    save_xgb(model, metadata, f'xgb_{model_id}')


def train_custom_partitions():
    """Create models for each given data partitions"""
    data_partition_ids = sorted(
        [k for k in data_partitions.keys() if re.match(r'client_\d+', k)]
    )

    # Train a model for each given data partitions
    for partition_id in data_partition_ids:
        model_id = partition_id.split('_')[-1]
        partition_indices = data_partitions[partition_id]
        print(
            f"Training model {model_id} "
            f"with {len(partition_indices)} samples"
        )
        train_partition(model_id, combined_dataset, partition_indices)


def train_random_partitions():
    """Create models for randomly partitioned data."""
    X, y = combined_dataset[:, :-1], combined_dataset[:, -1]
    dataset_indices = np.arange(len(X))
    print()
    if n_models == 1:
        # Training only single model, so partition is required
        indices = dataset_indices  # Use all indices
        print(f"Training model 1/1 with {len(indices)} samples")
        train_partition(1, combined_dataset, indices)
    else:
        # Partition data using StratifiedKFold where n_splits = n_models
        skf = StratifiedKFold(
            n_splits=n_models, shuffle=True, random_state=seed
        )

        # Create stratified dataset splits
        splits = list(skf.split(dataset_indices, y))

        # Create a model for each dataset split
        for i, (_, indices) in enumerate(splits):
            print(
                f"Training model {i + 1}/{n_models} "
                f"with {len(indices)} samples"
            )
            train_partition(i, combined_dataset, indices)


# Check if data partitions are provided
if data_partitions is not None:
    # if data partitions available, train a model for each data partition
    train_custom_partitions()
else:
    # Partition data into n_models randomly to train N client models
    train_random_partitions()
