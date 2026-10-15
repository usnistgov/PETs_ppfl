# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software

from typing import List, Tuple
from typing import Dict
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
import flwr as fl
from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters
from pathlib import Path
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic
from xgboost.core import DMatrix
from dataset import IndexedArrayDataset, load_npy_feature_label_data, build_label_to_index, normalize_label
from model import BaseModel, CNNModel, DPCNNModel, XGBoostModel
from report import Report
from utils import get_device, print_binned_counts

def get_torch_model_class(model_type: str):
    """Return the Torch model class for the configured model type."""
    return DPCNNModel if model_type == "dpcnn" else CNNModel

def update_global_history(problem_type, loss, accuracy, mse, classification_metrics=None):
    """Append one round of global metrics to in-memory history."""
    loss_rounds.append(float(loss))
    accuracy_rounds.append(float(accuracy))
    mse_rounds.append(float(mse))

    if problem_type == "classification" and classification_metrics is not None:
        precision_rounds.append(float(classification_metrics["precision_macro"]))
        recall_rounds.append(float(classification_metrics["recall_macro"]))

def save_global_outputs(output_dir: str, prefix: str, problem_type: str, class_labels,
                        accuracy_tolerance: float, save_model_fn):
    
    """Save global model metadata, report files, and model weights."""
    metadata = {"created on": str(datetime.now())}
    BaseModel.add_task_report_metadata(metadata, problem_type, class_labels, accuracy_tolerance)
    metadata.update({"loss per round": np.array(loss_rounds), "accuracy per round": np.array(accuracy_rounds)})

    if problem_type == "classification":
        BaseModel.add_global_classification_report_metrics(metadata, precision_rounds, recall_rounds)
    else:
        metadata["mse per round"] = np.array(mse_rounds)

    np.savez(Path(output_dir, f"{prefix}_global_metadata.npz"), **metadata)
    Report(metadata).save_to_file(Path(output_dir, f"{prefix}_global_metadata.json"))
    save_model_fn(Path(output_dir, f"{prefix}_global"))

def build_test_data(strategy_params):
    """Build the holdout evaluation data for the server strategy."""
    _, _, ho_vcf, ho_pheno, _, _ = load_npy_feature_label_data(strategy_params["data_dir"])
    all_indices = np.arange(len(ho_vcf))
    num_data_features = ho_vcf.shape[1]
    
    label_to_index = (
        build_label_to_index(strategy_params.get("class_labels"))
        if strategy_params.get("problem_type") == "classification"
        else None
    )

    if strategy_params.get("model_type") == "xgboost":
        # XGBoost bypasses IndexedArrayDataset, so encode its labels here.
        xgb_labels = ho_pheno
        if label_to_index is not None:
            xgb_labels = np.asarray([label_to_index[normalize_label(label)] for label in ho_pheno])
        test_loader = DMatrix(data=ho_vcf, label=xgb_labels)
    else:
        # Keep raw labels: IndexedArrayDataset encodes each label exactly once.
        test_dataset = IndexedArrayDataset([ho_vcf], [ho_pheno], all_indices, label_to_index)
        test_loader = DataLoader(test_dataset, batch_size=strategy_params["batch_size"], shuffle=False)

    # count labels in train and test set
    print("\n\nHOLDOUT DATASET FOR EVALUATION")
    if strategy_params.get("problem_type") == "classification":
        print(f"Holdout dataset label counts")
        test_counts = Counter(int(x) for x in np.asarray(ho_pheno))
        for cls, count in sorted(test_counts.items()):
            print(f"class {cls}: {count} records")
    else:
        print(f'Holdout dataset binned label counts')
        print_binned_counts(ho_pheno.reshape(-1, 1), np.arange(len(ho_pheno)))
    print("\n\n")

    return num_data_features, test_loader

def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client metrics using example-count weighted averages."""
    metrics = [(num, dict(m)) for num, m in metrics if num > 0 and len(m) > 0]
    total = sum(num for num, _ in metrics)
    if total == 0:
        return {}

    names = sorted({name for _, m in metrics for name in m.keys()})
    return {
        name: sum(num * m.get(name, 0.0) for num, m in metrics) / total
        for name in names
    }

loss_rounds = []  # loss per global round
accuracy_rounds = []  # accuracy per global round
mse_rounds = []  # mse per global round
precision_rounds = []  # macro precision per global round
recall_rounds = []  # macro recall per global round

def get_evaluate_fn(
    num_data_features: int,
    num_rounds: int,
    test_loader: DataLoader,
    output_dir: str,
    model_type: str,
    problem_type: str,
    class_labels=None,
    num_classes: int = 1,
    accuracy_tolerance: float = 0.1,
):
    """Return the global evaluation callback for Torch strategies."""

    def evaluate_fn(server_round: int, parameters, config):
        """Evaluate the global Torch model for one server round."""
        if server_round == 0:
            update_global_history(problem_type, 0.0, 0.0, 0.0, {
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            } if problem_type == "classification" else None)
            return 0.0, {"accuracy": 0.0, "mse": 0.0}

        model_class = get_torch_model_class(model_type)
        global_model = model_class(
            model_id="global",
            output_dir=output_dir,
            problem_type=problem_type,
        )
        output_dim = num_classes if problem_type == "classification" else 1
        global_model.build_model(num_data_features, output_dim=output_dim)
        global_model.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss() if problem_type == "classification" else nn.MSELoss()

        test_mse, test_accuracy, test_loss, test_class_metrics = global_model.compute_test_metrics(
            test_loader,
            criterion,
            get_device(),
            accuracy_tolerance=accuracy_tolerance,
        )

        update_global_history(
            problem_type,
            test_loss,
            test_accuracy,
            test_mse,
            test_class_metrics if problem_type == "classification" else None,
        )

        print(
            "GLOBAL ACCURACY:",
            test_accuracy,
            "GLOBAL MSE:",
            test_mse,
            "SERVER ROUND:",
            server_round,
        )

        if server_round == num_rounds:
            prefix = "dpcnn_opacus" if model_type == "dpcnn" else "cnn"
            save_global_outputs(
                output_dir=output_dir,
                prefix=prefix,
                problem_type=problem_type,
                class_labels=class_labels,
                accuracy_tolerance=accuracy_tolerance,
                save_model_fn=lambda base_path: torch.save(
                    global_model.model.state_dict(),
                    Path(f"{base_path}.torch"),
                ),
            )

        return float(test_loss), {
            "accuracy": float(test_accuracy),
            "mse": float(test_mse),
        }

    return evaluate_fn

def fit_round(server_round: int):
    """Configure client fit calls for a server round."""
    return {"server_round": server_round}

def evaluate_and_save_xgboost_global(
    server_round: int,
    parameters,
    num_rounds: int,
    test_data: DMatrix,
    output_dir: str,
    xgboost_params: Dict,
    problem_type: str,
    class_labels=None,
    accuracy_tolerance: float = 0.1,
):
    """Evaluate and save the aggregated XGBoost global model."""
    if parameters is None or not parameters.tensors:
        return 0.0, {"accuracy": 0.0, "mse": 0.0}

    global_model = XGBoostModel(
        model_id="global",
        output_dir=output_dir,
        params=xgboost_params or {},
        problem_type=problem_type,
        class_labels=class_labels,
        accuracy_tolerance=accuracy_tolerance,
    )
    global_model.set_parameters([parameters.tensors[-1]])

    labels = test_data.get_label()
    predictions = global_model.predict(global_model.model, test_data)
    metrics = global_model.task.metrics(labels, predictions, accuracy_tolerance)

    accuracy = float(metrics["accuracy"])
    mse = float(metrics.get("mse", 0.0))
    loss = global_model.evaluate(test_data)[1] if problem_type == "classification" else mse

    update_global_history(
        problem_type,
        loss,
        accuracy,
        mse,
        metrics if problem_type == "classification" else None,
    )

    print(
        "GLOBAL ACCURACY:",
        accuracy,
        "GLOBAL MSE:",
        mse,
        "SERVER ROUND:",
        server_round,
    )

    if server_round == num_rounds:
        save_global_outputs(
            output_dir=output_dir,
            prefix="xgb",
            problem_type=problem_type,
            class_labels=class_labels,
            accuracy_tolerance=accuracy_tolerance,
            save_model_fn=lambda base_path: global_model.model.save_model(
                Path(f"{base_path}.ubj")
            ),
        )

    return float(loss), {
        "accuracy": float(accuracy),
        "mse": float(mse),
    }

class GlobalOutputFedXgbCyclic(FedXgbCyclic):
    def __init__(self, *args, global_output_config=None, **kwargs):
        """Initialize the GlobalOutputFedXgbCyclic instance."""
        super().__init__(*args, **kwargs)
        self.global_output_config = global_output_config or {}

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate cyclic XGBoost updates and save global outputs."""
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        evaluate_and_save_xgboost_global(
            server_round,
            parameters,
            **self.global_output_config,
        )
        return parameters, metrics

def create_xgboost_strategy(strategy_params):
    """Create the Flower strategy for XGBoost federation."""
    def xgboost_round_config(server_round: int) -> Dict[str, str]:
        """Return XGBoost round metadata for clients."""
        return {"global_round": str(server_round)}
    
    def get_xgboost_evaluate_fn(global_output_config):
        """Return the global evaluation callback for XGBoost bagging."""

        def evaluate_fn(server_round, parameters, config):
            """Evaluate the global XGBoost model for one server round."""
            return evaluate_and_save_xgboost_global(
                server_round,
                parameters,
                **global_output_config,
            )

        return evaluate_fn
    
    train_method = strategy_params.get("train_method")
    pool_size = strategy_params["num_clients"]
    min_fit_clients = strategy_params["num_clients"]
    min_evaluate_clients = strategy_params["num_clients"]
    centralized_eval = strategy_params.get("centralized_eval")
    _, test_data = build_test_data(strategy_params)
    global_output_config = {
        "num_rounds": strategy_params["num_rounds"],
        "test_data": test_data,
        "output_dir": strategy_params["output_dir"],
        "xgboost_params": strategy_params.get("xgboost_params") or {},
        "problem_type": strategy_params.get("problem_type", "regression"),
        "class_labels": strategy_params.get("class_labels"),
        "accuracy_tolerance": strategy_params.get("accuracy_tolerance"),
    }

    if train_method == "bagging":
        return FedXgbBagging(
            evaluate_function=get_xgboost_evaluate_fn(global_output_config),
            fraction_fit=1.0,
            min_fit_clients=min_fit_clients,
            min_available_clients=pool_size,
            min_evaluate_clients=(
                min_evaluate_clients if not centralized_eval else 0
            ),
            fraction_evaluate=1.0 if not centralized_eval else 0.0,
            on_evaluate_config_fn=xgboost_round_config,
            on_fit_config_fn=xgboost_round_config,
            evaluate_metrics_aggregation_fn=weighted_average_metrics if not centralized_eval else None
        )

    return GlobalOutputFedXgbCyclic(
        fraction_fit=1.0,
        min_available_clients=pool_size,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
        on_evaluate_config_fn=xgboost_round_config,
        on_fit_config_fn=xgboost_round_config,
        global_output_config=global_output_config,
    )


def create_strategy(strategy_params) -> fl.server.strategy.FedAvg:
    """Create the Flower server strategy for the configured model type."""
    num_classes = strategy_params.get("num_classes", 1)
    problem_type = strategy_params.get("problem_type", "regression")

    if strategy_params.get("model_type") == "xgboost":
        return create_xgboost_strategy(strategy_params)
    num_data_features, test_loader = build_test_data(strategy_params)

    model_type = strategy_params.get("model_type")
    model_class = get_torch_model_class(model_type)

    initial_model = model_class(model_id="initial", output_dir=strategy_params["output_dir"], problem_type=strategy_params["problem_type"])
    output_dim = num_classes if problem_type == "classification" else 1
    initial_model.build_model(num_data_features, output_dim=output_dim)
    params = initial_model.get_parameters(None)
    accuracy_tolerance = strategy_params.get("accuracy_tolerance")
    if accuracy_tolerance is None:
        accuracy_tolerance = 0.1

    strategy = fl.server.strategy.FedAvg(
    initial_parameters=ndarrays_to_parameters(params),
    evaluate_fn=get_evaluate_fn(
        num_data_features,
        strategy_params['num_rounds'],
        test_loader,
        strategy_params['output_dir'],
        model_type,
        problem_type,
        strategy_params.get("class_labels"),
        num_classes,
        accuracy_tolerance,
    ),
    evaluate_metrics_aggregation_fn=weighted_average_metrics,
    min_fit_clients=strategy_params['num_clients'],
    min_evaluate_clients=strategy_params['num_clients'],
    min_available_clients=strategy_params['num_clients'],
    on_fit_config_fn=fit_round,
    on_evaluate_config_fn=fit_round,
    )
    return strategy
