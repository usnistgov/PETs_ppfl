from typing import List, Tuple
from typing import Dict
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters
from pathlib import Path
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic
import xgboost as xgb
from xgboost.core import DMatrix
from sklearn.metrics import accuracy_score, mean_squared_error

from dataset import IndexedArrayDataset, load_npy_feature_label_data
from model import BaseModel, CNNModel, DPCNNModel
from report import Report
from utils import get_device

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
    problem_type: str = "regression",
    class_labels=None,
    num_classes: int = 1,
    accuracy_tolerance: float = 0.1,
):
    """Return a function that can be called to do global evaluation."""

    def evaluate_fn(server_round: int, parameters, config):
        """Evaluate global model on the whole test set."""
        if server_round == 0:
            loss_rounds.append(0.0)
            accuracy_rounds.append(0.0)
            mse_rounds.append(0.0)
            if problem_type == "classification":
                precision_rounds.append(0.0)
                recall_rounds.append(0.0)
            return 0.0, {"accuracy": 0.0, "mse": 0.0}

        model_class = DPCNNModel if model_type == "dpcnn" else CNNModel
        global_model = model_class(model_id="global", output_dir=output_dir)
        output_dim = num_classes if problem_type == "classification" else 1
        global_model.build_model(num_data_features, output_dim=output_dim)
        global_model.set_parameters(parameters)

        criterion = (
            nn.CrossEntropyLoss()
            if problem_type == "classification"
            else nn.MSELoss()
        )

        (
            test_mse,
            test_accuracy,
            test_loss,
            _test_class_metrics,
        ) = global_model.compute_test_metrics(
            test_loader,
            criterion,
            get_device(),
            problem_type=problem_type,
            accuracy_tolerance=accuracy_tolerance,
        )

        loss_rounds.append(float(test_loss))
        accuracy_rounds.append(float(test_accuracy))
        mse_rounds.append(float(test_mse))
        if problem_type == "classification":
            precision_rounds.append(float(_test_class_metrics["precision_macro"]))
            recall_rounds.append(float(_test_class_metrics["recall_macro"]))

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
            metadata = {
                "created on": str(datetime.now()),
            }
            BaseModel.add_task_report_metadata(
                metadata,
                problem_type,
                class_labels,
                accuracy_tolerance,
            )
            metadata.update({
                "loss per round": np.array(loss_rounds),
                "accuracy per round": np.array(accuracy_rounds),
            })
            if problem_type == "classification":
                BaseModel.add_global_classification_report_metrics(
                    metadata,
                    precision_rounds,
                    recall_rounds,
                )
            else:
                metadata["mse per round"] = np.array(mse_rounds)

            np.savez(
                Path(output_dir, f"{prefix}_global_metadata.npz"),
                **metadata,
            )
            report = Report(metadata)
            report.save_to_file(Path(output_dir, f"{prefix}_global_metadata.json"))

            torch.save(
                global_model.model.state_dict(),
                Path(output_dir, f"{prefix}_global.torch"),
            )

        return float(test_loss), {
            "accuracy": float(test_accuracy),
            "mse": float(test_mse),
        }

    return evaluate_fn


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    # Multiply accuracy of each client by number of examples used
    accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
    mse = sum(num_examples * m["mse"] for num_examples, m in metrics) / total_examples
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": accuracy, "mse": mse,}


def fit_round(server_round: int):
    """Configure the fit function for each round."""
    return {"server_round": server_round}


def xgboost_round_config(server_round: int) -> Dict[str, str]:
    return {"global_round": str(server_round)}


def evaluate_and_save_xgboost_global(
    server_round: int,
    parameters,
    num_rounds: int,
    test_data: DMatrix,
    output_dir: str,
    xgboost_params: Dict,
    problem_type: str = "regression",
    class_labels=None,
    accuracy_tolerance: float = 0.1,
):
    """Evaluate and save the aggregated XGBoost global model."""
    if parameters is None or not parameters.tensors:
        return 0.0, {"accuracy": 0.0, "mse": 0.0}

    bst = xgb.Booster(params=xgboost_params or {})
    bst.load_model(bytearray(parameters.tensors[-1]))

    labels = test_data.get_label()
    predictions = bst.predict(test_data)
    if problem_type == "classification":
        if predictions.ndim == 2:
            predicted_labels = np.argmax(predictions, axis=1)
        else:
            predicted_labels = np.rint(predictions)
        accuracy = accuracy_score(labels, predicted_labels)
        mse = 0.0
        classification_metrics = BaseModel.get_classification_metrics(labels, predicted_labels)
    else:
        rounded_predictions = np.rint(predictions)
        accuracy = accuracy_score(labels, rounded_predictions)
        mse = mean_squared_error(labels, predictions)

    loss_rounds.append(float(mse))
    accuracy_rounds.append(float(accuracy))
    mse_rounds.append(float(mse))
    if problem_type == "classification":
        precision_rounds.append(float(classification_metrics["precision_macro"]))
        recall_rounds.append(float(classification_metrics["recall_macro"]))

    print(
        "GLOBAL ACCURACY:",
        accuracy,
        "GLOBAL MSE:",
        mse,
        "SERVER ROUND:",
        server_round,
    )

    if server_round == num_rounds:
        metadata = {
            "created on": str(datetime.now()),
        }
        BaseModel.add_task_report_metadata(
            metadata,
            problem_type,
            class_labels,
            accuracy_tolerance,
        )
        metadata.update({
            "loss per round": np.array(loss_rounds),
            "accuracy per round": np.array(accuracy_rounds),
        })
        if problem_type == "classification":
            BaseModel.add_global_classification_report_metrics(
                metadata,
                precision_rounds,
                recall_rounds,
            )
        else:
            metadata["mse per round"] = np.array(mse_rounds)

        np.savez(
            Path(output_dir, "xgb_global_metadata.npz"),
            **metadata,
        )
        report = Report(metadata)
        report.save_to_file(Path(output_dir, "xgb_global_metadata.json"))
        bst.save_model(Path(output_dir, "xgb_global.ubj"))

    return float(mse), {
        "accuracy": float(accuracy),
        "mse": float(mse),
    }


def get_xgboost_evaluate_fn(global_output_config):
    """Return a CNN/DPCNN-style server evaluate function for XGBoost bagging."""

    def evaluate_fn(server_round, parameters, config):
        return evaluate_and_save_xgboost_global(
            server_round,
            parameters,
            **global_output_config,
        )

    return evaluate_fn


class GlobalOutputFedXgbCyclic(FedXgbCyclic):
    def __init__(self, *args, global_output_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_output_config = global_output_config or {}

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        evaluate_and_save_xgboost_global(
            server_round,
            parameters,
            **self.global_output_config,
        )
        return parameters, metrics


def evaluate_xgboost_metrics(eval_metrics):
    """Aggregate XGBoost client metrics using the CNN/DPCNN metric shape."""
    eval_metrics = [
        (num, dict(metrics))
        for num, metrics in eval_metrics
        if num > 0 and len(metrics) > 0
    ]
    total_num = sum([num for num, _ in eval_metrics])
    if total_num == 0:
        return {}

    metric_names = sorted(
        {
            metric_name
            for _, metrics in eval_metrics
            for metric_name in metrics.keys()
        }
    )
    return {
        metric_name: (
            sum(
                metrics.get(metric_name, 0.0) * num
                for num, metrics in eval_metrics
            )
            / total_num
        )
        for metric_name in metric_names
    }


def create_xgboost_strategy(strategy_params):
    train_method = strategy_params.get("train_method")
    pool_size = strategy_params["num_clients"]
    min_fit_clients = strategy_params["num_clients"]
    min_evaluate_clients = strategy_params["num_clients"]
    centralised_eval = strategy_params.get("centralised_eval")
    tt_vcf, tt_pheno, ho_vcf, ho_pheno = load_npy_feature_label_data(
        strategy_params["data_dir"]
    )
    test_features = np.concatenate([tt_vcf, ho_vcf])
    test_labels = np.concatenate([tt_pheno, ho_pheno]).reshape(-1)
    if strategy_params.get("problem_type") == "classification":
        label_to_index = {
            label: i for i, label in enumerate(strategy_params.get("class_labels"))
        }
        test_labels = np.array([label_to_index[label] for label in test_labels])
    test_data = DMatrix(data=test_features, label=test_labels)
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
                min_evaluate_clients if not centralised_eval else 0
            ),
            fraction_evaluate=1.0 if not centralised_eval else 0.0,
            on_evaluate_config_fn=xgboost_round_config,
            on_fit_config_fn=xgboost_round_config,
            evaluate_metrics_aggregation_fn=(
                evaluate_xgboost_metrics if not centralised_eval else None
            ),
        )

    return GlobalOutputFedXgbCyclic(
        fraction_fit=1.0,
        min_available_clients=pool_size,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=evaluate_xgboost_metrics,
        on_evaluate_config_fn=xgboost_round_config,
        on_fit_config_fn=xgboost_round_config,
        global_output_config=global_output_config,
    )


def create_strategy(strategy_params) -> fl.server.strategy.FedAvg:
    if strategy_params.get("model_type") == "xgboost":
        return create_xgboost_strategy(strategy_params)

    tt_vcf, tt_pheno, ho_vcf, ho_pheno = load_npy_feature_label_data(strategy_params['data_dir'])
    num_data_features = tt_vcf.shape[1]
    all_indices = np.arange(len(tt_vcf) + len(ho_vcf))
    problem_type = strategy_params.get("problem_type", "regression")
    num_classes = strategy_params.get("num_classes", 1)
    label_to_index = (
        {label: i for i, label in enumerate(strategy_params.get("class_labels"))}
        if problem_type == "classification"
        else None
    )
    test_dataset = IndexedArrayDataset(
        tt_vcf,
        tt_pheno,
        ho_vcf,
        ho_pheno,
        all_indices,
        label_to_index,
    )
    total_rows = len(tt_vcf) + len(ho_vcf)
    batch_size = max(1, total_rows // strategy_params['batch_divisor'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_type = strategy_params.get("model_type")
    model_class = DPCNNModel if model_type == "dpcnn" else CNNModel

    initial_model = model_class(model_id="initial", output_dir=strategy_params["output_dir"])
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
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=strategy_params['num_clients'],
    min_evaluate_clients=strategy_params['num_clients'],
    min_available_clients=strategy_params['num_clients'],
    on_fit_config_fn=fit_round,
    )
    return strategy
