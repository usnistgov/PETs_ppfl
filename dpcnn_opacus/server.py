from typing import List, Tuple
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

from dataset import IndexedArrayDataset, load_npy_feature_label_data
from model import CNNModel, DPCNNModel #Net, unpack_batch
from report import Report

loss_rounds = []  # loss per global round
accuracy_rounds = []  # accuracy per global round
mse_rounds = []  # mse per global round

def get_evaluate_fn(
    num_data_features: int,
    num_rounds: int,
    test_loader: DataLoader,
    output_dir: str,
    model_type: str,
):
    """Return a function that can be called to do global evaluation."""

    def evaluate_fn(server_round: int, parameters, config):
        """Evaluate global model on the whole test set."""
        if server_round == 0:
            loss_rounds.append(0.0)
            accuracy_rounds.append(0.0)
            mse_rounds.append(0.0)
            return 0.0, {"accuracy": 0.0, "mse": 0.0}

        model_class = DPCNNModel if model_type == "dpcnn" else CNNModel
        global_model = model_class(model_id="global", output_dir=output_dir)
        global_model.build_model(num_data_features)
        global_model.set_parameters(parameters)

        criterion = nn.MSELoss()

        (
            _train_accuracy,
            test_accuracy,
            _train_loss,
            test_loss,
            _train_mse,
            test_mse,
            _train_preds,
            _test_preds,
        ) = global_model.evaluate(
            test_loader,
            test_loader,
            criterion,
        )

        loss_rounds.append(float(test_loss))
        accuracy_rounds.append(float(test_accuracy))
        mse_rounds.append(float(test_mse))

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
                "loss per round": np.array(loss_rounds),
                "accuracy per round": np.array(accuracy_rounds),
                "mse per round": np.array(mse_rounds),
            }

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

def create_strategy(strategy_params) -> fl.server.strategy.FedAvg:
    tt_vcf, tt_pheno, ho_vcf, ho_pheno = load_npy_feature_label_data(strategy_params['data_dir'])
    num_data_features = tt_vcf.shape[1]
    all_indices = np.arange(len(tt_vcf) + len(ho_vcf))
    test_dataset = IndexedArrayDataset(tt_vcf, tt_pheno, ho_vcf, ho_pheno, all_indices)
    total_rows = len(tt_vcf) + len(ho_vcf)
    batch_size = max(1, total_rows // strategy_params['batch_divisor'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_type = strategy_params.get("model_type")
    model_class = DPCNNModel if model_type == "dpcnn" else CNNModel

    initial_model = model_class(model_id="initial", output_dir=strategy_params["output_dir"])
    initial_model.build_model(num_data_features)
    params = initial_model.get_parameters()

    strategy = fl.server.strategy.FedAvg(
    initial_parameters=ndarrays_to_parameters(params),
    evaluate_fn=get_evaluate_fn(
        num_data_features,
        strategy_params['num_rounds'],
        test_loader,
        strategy_params['output_dir'],
        model_type,
    ),
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=strategy_params['min_fit_clients'],
    min_evaluate_clients=strategy_params['min_evaluate_clients'],
    min_available_clients=strategy_params['min_available_clients'],
    on_fit_config_fn=fit_round,
    )
    return strategy
