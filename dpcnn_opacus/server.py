from typing import List, Tuple
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters

from dataset import load_pickle_data
from model import Net


tol_offset = 0.01  # Small constant to avoid zero tolerance

loss_rounds = []  # loss per global round
accuracy_rounds = []  # accuracy per global round
mse_rounds = []  # mse per global round


def eval_model(model, test_loader, accuracy_tolerance):
    correct = 0
    total = 0
    test_loss = 0
    total_mse = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for data in test_loader:
            # data should have at least 2 samples, otherwise
            # it will fail at batch normalization layer
            if data.shape[0] < 2:
                continue
            inputs = data[:, :-1]
            labels = data[:, -1]
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            total += labels.size(0)
            correct += torch.sum(
                torch.abs(outputs - labels.float())
                <= accuracy_tolerance * labels + tol_offset
            ).item()
            total_mse += torch.sum(torch.abs(outputs - labels)).item()

    test_loss = test_loss / len(test_loader)
    test_accuracy = correct / total
    test_mse = total_mse / total

    return test_loss, test_mse, test_accuracy


def get_evaluate_fn(
    num_data_features: int,
    num_rounds: int,
    test_loader: DataLoader,
    accuracy_tolerance: float,
    output_dir: str,
):
    """Return a function that can be called to do global evaluation."""

    def evaluate_fn(server_round: int, parameters, config):
        """Evaluate global model on the whole test set."""
        if server_round == 0:
            loss_rounds.append(0.0)
            accuracy_rounds.append(0.0)
            mse_rounds.append(0.0)
            return 0.0, {"accuracy": 0.0, "mse": 0.0}
        model = Net(num_data_features)
        # set parameters to the model
        keys = [k for k in model.state_dict().keys() if "batch_norm" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        loss, mse, accuracy = eval_model(
            model, test_loader, accuracy_tolerance
        )
        loss_rounds.append(loss)
        accuracy_rounds.append(accuracy)
        mse_rounds.append(mse)
        print(
            'GLOBAL ACCURACY:',
            accuracy,
            'GLOBAL MSE:',
            mse,
            'SERVER ROUND:',
            server_round,
        )
        if server_round == num_rounds:
            metadata = {
                "created on": str(datetime.now()),
                "loss per round": np.array(loss_rounds),
                "accuracy per round": np.array(accuracy_rounds),
                "mse per round": np.array(mse_rounds),
            }
            print('Loss per round:', loss_rounds)
            print('Accuracy per round:', accuracy_rounds)
            print('MSE per round:', mse_rounds)
            np.savez(
                Path(output_dir, "dpcnn_opacus_global_metadata.npz"),
                **metadata
            )
            torch.save(
                model.state_dict(),
                Path(output_dir, "dpcnn_opacus_global.torch"),
            )
        return loss, {"accuracy": accuracy, "mse": mse}

    return evaluate_fn


# Accuracy metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply MSE of each client by number of examples used
    mse_values = [num_examples * m["mse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Compute weighted average MSE
    return {"mse": sum(mse_values) / sum(examples)}


def fit_round(server_round: int):
    """Configure the fit function for each round."""
    return {"server_round": server_round}


def get_parameters(net) -> List[np.ndarray]:
    return [
        val.cpu().numpy()
        for name, val in net.state_dict().items()
        if "batch_norm" not in name
    ]


# Define strategy


def create_strategy(strategy_params) -> fl.server.strategy.FedAvg:
    ohe, vcf, pheno = load_pickle_data()
    combined_dataset = np.concatenate((vcf, pheno), axis=1)
    num_data_features = vcf.shape[1]
    test_loader = DataLoader(combined_dataset, batch_size=64, shuffle=False)

    params = get_parameters(Net(num_data_features))

    min_fit_clients = strategy_params['min_fit_clients']
    min_evaluate_clients = strategy_params['min_evaluate_clients']
    min_available_clients = strategy_params['min_available_clients']
    num_rounds = strategy_params['num_rounds']
    accuracy_tolerance = strategy_params['accuracy_tolerance']
    output_dir = strategy_params['output_dir']

    strategy = fl.server.strategy.FedAvg(
        initial_parameters=ndarrays_to_parameters(params),
        evaluate_fn=get_evaluate_fn(
            num_data_features,
            num_rounds,
            test_loader,
            accuracy_tolerance,
            output_dir,
        ),
        evaluate_metrics_aggregation_fn=weighted_average,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=fit_round,
        on_evaluate_config_fn=fit_round,
    )
    return strategy
