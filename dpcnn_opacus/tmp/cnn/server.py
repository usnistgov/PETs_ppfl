from typing import List, Tuple
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters
from pathlib import Path

from dataset import load_pickle_data
from model import Net

def eval_model(model, test_loader):
    correct = 0
    total = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in test_loader:
            # data should have at least 2 samples, otherwise
            # it will fail at batch normalization layer
            if data.shape[0] < 2:
                continue
            inputs = data[:, :-1]
            labels = data[:, -1]
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = test_loss / len(test_loader)
    test_accuracy = (correct / total) * 100
    return test_loss, test_accuracy


def get_evaluate_fn(num_data_features, num_rounds, test_loader, output_dir):
    """Return a function that can be called to do global evaluation."""

    def evaluate_fn(server_round: int, parameters, config):
        """Evaluate global model on the whole test set."""
        if server_round == 0:
            return 0.0, {"accuracy": 0.0}

        model = Net(num_data_features)

        state_dict = OrderedDict()
        for (key, ref_tensor), value in zip(model.state_dict().items(), parameters):
            state_dict[key] = torch.tensor(
                value,
                dtype=ref_tensor.dtype,
                device=ref_tensor.device,
            )

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        loss, accuracy = eval_model(model, test_loader)
        print("GLOBAL ACCURACY:", accuracy)

        if server_round == num_rounds:
            out_dir = Path(output_dir).absolute()
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(out_dir, "cnn_global.torch"))

        return loss, {"accuracy": accuracy}

    return evaluate_fn


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_round(server_round: int):
    """Configure the fit function for each round."""
    return {"server_round": server_round}

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def create_strategy(strategy_params) -> fl.server.strategy.FedAvg:
    _, vcf, pheno = load_pickle_data(strategy_params['data_dir'])
    combined_dataset = np.concatenate((vcf, pheno), axis=1)
    test_loader = DataLoader(combined_dataset, batch_size=64, shuffle=False)
    num_data_features = vcf.shape[1]
    params = get_parameters(Net(num_data_features))

    strategy = fl.server.strategy.FedAvg(
    initial_parameters=ndarrays_to_parameters(params),
    evaluate_fn=get_evaluate_fn(
        num_data_features,
        strategy_params['num_rounds'],
        test_loader,
        strategy_params['output_dir'],
    ),
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=strategy_params['min_fit_clients'],
    min_evaluate_clients=strategy_params['min_evaluate_clients'],
    min_available_clients=strategy_params['min_available_clients'],
    on_fit_config_fn=fit_round,
    )
    return strategy
