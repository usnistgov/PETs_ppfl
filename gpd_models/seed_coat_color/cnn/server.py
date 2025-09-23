from typing import List, Tuple
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
from utils import server_args_parser


# Parse arguments for client parameters
args = server_args_parser()
print(f"server args: {args}")
num_rounds = args.num_rounds
min_fit_clients = args.min_fit_clients
min_evaluate_clients = args.min_evaluate_clients
min_available_clients = args.min_available_clients


ohe, vcf, pheno = load_pickle_data()
combined_dataset = np.concatenate((vcf, pheno), axis=1)

test_loader = DataLoader(combined_dataset, batch_size=64, shuffle=False)


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


def get_evaluate_fn(test_loader: DataLoader):
    """Return a function that can be called to do global evaluation."""

    def evaluate_fn(server_round: int, parameters, config):
        """Evaluate global model on the whole test set."""
        if server_round == 0:
            return 0.0, {"accuracy": 0.0}
        model = Net(vcf.shape[1])
        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        loss, accuracy = eval_model(model, test_loader)
        print('GLOBAL ACCURACY:', accuracy)
        if server_round == num_rounds:
            torch.save(model.state_dict(), "cnn_global.torch")
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

params = get_parameters(Net(vcf.shape[1]))

# Define strategy
strategy = fl.server.strategy.FedAvg(
    initial_parameters=ndarrays_to_parameters(params),
    evaluate_fn=get_evaluate_fn(test_loader),
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=min_fit_clients,
    min_evaluate_clients=min_evaluate_clients,
    min_available_clients=min_available_clients,
    on_fit_config_fn=fit_round,
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)
