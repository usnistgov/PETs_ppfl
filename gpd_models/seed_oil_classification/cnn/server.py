from typing import List, Tuple
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters

from dataset import load_pickle_data, IndexedDataset
from model import Net, compute_accuracy


loss_rounds = []  # loss per global round
accuracy_rounds = []  # accuracy per global round
predictions_rounds = []


def get_evaluate_fn(
    num_data_features: int,
    num_rounds: int,
    test_loader: DataLoader,
    output_dir: str,
):
    """Return a function that can be called to do global evaluation."""

    def evaluate_fn(server_round: int, parameters, config):
        """Evaluate global model on the whole test set."""
        if server_round == 0:
            loss_rounds.append(0.0)
            accuracy_rounds.append(0.0)
            return 0.0, {"accuracy": 0.0}
        model = Net(num_data_features)
        # set parameters to the model
        keys = [k for k in model.state_dict().keys() if "batch_norm" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        accuracy, loss, predictions = compute_accuracy(
            model, test_loader, criterion
        )
        loss_rounds.append(loss)
        accuracy_rounds.append(accuracy)
        predictions_rounds.append(predictions)
        print(
            'GLOBAL ACCURACY:', accuracy * 100, 'SERVER ROUND:', server_round
        )
        outpath = Path(output_dir)
        if not outpath.exists():
            outpath.mkdir(parents=True)
        if server_round == num_rounds:
            metadata = {
                "created on": str(datetime.now()),
                "loss per round": np.array(loss_rounds),
                "accuracy per round": np.array(accuracy_rounds),
                "predictions per rounds": json.dumps(predictions_rounds),
            }
            np.savez(Path(outpath, "cnn_oil_global_metadata.npz"), **metadata)
            torch.save(
                model.state_dict(), Path(outpath, "cnn_oil_global.torch")
            )
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
    return [
        val.cpu().numpy()
        for name, val in net.state_dict().items()
        if "batch_norm" not in name
    ]


# Define strategy


def create_strategy(strategy_params) -> fl.server.strategy.FedAvg:
    vcf, pheno, bin_ranges = load_pickle_data()
    combined_dataset = np.concatenate((vcf, pheno), axis=1)
    num_data_features = vcf.shape[1]
    data_indices = np.arange(combined_dataset.shape[0])
    data_indexed = IndexedDataset(combined_dataset, data_indices)
    test_loader = DataLoader(data_indexed, batch_size=64, shuffle=False)

    params = get_parameters(Net(num_data_features))

    min_fit_clients = strategy_params['min_fit_clients']
    min_evaluate_clients = strategy_params['min_evaluate_clients']
    min_available_clients = strategy_params['min_available_clients']
    num_rounds = strategy_params['num_rounds']
    output_dir = strategy_params['output_dir']

    strategy = fl.server.strategy.FedAvg(
        initial_parameters=ndarrays_to_parameters(params),
        evaluate_fn=get_evaluate_fn(
            num_data_features, num_rounds, test_loader, output_dir
        ),
        evaluate_metrics_aggregation_fn=weighted_average,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=fit_round,
        on_evaluate_config_fn=fit_round,
    )
    return strategy
