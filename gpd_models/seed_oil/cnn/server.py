from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

from utils import server_args_parser


# Parse arguments for client parameters
args = server_args_parser()
print(f"server args: {args}")
num_rounds = args.num_rounds
min_fit_clients = args.min_fit_clients
min_evaluate_clients = args.min_evaluate_clients
min_available_clients = args.min_available_clients


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


# Define strategy
strategy = fl.server.strategy.FedAvg(
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
