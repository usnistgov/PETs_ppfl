import warnings

import flwr as fl
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic

from server_utils import (
    CyclicClientManager,
    eval_config,
    evaluate_metrics_aggregation,
    fit_config,
)

warnings.filterwarnings("ignore", category=UserWarning)


def create_strategy(strategy_params):
    train_method = strategy_params.get("train_method", "bagging")
    pool_size = strategy_params["min_available_clients"]
    min_fit_clients = strategy_params["min_fit_clients"]
    min_evaluate_clients = strategy_params["min_evaluate_clients"]
    centralised_eval = strategy_params.get("centralised_eval", False)

    if train_method == "bagging":
        return FedXgbBagging(
            evaluate_function=None,
            fraction_fit=(float(min_fit_clients) / pool_size),
            min_fit_clients=min_fit_clients,
            min_available_clients=pool_size,
            min_evaluate_clients=(
                min_evaluate_clients if not centralised_eval else 0
            ),
            fraction_evaluate=1.0 if not centralised_eval else 0.0,
            on_evaluate_config_fn=eval_config,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=(
                evaluate_metrics_aggregation if not centralised_eval else None
            ),
        )

    return FedXgbCyclic(
        fraction_fit=1.0,
        min_available_clients=pool_size,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=eval_config,
        on_fit_config_fn=fit_config,
    )