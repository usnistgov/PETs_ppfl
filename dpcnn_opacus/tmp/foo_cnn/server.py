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
from tmp.cnn.model import Net, unpack_batch
from report import Report

loss_rounds = []  # loss per global round
accuracy_rounds = []  # accuracy per global round
mse_rounds = []  # mse per global round

def eval_model(model, test_loader):
    correct = 0
    total = 0
    test_loss = 0
    total_mse = 0
    pred_correct_test = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = unpack_batch(data)
            # data should have at least 2 samples, otherwise
            # it will fail at batch normalization layer
            if inputs.shape[0] < 2:
                continue
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            pred_classes = torch.round(outputs)
            pred_correct = pred_classes == labels
            pred_correct_test += pred_correct.sum().item()
            total += labels.size(0)
            total_mse += torch.sum((outputs - labels.float()) ** 2).item()

    test_loss = test_loss / len(test_loader)
    test_accuracy = pred_correct_test / total
    test_mse = total_mse / total

    return test_loss, test_mse, test_accuracy


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
            mse_rounds.append(0.0)
            return 0.0, {"accuracy": 0.0, "mse": 0.0}
        model = Net(num_data_features)
        # set parameters to the model
        keys = [k for k in model.state_dict().keys() if "batch_norm" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        loss, mse, accuracy = eval_model(model, test_loader)
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
                Path(output_dir, "cnn_global_metadata.npz"),
                **metadata
            )
            report = Report(metadata)
            report.save_to_file(Path(output_dir, "cnn_global_metadata.json"))
            torch.save(
                model.state_dict(),
                Path(output_dir, "cnn_global.torch"),
            )
        return loss, {"accuracy": accuracy, "mse": mse}

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
    tt_vcf, tt_pheno, ho_vcf, ho_pheno = load_npy_feature_label_data(strategy_params['data_dir'])
    num_data_features = tt_vcf.shape[1]
    all_indices = np.arange(len(tt_vcf) + len(ho_vcf))
    test_dataset = IndexedArrayDataset(tt_vcf, tt_pheno, ho_vcf, ho_pheno, all_indices)
    total_rows = len(tt_vcf) + len(ho_vcf)
    batch_size = max(1, total_rows // strategy_params['batch_divisor'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
