from typing import Dict, Any
from collections import OrderedDict
from pathlib import Path

import json
from datetime import datetime

import flwr as fl
from flwr.common import Context
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine

from .dataset import (
    load_pickle_data,
    load_random_partitions,
    load_custom_partitions
)

from .model import Net, eval_cnn, save_cnn, train_cnn
from .utils import get_device


# warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = get_device()
print('USING DEVICE: ', DEVICE)


def create_dataloaders(
    client_id: int,
    data_partitions_file,
    num_partitions,
    partitioner_type,
    test_fraction,
    seed,
    batch_divisor,
):
    vcf, pheno, bin_ranges = load_pickle_data()
    num_data_features = vcf.shape[1]
    combined_dataset = np.concatenate((vcf, pheno), axis=1)
    batch_size = max(1, vcf.shape[0] // batch_divisor)
    print('BATCH DIVISOR: ', batch_divisor, 'BATCH SIZE: ', batch_size)
    # print_bin_ranges(combined_dataset, bin_ranges)

    # Check if data partitions file is provided and exists
    data_partitions = None
    if data_partitions_file and Path(data_partitions_file).exists():
        data_partitions = np.load(data_partitions_file)
        print(f"Loaded data partitions from {data_partitions_file}")
    else:
        print(
            f"Data partitions file not found at {data_partitions_file}. "
            f"Using a random partition for client {client_id}"
        )

    if data_partitions is not None:
        # if data partitions available, train a model for each data partition
        train_loader, test_loader, train_indices, test_indices = (
            load_custom_partitions(
                client_id,
                combined_dataset,
                data_partitions,
                batch_size,
                test_fraction,
                seed,
            )
        )
    else:
        # Partition data into n_models randomly to train N client models
        train_loader, test_loader, train_indices, test_indices = (
            load_random_partitions(
                client_id,
                combined_dataset,
                batch_size,
                test_fraction,
                seed,
                num_partitions,
                partitioner_type,
            )
        )

    partitions_path = (
        Path(data_partitions_file).name if data_partitions else 'none'
    )
    return (
        num_data_features,
        partitions_path,
        train_loader,
        test_loader,
        train_indices,
        test_indices,
    )


def init_model(
    optimizer_name,
    learning_rate,
    weight_decay,
    num_data_features,
    opacus_params,
):
    model = Net(num_data_features).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "adamax":
        optimizer = optim.Adamax(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")
    privacy_engine = PrivacyEngine(
        accountant='rdp',
        secure_mode=opacus_params.get('secure_mode', False),
    )
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        epochs=opacus_params.get('epochs', 1),
        target_epsilon=opacus_params.get('epsilon', 1.0),
        target_delta=opacus_params.get('delta', 1e-5),
        module=model,
        optimizer=optimizer,
        data_loader=opacus_params.get('train_loader', None),
        max_grad_norm=opacus_params.get('max_grad_norm', 1.0),
    )
    privacy_engine.accountant.alphas = [1 + x / 10.0 for x in range(1000)]

    return model, criterion, optimizer, privacy_engine


def save_client(
    output_dir: str,
    federated_round: int,
    client_id: int,
    model,
    partitions_path: str,
    train_indices: list,
    test_indices: list,
    train_metrics: dict,
    test_metrics: dict,
    per_epoch_metrics: dict,
    predictions: dict,
    hyperparams: dict,
):
    metadata = {
        "created on": str(datetime.now()),
        "model id": client_id,
        "partitions file": partitions_path,
        "federated round": federated_round,
        "train accuracy": float(train_metrics["accuracy"]),
        "test accuracy": float(test_metrics["accuracy"]),
        "train loss": float(train_metrics["loss"]),
        "test loss": float(test_metrics["loss"]),
        "train indices": train_indices,
        "test indices": test_indices,
        "train accuracy per epoch": np.array(per_epoch_metrics["train_acc"]),
        "test accuracy per epoch": np.array(per_epoch_metrics["test_acc"]),
        "losses per epoch": np.array(per_epoch_metrics["losses"]),
        "epsilon per epoch": np.array(per_epoch_metrics.get("eps_spent", [])),
        "predictions": np.array(predictions["predictions"]),
        "hyperparameters": json.dumps(hyperparams),
    }

    eps_str = str(hyperparams['epsilon'])
    eps_str = eps_str[:-2] if eps_str.endswith('.0') else eps_str
    filename = f'dpcnn{eps_str}_opacus_oil_{client_id}'
    save_cnn(model, metadata, filename, federated_round, output_dir)


def train_partition(
    client_id,
    train_loader,
    test_loader,
    epochs,
    model,
    optimizer,
    criterion,
    delta,
    privacy_engine,
):
    model.to(DEVICE)
    (
        train_acc_epochs,
        test_acc_epochs,
        losses_epochs,
        epsilon_spent_epochs,
    ) = train_cnn(
        client_id,
        train_loader,
        test_loader,
        epochs,
        model,
        optimizer,
        criterion,
        delta,
        privacy_engine,
        DEVICE,
    )

    return (
        train_acc_epochs,
        test_acc_epochs,
        losses_epochs,
        epsilon_spent_epochs,
    )


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self, context: Context, client_id: int, params: Dict[str, Any]
    ):
        self.seed = params.get('seed', 42)
        torch.manual_seed(self.seed)

        self.current_round = 0
        self.client_state = context.state
        self.client_id = client_id
        self.partitions_type = params.get('partitions_type', 'uniform')
        self.num_partitions = params.get('num_partitions', 4)
        self.partition_id = params.get('partition_id', 0)
        self.batch_divisor = params.get('batch_divisor', 40)
        self.learning_rate = params.get('learning_rate', 0.003)
        self.weight_decay = params.get('weight_decay', 0.0001)
        self.test_fraction = params.get('test_fraction', 0.2)
        self.epochs = params.get('epochs', 10)
        self.data_partitions_file = params.get('data_partitions_file', None)
        self.optimizer_name = params.get('optimizer_name', 'adamax')
        self.epsilon = params.get('epsilon', 1.0)
        self.delta = params.get('delta', 1e-5)
        self.max_grad_norm = params.get('max_grad_norm', 1.0)
        self.opacus_secure_mode = params.get('opacus_secure_mode', False)
        self.output_dir = params.get('output_dir', None)

        (
            self.num_data_features,
            self.partitions_file,
            self.train_loader,
            self.test_loader,
            self.train_indices,
            self.test_indices,
        ) = create_dataloaders(
            self.client_id,
            self.data_partitions_file,
            self.num_partitions,
            self.partitions_type,
            self.test_fraction,
            self.seed,
            self.batch_divisor,
        )

        self.opacus_params = {
            'epochs': self.epochs,
            'epsilon': self.epsilon,
            'delta': self.delta,
            'max_grad_norm': self.max_grad_norm,
            'secure_mode': self.opacus_secure_mode,
            'train_loader': self.train_loader,
        }
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.privacy_engine = None

    def get_parameters(self, config):
        self.model.eval()
        return [
            val.detach().cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "batch_norm" not in name
        ]

    def set_parameters(self, parameters):
        keys = [
            k for k in self.model.state_dict().keys() if "batch_norm" not in k
        ]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v).float().to(DEVICE) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, False)

    def fit(self, parameters, config):
        self.model, self.criterion, self.optimizer, self.privacy_engine = (
            init_model(
                self.optimizer_name,
                self.learning_rate,
                self.weight_decay,
                self.num_data_features,
                self.opacus_params,
            )
        )
        self.current_round = config.get('server_round', None) - 1
        if self.current_round > 0:
            self.set_parameters(parameters)
        (
            train_acc_epochs,
            test_acc_epochs,
            losses_epochs,
            epsilon_spent_epochs,
        ) = train_partition(
            self.client_id,
            self.train_loader,
            self.test_loader,
            self.epochs,
            self.model,
            self.optimizer,
            self.criterion,
            self.delta,
            self.privacy_engine,
        )
        (train_acc, test_acc, train_loss, test_loss, predictions) = eval_cnn(
            self.client_id,
            self.train_loader,
            self.test_loader,
            self.model,
            self.criterion,
        )

        save_client(
            output_dir=self.output_dir,
            federated_round=self.current_round,
            client_id=self.client_id,
            model=self.model,
            partitions_path=self.partitions_file,
            train_indices=self.train_indices,
            test_indices=self.test_indices,
            train_metrics={
                "accuracy": train_acc,
                "loss": train_loss,
            },
            test_metrics={
                "accuracy": test_acc,
                "loss": test_loss,
            },
            per_epoch_metrics={
                "train_acc": train_acc_epochs,
                "test_acc": test_acc_epochs,
                "losses": losses_epochs,
                "eps_spent": epsilon_spent_epochs,  # Optional
            },
            predictions={"predictions": predictions},
            hyperparams={
                "learning rate": self.learning_rate,
                "weight decay": self.weight_decay,
                "batch divisor": self.batch_divisor,
                "epochs": self.epochs,
                "seed": self.seed,
                "test fraction": self.test_fraction,
                'optimizer': self.optimizer_name,
                'epsilon': self.epsilon,
                'delta': self.delta,
                'max grad norm': self.max_grad_norm,
            },
        )

        return (
            self.get_parameters(config=config),
            len(self.train_loader.dataset),
            {},
        )

    def evaluate(self, parameters, config):
        self.model, self.criterion, self.optimizer, self.privacy_engine = (
            init_model(
                self.optimizer_name,
                self.learning_rate,
                self.weight_decay,
                self.num_data_features,
                self.opacus_params,
            )
        )
        self.set_parameters(parameters)
        self.current_round = config.get('server_round', None) - 1
        print(
            f"Model {self.client_id} "
            f"| Evaluating model | Round {self.current_round}"
        )
        (train_acc, test_acc, train_loss, test_loss, predictions) = eval_cnn(
            self.client_id,
            self.train_loader,
            self.test_loader,
            self.model,
            self.criterion,
        )

        # convert loss to float to avoid Flower-Numpy type error
        return (
            float(test_loss),
            len(self.test_loader.dataset),
            {"accuracy": test_acc},
        )
