import sys
from typing import Any, Dict
from collections import OrderedDict
from pathlib import Path
import re
import json
from datetime import datetime
from collections import Counter

import flwr as fl
from flwr.common import Context
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import (
    instantiate_partitioner,
    train_test_partition_split,
    load_pickle_data,
    train_test_indices_split,
)
from model import Net, eval_cnn, save_cnn, train_cnn
from utils import get_device


# warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = get_device()
print('DEVICE: ', DEVICE)


def create_dataloaders(
    client_id,
    data_partitions_file,
    data_directory,
    num_partitions,
    partitioner_type,
    test_fraction,
    seed,
    batch_divisor,
):
    _, tt_vcf, tt_pheno = load_pickle_data(data_directory)
    combined_dataset = np.concatenate((tt_vcf, tt_pheno), axis=1)
    batch_size = max(1, tt_vcf.shape[0] // batch_divisor)

    data_partitions = None
    if data_partitions_file and Path(data_partitions_file).exists():
        data_partitions = np.load(data_partitions_file)
        print(f"Loaded data partitions from {data_partitions_file}")

    if data_partitions is not None:
        data_partition_ids = sorted(
            [k for k in data_partitions.keys() if re.match(r'client_\d+', k)]
        )
        numeric_id = [int(ci.split('_')[-1]) for ci in data_partition_ids]
        if client_id not in numeric_id:
            print(
                f"Cannot use client {client_id} for training because it was "
                f"not found in the data partitions."
            )
            sys.exit(1)
        partition_indices = data_partitions[data_partition_ids[client_id]]
        train_indices, test_indices = train_test_indices_split(
            combined_dataset, partition_indices, test_fraction, seed
        )
    else:
        partitioner = instantiate_partitioner(
            partitioner_type, num_partitions, data_directory
        )
        partition = partitioner.load_partition(client_id)
        train_indices, test_indices, _, _ = train_test_partition_split(
            partition, test_fraction=test_fraction, seed=seed
        )
        train_indices = train_indices.to_pandas().to_numpy().flatten()
        test_indices = test_indices.to_pandas().to_numpy().flatten()

    train_data = combined_dataset[train_indices]
    test_data = combined_dataset[test_indices]
    print(
        f"Train class counts: {Counter(train_data[:, -1])}, "
        f"Test class counts: {Counter(test_data[:, -1])}"
    )

    return (
        tt_vcf.shape[1],
        Path(data_partitions_file).name if data_partitions is not None else 'none',
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size, shuffle=False),
        train_indices,
        test_indices,
    )


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self, context: Context, client_id: int, params: Dict[str, Any]
    ):
        self.current_round = 0
        self.client_state = context.state
        self.client_id = client_id
        self.num_partitions = params.get('num_partitions')
        self.partitioner_type = params.get('partitioner_type')
        self.batch_divisor = params.get('batch_division')
        self.learning_rate = params.get('learning_rate')
        self.weight_decay = params.get('weight_decay')
        self.test_fraction = params.get('test_fraction')
        self.epochs = params.get('epochs')
        self.data_partitions_file = params.get('data_partitions_file')
        self.optimizer_name = params.get('optimizer_name')
        self.output_dir = params.get('output_dir')
        self.data_dir = params.get('data_dir')
        self.seed = params.get('seed')
        torch.manual_seed(self.seed)

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
            self.data_dir,
            self.num_partitions,
            self.partitioner_type,
            self.test_fraction,
            self.seed,
            self.batch_divisor,
        )
        self.model = self._init_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._init_optimizer()
        self.train_acc_per_epoch = []
        self.test_acc_per_epoch = []
        self.losses = []

    def _init_model(self):
        return Net(self.num_data_features).to(DEVICE)

    def _init_optimizer(self):
        if self.optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        return optim.Adamax(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def get_parameters(self, config):
        self.model.eval()
        return [
            val.detach().cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):
        state_dict = OrderedDict()
        for (key, ref_tensor), value in zip(self.model.state_dict().items(), parameters):
            state_dict[key] = torch.tensor(
                value,
                dtype=ref_tensor.dtype,
                device=ref_tensor.device,
            )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.current_round = config.get('server_round', 1) - 1
        self.train_acc_per_epoch, self.test_acc_per_epoch, self.losses = (
            train_cnn(
                self.train_loader,
                self.test_loader,
                self.epochs,
                self.model,
                self.optimizer,
                self.criterion,
                DEVICE,
            )
        )
        return (
            self.get_parameters(config=config),
            len(self.train_loader.dataset),
            {},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        train_acc, test_acc, train_loss, test_loss, predicted_labels = (
            eval_cnn(
                self.train_loader, self.test_loader, self.model, self.criterion
            )
        )
        print(f"Client {self.client_id} predicted label counts: {predicted_labels}") # DEBUG
        print(f"Client {self.client_id} test accuracy: {test_acc}") # DEBUG
        metadata = {
            'created on': str(datetime.now()),
            'model id': int(self.client_id),
            'partitions file': self.partitions_file,
            'train accuracy': float(train_acc),
            'test accuracy': float(test_acc),
            'train loss': float(train_loss),
            'test loss': float(test_loss),
            'train indices': self.train_indices,
            'test indices': self.test_indices,
            "train accuracy per epoch": np.array(self.train_acc_per_epoch),
            "test accuracy per epoch": np.array(self.test_acc_per_epoch),
            "losses per epoch": np.array(self.losses),
            "predicted labels": json.dumps(predicted_labels),
            'hyperparameters': json.dumps({
                'learning rate': float(self.learning_rate),
                'weight decay': float(self.weight_decay),
                'batch divisor': int(self.batch_divisor),
                'epochs': int(self.epochs),
                'seed': int(self.seed),
                'test fraction': float(self.test_fraction),
            }),
        }
        save_cnn(
            self.model,
            metadata,
            f'flcnn_{self.client_id}',
            self.current_round,
            self.output_dir,
        )
        return (
            float(test_loss),
            len(self.test_loader.dataset),
            {"accuracy": test_acc},
        )
