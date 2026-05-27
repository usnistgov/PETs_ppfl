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
    load_random_partitions,
    load_custom_partitions,
    load_npy_feature_label_data,
)

from model import Net, eval_cnn, save_cnn, train_cnn
from utils import get_device


# warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = get_device()
print('DEVICE: ', DEVICE)


def create_dataloaders(
    client_id: int,
    data_partitions_file,
    data_directory,
    num_partitions,
    partitioner_type,
    test_fraction,
    seed,
    batch_divisor,
):
    tt_vcf, tt_pheno, ho_vcf, ho_pheno = load_npy_feature_label_data(data_directory)
    num_data_features = tt_vcf.shape[1]
    total_rows = len(tt_vcf) + len(ho_vcf)
    batch_size = max(1, total_rows // batch_divisor)

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
                tt_vcf,
                tt_pheno,
                ho_vcf,
                ho_pheno,
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
                tt_vcf,
                tt_pheno,
                ho_vcf,
                ho_pheno,
                batch_size,
                test_fraction,
                seed,
                num_partitions,
                partitioner_type,
                data_directory,
            )
        )

    partitions_path = (
        Path(data_partitions_file).name if data_partitions is not None else 'none'
    )
    return (
        num_data_features,
        partitions_path,
        train_loader,
        test_loader,
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
        self.partitioner_type = params.get('partitions_type')
        self.partition_id = params.get('partition_id')
        self.batch_divisor = params.get('batch_divisor')
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
        self.criterion = nn.MSELoss()
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
        _, _, self.train_acc_per_epoch, self.test_acc_per_epoch, self.losses = (
            train_cnn(
                self.client_id,
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
        (
            train_acc,
            test_acc,
            train_loss,
            test_loss,
            train_mse,
            test_mse,
            train_preds,
            test_preds,
        ) = (
            eval_cnn(
                self.client_id, self.train_loader, self.test_loader, self.model, self.criterion
            )
        )
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
            "predicted labels": test_preds,
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
