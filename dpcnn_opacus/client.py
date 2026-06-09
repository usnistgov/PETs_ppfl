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
from xgboost.core import DMatrix

from dataset import (
    load_random_partitions,
    load_custom_partitions,
    load_npy_feature_label_data,
)

from model import CNNModel, DPCNNModel, XGBoostModel
from utils import get_device, configure_warning_logging
from opacus import PrivacyEngine
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from flwr.common.logger import log
from logging import INFO


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


def loader_to_dmatrix(loader):
    features = []
    labels = []
    for batch_features, batch_labels in loader:
        features.append(np.asarray(batch_features))
        labels.append(np.asarray(batch_labels).reshape(-1))
    return DMatrix(data=np.concatenate(features), label=np.concatenate(labels))


def create_xgboost_data(
    client_id: int,
    data_partitions_file,
    data_directory,
    num_partitions,
    partitioner_type,
    test_fraction,
    seed,
    batch_divisor,
):
    (
        _num_data_features,
        partitions_file,
        train_loader,
        test_loader,
        train_indices,
        test_indices,
    ) = create_dataloaders(
        client_id,
        data_partitions_file,
        data_directory,
        num_partitions,
        partitioner_type,
        test_fraction,
        seed,
        batch_divisor,
    )
    return (
        loader_to_dmatrix(train_loader),
        loader_to_dmatrix(test_loader),
        partitions_file,
        train_indices,
        test_indices,
    )


class TorchFlowerClient(fl.client.NumPyClient):
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
        self.print_warning_logs = params.get('print_warning_logs')
        torch.manual_seed(self.seed)
        if not self.print_warning_logs:
            configure_warning_logging(self.output_dir)

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

        self.model_type = params.get("model_type")
        self.use_dp = self.model_type == "dpcnn"
        self.epsilon = params.get("epsilon")
        self.delta = params.get("delta")
        self.max_grad_norm = params.get("max_grad_norm")
        self.opacus_secure_mode = params.get("opacus_secure_mode")
        self.eps_per_epoch = []

        model_class = DPCNNModel if self.use_dp else CNNModel

        self.cnn_model = model_class(model_id=self.client_id, output_dir=self.output_dir)
        self.cnn_model.build_model(self.num_data_features)
        self.cnn_model.model.to(DEVICE)

        self.criterion = nn.MSELoss()
        self.optimizer = self.cnn_model.build_optimizer(self.optimizer_name, self.learning_rate, self.weight_decay)

        self.privacy_engine = None

        if self.use_dp:
            opacus_params = {
                "epochs": self.epochs,
                "epsilon": self.epsilon,
                "delta": self.delta,
                "max_grad_norm": self.max_grad_norm,
                "secure_mode": self.opacus_secure_mode,
            }

            self.optimizer, self.train_loader, self.privacy_engine = (
                self.cnn_model.attach_privacy_engine(
                    self.optimizer,
                    self.train_loader,
                    opacus_params,
                )
            )

        self.train_acc_per_epoch = []
        self.test_acc_per_epoch = []
        self.losses = []

    def get_parameters(self, config):
        return self.cnn_model.get_parameters(config)

    def set_parameters(self, parameters):
        self.cnn_model.set_parameters(parameters)

    def fit(self, parameters, config):
        self.current_round = config.get("server_round", 1) - 1

        if not self.use_dp or self.current_round > 0:
            self.set_parameters(parameters)

        if self.use_dp:
            (
                self.train_mse_per_epoch,
                self.test_mse_per_epoch,
                self.train_acc_per_epoch,
                self.test_acc_per_epoch,
                self.losses,
                self.eps_per_epoch,
            ) = self.cnn_model.fit(
                self.train_loader,
                self.test_loader,
                self.epochs,
                self.optimizer,
                self.criterion,
                self.delta,
                self.privacy_engine,
                device=DEVICE,
            )
        else:
            (
                self.train_mse_per_epoch,
                self.test_mse_per_epoch,
                self.train_acc_per_epoch,
                self.test_acc_per_epoch,
                self.losses,
            ) = self.cnn_model.fit(
                self.train_loader,
                self.test_loader,
                self.epochs,
                self.optimizer,
                self.criterion,
                DEVICE)

        return self.get_parameters(config), len(self.train_loader.dataset), {}

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
        ) = self.cnn_model.evaluate(
            self.train_loader,
            self.test_loader,
            self.criterion,
        )
        hyperparameters = {
            'learning rate': float(self.learning_rate),
            'weight decay': float(self.weight_decay),
            'batch divisor': int(self.batch_divisor),
            'epochs': int(self.epochs),
            'seed': int(self.seed),
            'test fraction': float(self.test_fraction),
        }
        if self.use_dp:
            hyperparameters.update({
                "epsilon": float(self.epsilon),
                "delta": float(self.delta),
                "max grad norm": float(self.max_grad_norm),
                "opacus secure mode": bool(self.opacus_secure_mode),
            })

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
            'hyperparameters': json.dumps(hyperparameters),
        }
        if self.use_dp:
            metadata["epsilon per epoch"] = np.array(self.eps_per_epoch)
        model_prefix = "dpcnn" if self.use_dp else "cnn"
        name = f"{model_prefix}_client_{self.client_id}"
        self.cnn_model.save_model(
            metadata,
            name,
            self.current_round,
            self.output_dir,
        )
        return float(test_loss), len(self.test_loader.dataset), {"accuracy": float(test_acc), "mse": float(test_mse)}


class XGBoostFlowerClient(fl.client.Client):
    def __init__(self, context: Context, client_id: int, params: Dict[str, Any]):
        self.client_id = client_id
        self.params = params
        self.seed = params.get("seed")
        self.train_method = params.get("train_method")
        self.num_partitions = params.get("num_partitions")
        self.num_local_round = params.get("epochs")
        self.output_dir = params.get("output_dir")
        self.data_partitions_file = params.get("data_partitions_file")
        self.test_fraction = params.get("test_fraction")
        self.print_warning_logs = params.get('print_warning_logs')
        if not self.print_warning_logs:
            configure_warning_logging(self.output_dir)

        (
            self.train_data,
            self.test_data,
            self.partitions_file,
            self.train_indices,
            self.test_indices,
        ) = create_xgboost_data(
            self.client_id,
            self.data_partitions_file,
            params.get("data_dir"),
            self.num_partitions,
            params.get("partitions_type"),
            self.test_fraction,
            self.seed,
            params.get("batch_divisor"),
        )

        self.xgb_model = XGBoostModel(
            model_id=self.client_id,
            output_dir=self.output_dir,
            params=XGBoostModel.client_params(
                seed=self.seed,
                num_partitions=self.num_partitions,
                train_method=self.train_method,
                scaled_lr=params.get("scaled_lr"),
                base_params=params.get("xgboost_params"),
            ),
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        local_model_bytes = self.xgb_model.fit_round(
            self.train_data,
            self.test_data,
            self.num_local_round,
            self.train_method,
            global_round,
            ins.parameters.tensors,
        )
        train_acc, test_acc = self.xgb_model.save_client_output(
            self.train_data,
            self.test_data,
            self.train_indices,
            self.test_indices,
            self.partitions_file,
            self.seed,
            self.test_fraction,
            global_round,
            self.output_dir,
        )

        print(
            f"client {self.client_id} Train accuracy: {train_acc * 100:.2f}%",
            flush=True,
        )
        print(
            f"client {self.client_id} Test accuracy: {test_acc * 100:.2f}%",
            flush=True,
        )

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=len(self.test_indices),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if not ins.parameters.tensors:
            return EvaluateRes(
                status=Status(
                    code=Code.OK,
                    message="No model parameters available for evaluation.",
                ),
                loss=0.0,
                num_examples=0,
                metrics={"accuracy": 0.0, "mse": 0.0},
            )

        self.xgb_model.set_parameters(ins.parameters.tensors)
        accuracy, mse, _mae, _rmse = self.xgb_model.regression_metrics(self.test_data)
        global_round = ins.config["global_round"]
        log(
            INFO,
            f"accuracy = {accuracy:.4f}, mse = {mse:.4f} at round {global_round}",
        )

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(mse),
            num_examples=len(self.test_indices),
            metrics={"accuracy": float(accuracy), "mse": float(mse)},
        )


class FlowerClient:
    def __init__(self, context: Context, client_id: int, params: Dict[str, Any]):
        self.context = context
        self.client_id = client_id
        self.params = params

    def to_client(self):
        if self.params.get("model_type") == "xgboost":
            return XGBoostFlowerClient(
                self.context,
                self.client_id,
                self.params,
            )
        return TorchFlowerClient(
            self.context,
            self.client_id,
            self.params,
        ).to_client()
