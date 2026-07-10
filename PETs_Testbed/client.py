# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software

from typing import Any, Dict
from pathlib import Path
from datetime import datetime
from logging import INFO
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import (
    Code,
    Context,
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
from xgboost.core import DMatrix

from dataset import load_partitions, load_npy_feature_label_data
from model import CNNModel, DPCNNModel, XGBoostModel
from utils import configure_warning_logging, get_device

DEVICE = get_device()

def empty_evaluate_res(message="No model parameters available for evaluation.") -> EvaluateRes:
    """Return an empty successful Flower evaluation response."""
    return EvaluateRes(
        status=Status(code=Code.OK, message=message),
        loss=0.0,
        num_examples=0,
        metrics={"accuracy": 0.0, "mse": 0.0},
    )

def create_dataloaders(client_id, data_partitions_file, data_directory, num_partitions, partitioner_type,
                       test_fraction, seed, batch_size, problem_type, class_labels=None, 
                       use_public_data=False,using_xgboost=False):
    """Load client data partitions and return loaders or XGBoost matrices."""
    tt_vcf, tt_pheno, _, _, pub_vcf, pub_pheno = load_npy_feature_label_data(data_directory)
    if use_public_data:
        vcf = [tt_vcf, pub_vcf]
        pheno = [tt_pheno, pub_pheno]
    else:
        vcf = [tt_vcf]
        pheno = [tt_pheno]
    num_data_features = tt_vcf.shape[1]

    data_partitions = None
    if data_partitions_file and Path(data_partitions_file).exists():
        data_partitions = np.load(data_partitions_file)
        partitions_path = Path(data_partitions_file).name
        print(f"Loaded data partitions from {data_partitions_file}")
    else:
        partitions_path = "none"
        print(f"Data partitions file not found at {data_partitions_file}. "
              f"Using a random partition for client {client_id}")

    train_loader, test_loader, train_indices, test_indices = \
        load_partitions(client_id, vcf, pheno, batch_size, test_fraction, seed, num_partitions,
                        partitioner_type, data_directory, problem_type, class_labels, data_partitions)

    if using_xgboost:
        train_loader = loader_to_dmatrix(train_loader)
        test_loader = loader_to_dmatrix(test_loader)
    return num_data_features, partitions_path, train_loader, test_loader, train_indices, test_indices

def loader_to_dmatrix(loader):
    """Convert a PyTorch loader into an XGBoost DMatrix."""
    features, labels = [], []
    for batch_features, batch_labels in loader:
        features.append(np.asarray(batch_features))
        labels.append(np.asarray(batch_labels).reshape(-1))
    return DMatrix(data=np.concatenate(features), label=np.concatenate(labels))

class TorchFlowerClient(fl.client.NumPyClient):
    def __init__(self, context: Context, client_id: int, params: Dict[str, Any]):
        """Initialize the TorchFlowerClient instance."""
        self.context = context
        self.client_state = context.state
        self.client_id = client_id
        self.params = params
        p = params

        self.current_round = 0
        self.output_dir = p["output_dir"]
        self.problem_type = p["problem_type"]
        self.class_labels = p.get("class_labels")
        self.num_classes = p.get("num_classes")
        self.accuracy_tolerance = p.get("accuracy_tolerance", 0.1)
        self.use_dp = p.get("model_type") == "dpcnn"
        self.eps_per_epoch = []

        torch.manual_seed(p["seed"])
        if not p.get("print_warning_logs"):
            configure_warning_logging(self.output_dir)

        (
            self.num_data_features,
            self.partitions_file,
            self.train_loader,
            self.test_loader,
            self.train_indices,
            self.test_indices,
        ) = create_dataloaders(
            client_id=self.client_id,
            data_partitions_file=p.get("data_partitions_file"),
            data_directory=p.get("data_dir"),
            num_partitions=p.get("num_partitions"),
            partitioner_type=p.get("partitions_type"),
            test_fraction=p.get("test_fraction"),
            seed=p.get("seed"),
            batch_size=p.get("batch_size"),
            problem_type=self.problem_type,
            class_labels=self.class_labels,
        )

        model_class = DPCNNModel if self.use_dp else CNNModel
        self.cnn_model = model_class(
            model_id=self.client_id,
            output_dir=self.output_dir,
            problem_type=self.problem_type,
        )
        self.cnn_model.build_model(
            self.num_data_features,
            output_dim=self.num_classes if self.problem_type == "classification" else 1,
        )
        self.cnn_model.model.to(DEVICE)

        self.criterion = nn.CrossEntropyLoss() if self.problem_type == "classification" else nn.MSELoss()
        self.optimizer = self.cnn_model.build_optimizer(
            p.get("optimizer_name"),
            p.get("learning_rate"),
            p.get("weight_decay"),
        )

        self.privacy_engine = None
        if self.use_dp:
            opacus_params = {
                "epochs": p.get("epochs"),
                "epsilon": p.get("epsilon"),
                "delta": p.get("delta"),
                "max_grad_norm": p.get("max_grad_norm"),
                "secure_mode": p.get("opacus_secure_mode"),
            }
            self.optimizer, self.train_loader, self.privacy_engine = \
                self.cnn_model.attach_privacy_engine(self.optimizer, self.train_loader, opacus_params)

        self.train_acc_per_epoch = []
        self.test_acc_per_epoch = []
        self.train_mse_per_epoch = []
        self.test_mse_per_epoch = []
        self.train_precision_per_epoch = []
        self.test_precision_per_epoch = []
        self.train_recall_per_epoch = []
        self.test_recall_per_epoch = []
        self.losses = []

    def hyperparameters_metadata(self):
        """Build report metadata for the client hyperparameters."""
        p = self.params
        hyperparameters = {
            "learning rate": float(p.get("learning_rate")),
            "weight decay": float(p.get("weight_decay")),
            "batch size": int(p.get("batch_size")),
            "epochs": int(p.get("epochs")),
            "seed": int(p.get("seed")),
            "test fraction": float(p.get("test_fraction")),
        }
        if self.use_dp:
            hyperparameters.update({
                "epsilon": float(p.get("epsilon")),
                "delta": float(p.get("delta")),
                "max grad norm": float(p.get("max_grad_norm")),
                "opacus secure mode": bool(p.get("opacus_secure_mode")),
            })
        return hyperparameters

    def epoch_metrics_path(self):
        """Return the temporary path for this round's epoch metrics."""
        return Path(self.output_dir, f".client_{self.client_id}_round_{self.current_round}_epoch_metrics.npz")

    def save_epoch_metrics(self):
        """Persist per-epoch metrics for later evaluation reporting."""
        path = self.epoch_metrics_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            train_acc=self.train_acc_per_epoch,
            test_acc=self.test_acc_per_epoch,
            train_mse=self.train_mse_per_epoch,
            test_mse=self.test_mse_per_epoch,
            train_precision=self.train_precision_per_epoch,
            test_precision=self.test_precision_per_epoch,
            train_recall=self.train_recall_per_epoch,
            test_recall=self.test_recall_per_epoch,
            losses=self.losses,
            eps=self.eps_per_epoch,
        )

    def load_epoch_metrics(self):
        """Load and remove saved per-epoch metrics for the current round."""
        path = self.epoch_metrics_path()
        if not path.exists():
            return {
                "train_acc": self.train_acc_per_epoch,
                "test_acc": self.test_acc_per_epoch,
                "train_mse": self.train_mse_per_epoch,
                "test_mse": self.test_mse_per_epoch,
                "train_precision": self.train_precision_per_epoch,
                "test_precision": self.test_precision_per_epoch,
                "train_recall": self.train_recall_per_epoch,
                "test_recall": self.test_recall_per_epoch,
                "losses": self.losses,
                "eps": self.eps_per_epoch,
            }
        metrics = dict(np.load(path, allow_pickle=True))
        path.unlink(missing_ok=True)
        return metrics

    def build_metadata(self, train_eval, test_eval, epoch_metrics):
        """Build the JSON/NPZ report metadata for a Torch client round."""
        train_acc, test_acc, train_loss, test_loss, train_mse, test_mse, train_preds, test_preds, train_metrics, test_metrics = (
            train_eval[0], test_eval[0], train_eval[1], test_eval[1],
            train_eval[2], test_eval[2], train_eval[3], test_eval[3],
            train_eval[4], test_eval[4],
        )

        metadata = {
            "created on": str(datetime.now()),
            "model id": int(self.client_id),
            "round number": int(self.current_round),
            "partitions file": self.partitions_file,
            "train accuracy": float(train_acc),
            "test accuracy": float(test_acc),
            "train loss": float(train_loss),
            "test loss": float(test_loss),
            "train indices": self.train_indices,
            "test indices": self.test_indices,
            "train predictions": train_preds,
            "test predictions": test_preds,
            "hyperparameters": self.hyperparameters_metadata(),
        }

        self.cnn_model.add_task_report_metadata(
            metadata,
            self.problem_type,
            self.class_labels,
            self.accuracy_tolerance,
        )

        if self.problem_type == "regression":
            metadata.update({
                "train mean squared error": float(train_mse),
                "test mean squared error": float(test_mse),
            })

        self.cnn_model.add_epoch_report_metrics(
            metadata,
            epoch_metrics["train_acc"],
            epoch_metrics["test_acc"],
            epoch_metrics["losses"],
            self.problem_type,
            train_mse=epoch_metrics["train_mse"],
            test_mse=epoch_metrics["test_mse"],
            train_precision=epoch_metrics["train_precision"] if self.problem_type == "classification" else None,
            test_precision=epoch_metrics["test_precision"] if self.problem_type == "classification" else None,
            train_recall=epoch_metrics["train_recall"] if self.problem_type == "classification" else None,
            test_recall=epoch_metrics["test_recall"] if self.problem_type == "classification" else None,
        )

        if self.use_dp:
            metadata["epsilon per epoch"] = np.array(epoch_metrics["eps"])

        self.cnn_model.task.add_report_metrics(metadata, train_metrics, test_metrics)
        return metadata

    def get_parameters(self, config):
        """Return model parameters in Flower NumPyClient format."""
        return self.cnn_model.get_parameters(config)

    def set_parameters(self, parameters):
        """Load Flower parameters into the local Torch model."""
        self.cnn_model.set_parameters(parameters)

    def fit(self, parameters, config):
        """Train the local Torch model for one federated round."""
        self.current_round = config.get("server_round", 1) - 1
        if not self.use_dp or self.current_round > 0:
            self.set_parameters(parameters)

        fit_kwargs = {
            "problem_type": self.problem_type,
            "accuracy_tolerance": self.accuracy_tolerance,
        }
        if self.use_dp:
            fit_kwargs.update({
                "privacy_engine": self.privacy_engine,
                "delta": self.params.get("delta"),
            })

        (
            self.train_mse_per_epoch,
            self.test_mse_per_epoch,
            self.train_acc_per_epoch,
            self.test_acc_per_epoch,
            self.train_precision_per_epoch,
            self.test_precision_per_epoch,
            self.train_recall_per_epoch,
            self.test_recall_per_epoch,
            self.losses,
            extra_history,
        ) = self.cnn_model.fit(
            self.train_loader,
            self.test_loader,
            self.params.get("epochs"),
            self.optimizer,
            self.criterion,
            DEVICE,
            **fit_kwargs,
        )

        self.eps_per_epoch = extra_history.get("epsilon_spent", [])
        self.save_epoch_metrics()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the local Torch model and save round artifacts."""
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
            train_metrics,
            test_metrics,
        ) = self.cnn_model.evaluate(
            self.train_loader,
            self.test_loader,
            self.criterion,
            problem_type=self.problem_type,
            accuracy_tolerance=self.accuracy_tolerance,
        )

        epoch_metrics = self.load_epoch_metrics()
        metadata = self.build_metadata(
            train_eval=(train_acc, train_loss, train_mse, train_preds, train_metrics),
            test_eval=(test_acc, test_loss, test_mse, test_preds, test_metrics),
            epoch_metrics=epoch_metrics,
        )

        name = f"dpcnn{self.params.get('epsilon')}_client_{self.client_id}" if self.use_dp else f"flcnn_client_{self.client_id}"
        self.cnn_model.save_model(metadata, name, self.current_round, self.output_dir)

        return float(test_loss), len(self.test_loader.dataset), {
            "accuracy": float(test_acc),
            "mse": float(test_mse),
        }

class XGBoostFlowerClient(fl.client.Client):
    def __init__(self, context: Context, client_id: int, params: Dict[str, Any]):
        """Initialize the XGBoostFlowerClient instance."""
        self.context = context
        self.client_id = client_id
        self.params = params
        p = params

        self.output_dir = p.get("output_dir")
        self.seed = p.get("seed")
        self.train_method = p.get("train_method")
        self.num_partitions = p.get("num_partitions")
        self.num_local_round = p.get("epochs")
        self.test_fraction = p.get("test_fraction")
        self.problem_type = p.get("problem_type")
        self.class_labels = p.get("class_labels")
        self.accuracy_tolerance = p.get("accuracy_tolerance", 0.1)

        if not p.get("print_warning_logs"):
            configure_warning_logging(self.output_dir)

        (
            self.num_data_features,
            self.partitions_file,
            self.train_data,
            self.test_data,
            self.train_indices,
            self.test_indices,
        ) = create_dataloaders(
            client_id=self.client_id,
            data_partitions_file=p.get("data_partitions_file"),
            data_directory=p.get("data_dir"),
            num_partitions=p.get("num_partitions"),
            partitioner_type=p.get("partitions_type"),
            test_fraction=p.get("test_fraction"),
            seed=p.get("seed"),
            batch_size=p.get("batch_size"),
            problem_type=self.problem_type,
            class_labels=self.class_labels,
            using_xgboost=True
        )

        self.xgb_model = XGBoostModel(
            model_id=self.client_id,
            output_dir=self.output_dir,
            params=XGBoostModel.client_params(
                seed=self.seed,
                num_partitions=self.num_partitions,
                train_method=self.train_method,
                scaled_lr=p.get("scaled_lr"),
                base_params=p.get("xgboost_params"),
            ),
            problem_type=self.problem_type,
            class_labels=self.class_labels,
            accuracy_tolerance=self.accuracy_tolerance,
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return an empty initial parameter payload for XGBoost clients."""
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Train the local XGBoost model for one federated round."""
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

        print(f"client {self.client_id} Train accuracy: {train_acc * 100:.2f}%", flush=True)
        print(f"client {self.client_id} Test accuracy: {test_acc * 100:.2f}%", flush=True)

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=len(self.test_indices),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the local XGBoost model for Flower aggregation."""
        if not ins.parameters.tensors:
            return empty_evaluate_res()

        self.xgb_model.set_parameters(ins.parameters.tensors)

        labels = self.test_data.get_label()
        predictions = self.xgb_model.predict(self.xgb_model.model, self.test_data)
        test_metrics = self.xgb_model.task.metrics(labels, predictions, self.accuracy_tolerance)

        accuracy = test_metrics["accuracy"]
        mse = test_metrics.get("mse", 0.0)
        loss = self.xgb_model.evaluate(self.test_data)[1] if self.problem_type == "classification" else mse

        log(INFO, f"accuracy = {accuracy:.4f}, mse = {mse:.4f} at round {ins.config['global_round']}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(loss),
            num_examples=len(self.test_indices),
            metrics={"accuracy": float(accuracy), "mse": float(mse)},
        )

class FlowerClient:
    def __init__(self, context: Context, client_id: int, params: Dict[str, Any]):
        """Initialize the FlowerClient instance."""
        self.context = context
        self.client_id = client_id
        self.params = params

    def to_client(self):
        """Create the concrete Flower client for the configured model type."""
        client_cls = XGBoostFlowerClient if self.params.get("model_type") == "xgboost" else TorchFlowerClient
        client = client_cls(self.context, self.client_id, self.params)
        return client if client_cls is XGBoostFlowerClient else client.to_client()
