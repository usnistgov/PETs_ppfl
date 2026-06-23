from report import Report
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
import xgboost as xgb
from datetime import datetime
####################################################################################################
#                                                                                                  #
#                                             Tasks                                                #
#                                                                                                  #
####################################################################################################

class ClassificationTask():
    def predictions(self, outputs):
        if isinstance(outputs, torch.Tensor):
            return torch.argmax(outputs, dim=1)
        return np.argmax(outputs, axis=1) if outputs.ndim == 2 else np.rint(outputs)

    def loss_and_predictions(self, criterion, outputs, labels):
        # raw logits for CE loss
        labels = labels.long()
        predictions = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        return loss, labels, predictions
    
    @classmethod
    def metrics(cls, labels, predictions, accuracy_tolerance=None):
        metrics = cls.get_classification_metrics(labels, predictions)
        metrics.update({
            "mae": 0,
            "mse": 0,
            "rmse": 0,
            "r2": 0,
            "mean": 0,
            "error_mean": 0,
        })
        return metrics
    
    def get_classification_metrics(labels, predictions, accuracy_tolerance=None):
        if len(labels) == 0:
            return {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}

        return {"accuracy": accuracy_score(labels, predictions),
                "precision_macro": precision_score(labels, predictions, average="macro", 
                                                   zero_division=0),
                "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
                "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0)}
    
    def add_task_report_metadata(self, metadata, class_labels=None, accuracy_tolerance=None):
        metadata["class labels"] = class_labels

    def add_report_metrics(self, metadata, train_metrics, test_metrics):
        metadata.update({"train precision macro": float(train_metrics["precision_macro"]),
                         "train recall macro": float(train_metrics["recall_macro"]),
                         "test precision macro": float(test_metrics["precision_macro"]),
                         "test recall macro": float(test_metrics["recall_macro"])})
        
    def format_metrics(self, name, metrics):
        return (f"{name} Macro Precision: {metrics['precision_macro']:.2f}, "
                f"{name} Macro Recall: {metrics['recall_macro']:.2f}, "
                f"{name} Macro F1: {metrics['f1_macro']:.2f}")
    

    @staticmethod
    def add_global_classification_report_metrics(metadata, precision_rounds, recall_rounds):
        metadata.update({
            "precision macro per round": np.array(precision_rounds),
            "recall macro per round": np.array(recall_rounds),
        })

    @classmethod
    def task_metrics(cls, labels, predictions):
        metrics = cls.get_classification_metrics(labels, predictions)
        metrics.update({"mae": 0, "mse": 0, "rmse": 0, "r2": 0, "mean": 0, "error_mean": 0})
        return metrics

class RegressionTask():
    def predictions(self, outputs):
        return outputs.squeeze() if isinstance(outputs, torch.Tensor) else outputs
    
    def loss_and_predictions(self, criterion, outputs, labels):
        # numeric values for regression loss
        labels = labels.float()
        predictions = outputs.squeeze()
        loss = criterion(predictions, labels)
        return loss, labels, predictions

    def metrics(self, labels, predictions, accuracy_tolerance):
        if len(labels) == 0:
            return {"accuracy": 0.0, "mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2": 0.0, "mean": 0.0, 
                    "error_mean": 0.0}

        labels = np.asarray(labels, dtype=float)
        predictions = np.asarray(predictions, dtype=float)
        mse = mean_squared_error(labels, predictions)
        rmse = mse**0.5
        mean = np.mean(labels)
        return {"accuracy": np.mean(np.abs(predictions - labels) <= accuracy_tolerance),
                "mae": mean_absolute_error(labels, predictions),
                "mse": mse,
                "rmse": rmse,
                "r2": r2_score(labels, predictions),
                "mean": mean,
                "error_mean": ((rmse / mean) * 100) if mean else 0}
    
    def add_task_report_metadata(self, metadata, class_labels=None, accuracy_tolerance=None):
        metadata["accuracy tolerance"] = accuracy_tolerance

    @staticmethod
    def add_report_metrics(metadata, train_metrics, test_metrics):
        return
    
    def format_metrics(name, metrics):
        return (f"{name} MAE: {metrics['mae']:.4f}, "
                f"{name} MSE: {metrics['mse']:.4f}, "
                f"{name} RMSE: {metrics['rmse']:.4f}, "
                f"{name} R2: {metrics['r2']:.4f}")

    @classmethod
    def task_metrics(cls, labels, predictions, accuracy_tolerance):
        return cls.get_regression_metrics(labels, predictions, accuracy_tolerance)

####################################################################################################
#                                                                                                  #
#                                           Base Class                                             #
#                                                                                                  #
####################################################################################################

class BaseModel:
    def __init__(self, model_id, output_dir, problem_type):
        self.model_id = model_id
        self.output_dir = output_dir
        self.model = None
        self.problem_type = problem_type
        self.task = (
            ClassificationTask()
            if problem_type == "classification"
            else RegressionTask()
        )

    def output_name(self, name, round_number):
        return f"{name}_round_{round_number}" if round_number is not None else name

    def output_path(self, filename, output_dir):
        out_dir = Path(output_dir or self.output_dir or Path(__file__).parent).absolute()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / filename

    def save_metadata(self, metadata, name, round_number, output_dir):
        out_name = self.output_name(name, round_number)
        metadata_path = self.output_path(f"{out_name}_meta.npz", output_dir)
        np.savez(metadata_path, **metadata, allow_pickle=True)

    
    def save_report(self, metadata, name, round_number, output_dir):
        out_name = self.output_name(name, round_number)
        report = Report(metadata)
        report.save_to_file(self.output_path(f"{out_name}.json", output_dir))

    def model_path(self, name, round_number, output_dir):
        return self.output_path(
            f"{self.output_name(name, round_number)}{self.model_extension}",
            output_dir,
        )

    def save_artifacts(self, metadata, name, round_number, output_dir):
        self.save_metadata(metadata, name, round_number, output_dir)
        self.save_report(metadata, name, round_number, output_dir)

    @staticmethod
    def add_epoch_report_metrics(metadata, train_acc, test_acc, losses, problem_type, train_mse=None, 
                                 test_mse=None, train_precision=None, test_precision=None, 
                                 train_recall=None, test_recall=None):
        metadata.update({
            "train accuracy per epoch": np.array(train_acc),
            "test accuracy per epoch": np.array(test_acc),
            "losses per epoch": np.array(losses),
        })

        if problem_type == "regression":
            metadata.update({
                "train mse per epoch": np.array(train_mse),
                "test mse per epoch": np.array(test_mse),
            })

        if problem_type == "classification":
            metadata.update({
                "train precision macro per epoch": np.array(train_precision),
                "test precision macro per epoch": np.array(test_precision),
                "train recall macro per epoch": np.array(train_recall),
                "test recall macro per epoch": np.array(test_recall),
            })

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, train_loader, test_loader, criterion, problem_type, accuracy_tolerance):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def get_parameters(self, *args, **kwargs):
        raise NotImplementedError

    def set_parameters(self, parameters):
        raise NotImplementedError

    def save_model(self, metadata, name, round_number, output_dir):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError
    
    @staticmethod
    def add_task_report_metadata(metadata, problem_type, class_labels, accuracy_tolerance):
        metadata["problem type"] = problem_type
        if problem_type == "classification":
            metadata["class labels"] = class_labels
        if problem_type == "regression":
            metadata["accuracy tolerance"] = accuracy_tolerance

    @staticmethod
    def add_global_classification_report_metrics(metadata, precision_rounds, recall_rounds):
        metadata.update({
            "precision macro per round": np.array(precision_rounds),
            "recall macro per round": np.array(recall_rounds),
        })

    @staticmethod
    def print_epoch_metrics(model_id, epoch, epochs, epoch_loss, train_accuracy, test_accuracy,
                            problem_type, train_metrics, test_metrics, mse, epoch_test_mse,
                            epsilon_spent=None):
        message = (f"Client {model_id} | Epoch {epoch + 1}/{epochs}, "
                   f"Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}, "
                   f"Test Acc: {test_accuracy:.2f}")
        if problem_type == "classification":
            message += (f", Train Macro Precision: {train_metrics['precision_macro']:.2f}, "
                        f"Train Macro Recall: {train_metrics['recall_macro']:.2f}, "
                        f"Train Macro F1: {train_metrics['f1_macro']:.2f}, "
                        f"Test Macro Precision: {test_metrics['precision_macro']:.2f}, "
                        f"Test Macro Recall: {test_metrics['recall_macro']:.2f}, "
                        f"Test Macro F1: {test_metrics['f1_macro']:.2f}")
        else:
            message += (f", Train MSE: {mse:.4f}, Test MSE: {epoch_test_mse:.4f}, "
                        f"MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
        if epsilon_spent is not None:
            message += f", ε: {epsilon_spent:.2f}"
        print(message)

    def add_epoch_values(
        self,
        train_mse,
        test_mse,
        train_acc,
        test_acc,
        train_precision,
        test_precision,
        train_recall,
        test_recall,
        train_metrics,
        test_metrics,
        epoch_train_mse,
        epoch_test_mse,
        train_accuracy,
        test_accuracy,
    ):
        train_mse.append(epoch_train_mse)
        test_mse.append(epoch_test_mse)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)

        if self.problem_type == "classification":
            train_precision.append(train_metrics["precision_macro"])
            test_precision.append(test_metrics["precision_macro"])
            train_recall.append(train_metrics["recall_macro"])
            test_recall.append(test_metrics["recall_macro"])
    
class TorchModelBase(BaseModel):
    model_extension = ".torch"

    def load_model(self, path, num_features, output_dim, device):
        if self.model is None:
            if num_features is None:
                raise ValueError("num_features is required when loading into an unbuilt model")
            self.build_model(num_features, output_dim=output_dim)

        state_dict = torch.load(path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        return self.model

    def save_model(self, metadata, name, round_number, output_dir):
        torch.save(self.model.state_dict(), self.model_path(name, round_number, output_dir))
        self.save_artifacts(metadata, name, round_number, output_dir)

    def get_parameters(self, config):
        self.model.eval()
        return [
            val.detach().cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):
        state_dict = {}

        for (key, ref_tensor), value in zip(self.model.state_dict().items(), parameters):
            state_dict[key] = torch.tensor(
                value,
                dtype=ref_tensor.dtype,
                device=ref_tensor.device,
            )

        self.model.load_state_dict(state_dict, strict=True)

    def build_optimizer(self, optimizer_name, learning_rate, weight_decay):
        if self.model is None:
            raise RuntimeError("build_model must be called before build_optimizer")

        if optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        if optimizer_name == "adamax":
            return optim.Adamax(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
    def predict(self, data, device):
        self.model.eval()

        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)

            data = data.float().to(device)

            if data.dim() == 2:
                data = data.unsqueeze(1)

            outputs = self.model(data).squeeze()

        return outputs.cpu().numpy()

    def unpack_batch(self, data, device):
        if isinstance(data, (tuple, list)):
            inputs, labels = data
        else:
            inputs = data[:, :-1]
            labels = data[:, -1]

        inputs = inputs.float()
        labels = labels.float()

        if device is not None:
            inputs = inputs.to(device)
            labels = labels.to(device)

        return inputs, labels
    
    def evaluate(self, train_loader, test_loader, criterion, problem_type, accuracy_tolerance):
        self.model.eval()
        device = next(self.model.parameters()).device
        self.model.to(device)
        train = self.evaluate_loader(train_loader, criterion, device, accuracy_tolerance)
        test = self.evaluate_loader(test_loader, criterion, device, accuracy_tolerance)

        print(f"\nClient {self.model_id} | Final Results:")
        if self.problem_type == "classification":
            print(f"Client {self.model_id} | Train: Accuracy: {train['accuracy']:.2f}, Loss: {train['loss']:.4f}")
            print(f"Client {self.model_id} | Test: Accuracy: {test['accuracy']:.2f}, Loss: {test['loss']:.4f}")
        else:
            print(f"Client {self.model_id} | Train: Accuracy: {train['accuracy']:.2f}, "
                  f"Loss: {train['loss']:.4f}, MAE: {train['mae']:.4f}, "
                  f"MSE: {train['mse']:.4f}, RMSE: {train['rmse']:.4f}")
            print(f"Client {self.model_id} | Test: Accuracy: {test['accuracy']:.2f}, "
                  f"Loss: {test['loss']:.4f}, MAE: {test['mae']:.4f}, "
                  f"MSE: {test['mse']:.4f}, RMSE: {test['rmse']:.4f}")
        return (train["accuracy"], test["accuracy"], train["loss"], test["loss"], train.get("mse", 0),
                test.get("mse", 0), train["predictions"], test["predictions"], train, test)
    
    def evaluate_loader(self, loader, criterion, device, accuracy_tolerance):
        total_loss = 0
        processed_batches = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                inputs, labels = self.unpack_batch(batch, device)
                if inputs.shape[0] < 2:
                    continue

                outputs = self.model(inputs)
                loss, labels, predictions = self.task.loss_and_predictions(
                    criterion,
                    outputs,
                    labels,
                )

                total_loss += loss.item()
                processed_batches += 1
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = self.task.metrics(all_labels, all_preds, accuracy_tolerance)
        metrics["loss"] = total_loss / processed_batches if processed_batches else 0
        metrics["predictions"] = all_preds
        return metrics

    def compute_test_metrics(self, test_loader, criterion, device, accuracy_tolerance):
        metrics = self.evaluate_loader(test_loader, criterion, device, accuracy_tolerance)
        return (metrics["mse"], metrics["accuracy"], metrics["loss"], 
                metrics if self.problem_type == "classification" else {})
    
    def _before_fit(self, fit_context):
        return fit_context

    def _after_epoch(self, epoch, fit_context):
        return {}

    def fit(self, train_loader, test_loader, epochs, optimizer, criterion, device, problem_type,
            accuracy_tolerance, **kwargs):
        fit_context = {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "epochs": epochs,
            "optimizer": optimizer,
            "criterion": criterion,
            "device": device,
            "problem_type": problem_type,
            "accuracy_tolerance": accuracy_tolerance,
            **kwargs,
        }

        fit_context = self._before_fit(fit_context)

        self.model.train()
        train_mse = []
        test_mse = []
        train_acc = []
        test_acc = []
        train_precision = []
        test_precision = []
        train_recall = []
        test_recall = []
        losses = []
        extra_history = {}

        for epoch in range(epochs):
            epoch_loss = 0
            train_labels, train_preds = [], []

            for data in fit_context["train_loader"]:
                inputs, labels = self.unpack_batch(data, fit_context["device"])
                if inputs.shape[0] < 2:
                    continue

                fit_context["optimizer"].zero_grad()
                outputs = self.model(inputs)
                loss, labels, pred_values = self.task.loss_and_predictions(
                    fit_context["criterion"],
                    outputs,
                    labels,
                )
                loss.backward()
                fit_context["optimizer"].step()

                epoch_loss += loss.item()
                train_labels.extend(labels.cpu().numpy())
                train_preds.extend(pred_values.detach().cpu().numpy())

            train_metrics = self.task.metrics(
                train_labels,
                train_preds,
                fit_context["accuracy_tolerance"],
            )
            train_accuracy = train_metrics["accuracy"]
            mse = train_metrics.get("mse", 0)

            self.model.eval()
            test_metrics = self.evaluate_loader(
                fit_context["test_loader"],
                fit_context["criterion"],
                fit_context["device"],
                fit_context["accuracy_tolerance"],
            )
            self.model.train()

            epoch_test_acc = test_metrics["accuracy"]
            epoch_test_mse = test_metrics.get("mse", 0)

            self.add_epoch_values(
                train_mse, test_mse, train_acc, test_acc,
                train_precision, test_precision, train_recall, test_recall,
                train_metrics, test_metrics, mse, epoch_test_mse,
                train_accuracy, epoch_test_acc,
            )
            losses.append(epoch_loss)

            epoch_extras = self._after_epoch(epoch, fit_context)
            for key, value in epoch_extras.items():
                extra_history.setdefault(key, []).append(value)

            self.print_epoch_metrics(
                self.model_id,
                epoch,
                epochs,
                epoch_loss,
                train_accuracy,
                epoch_test_acc,
                fit_context["problem_type"],
                train_metrics,
                test_metrics,
                mse,
                epoch_test_mse,
                **epoch_extras,
            )

        return (train_mse, test_mse, train_acc, test_acc, train_precision, test_precision,
            train_recall, test_recall, losses, extra_history)
 
####################################################################################################
#                                                                                                  #
#                                         Callable Classes                                         #
#                                                                                                  #
####################################################################################################

class CNNModel(TorchModelBase):
    def __init__(self, model_id, output_dir, problem_type):
        super().__init__(model_id=model_id, output_dir=output_dir, problem_type=problem_type)

    def build_model(self, num_features, output_dim):
        self.model = _CNNNet(num_features, output_dim=output_dim)
        return self.model

class DPCNNModel(TorchModelBase):
    def __init__(self, model_id, output_dir, problem_type):
        super().__init__(model_id=model_id, output_dir=output_dir, problem_type=problem_type)

    def build_model(self, num_features, output_dim):
        self.model = _CNNNet(num_features, output_dim=output_dim)
        return self.model

    def attach_privacy_engine(self, optimizer, train_loader, opacus_params):
        privacy_engine = PrivacyEngine(
            accountant="rdp",
            secure_mode=opacus_params.get("secure_mode"),
        )

        private_model, private_optimizer, private_train_loader = (
            privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=opacus_params.get("epochs"),
                target_epsilon=opacus_params.get("epsilon"),
                target_delta=opacus_params.get("delta"),
                max_grad_norm=opacus_params.get("max_grad_norm"),
            )
        )

        privacy_engine.accountant.alphas = [1 + x / 10.0 for x in range(1000)]

        self.model = private_model
        return private_optimizer, private_train_loader, privacy_engine
    
    def _before_fit(self, fit_context):
        opacus_params = fit_context.get("opacus_params")
        if opacus_params is None:
            return fit_context

        private_optimizer, private_train_loader, privacy_engine = self.attach_privacy_engine(
            fit_context["optimizer"],
            fit_context["train_loader"],
            opacus_params,
        )

        fit_context["optimizer"] = private_optimizer
        fit_context["train_loader"] = private_train_loader
        fit_context["privacy_engine"] = privacy_engine
        fit_context["delta"] = opacus_params.get("delta")

        return fit_context

    def _after_epoch(self, epoch, fit_context):
        privacy_engine = fit_context.get("privacy_engine")
        delta = fit_context.get("delta")

        if privacy_engine is None or delta is None:
            return {}

        epsilon_spent, _ = privacy_engine.accountant.get_privacy_spent(delta=delta)
        return {"epsilon_spent": epsilon_spent}

class XGBoostModel(BaseModel):
    model_extension = ".ubj"
    def __init__(self, model_id, output_dir, params, problem_type, class_labels, accuracy_tolerance):
        super().__init__(model_id=model_id, output_dir=output_dir, problem_type=problem_type)
        self.params = params.copy()
        self.class_labels = class_labels
        self.accuracy_tolerance = accuracy_tolerance
        self.train_loss_per_epoch = []
        self.test_loss_per_epoch = []
        self.final_train_loss = 0.0
        self.final_test_loss = 0.0
        self._configure_xgb_params()
    
    def _configure_xgb_params(self):
        if self.problem_type == "classification":
            if len(self.class_labels) == 2:
                self.objective = "binary:logistic"
                self.eval_metric = "logloss"
                self.params["objective"] = self.objective
                self.params["eval_metric"] = self.eval_metric
                self.params.pop("num_class", None)
            else:
                self.objective = "multi:softprob"
                self.eval_metric = "mlogloss"
                self.num_class = len(self.class_labels)
                self.params["objective"] = self.objective
                self.params["eval_metric"] = self.eval_metric
                self.params["num_class"] = self.num_class
        else:
            self.objective = self.params.get("objective")
            self.eval_metric = self.params.get("eval_metric", "rmse")

    @classmethod
    def client_params(cls, seed, num_partitions, train_method, scaled_lr, base_params):
        params = base_params.copy()
        params["random_state"] = seed

        if train_method == "bagging" and scaled_lr:
            params["eta"] = params["eta"] / num_partitions if num_partitions else 0

        return params

    def build_model(self, params):
        self.params = params or self.params
        self.model = xgb.Booster(params=self.params)
        return self.model

    def get_parameters(self, config):
        if self.model is None:
            return []
        return [bytes(self.model.save_raw("json"))]

    def set_parameters(self, parameters):
        self.build_model(self.params)
        if not parameters:
            return self.model
        self.model.load_model(bytearray(parameters[-1]))
        return self.model

    def fit_round(self, train_data, test_data, num_local_round, train_method, global_round, parameters):
        if global_round > 1 and parameters:
            self.set_parameters(parameters)

        self.fit(train_data, test_data, num_local_round, train_method)
        return self.get_parameters(None)[0]

    def evaluate_parameters(self, parameters, test_data):
        if not parameters:
            metric_name = self.params.get("eval_metric", "metric")
            return metric_name, 0.0

        self.set_parameters(parameters)
        return self.evaluate(test_data)

    def fit(self, train_data, test_data, num_local_round, train_method):
        self.train_mse_per_epoch = []
        self.test_mse_per_epoch = []
        self.train_acc_per_epoch = []
        self.test_acc_per_epoch = []
        self.train_precision_per_epoch = []
        self.test_precision_per_epoch = []
        self.train_recall_per_epoch = []
        self.test_recall_per_epoch = []
        self.losses = []
        self.final_train_loss = 0.0
        self.final_test_loss = 0.0

        for i in range(num_local_round):
            evals_result = {}

            if self.model is None or self.model.num_boosted_rounds() == 0:
                self.model = xgb.train(
                    self.params,
                    train_data,
                    num_boost_round=1,
                    evals=[
                        (test_data, "validate"),
                        (train_data, "train"),
                    ],
                    evals_result=evals_result,
                    verbose_eval=False,
                )
            else:
                self.model = xgb.train(
                    self.params,
                    train_data,
                    num_boost_round=1,
                    evals=[
                        (test_data, "validate"),
                        (train_data, "train"),
                    ],
                    xgb_model=self.model,
                    evals_result=evals_result,
                    verbose_eval=False,
                )

            train_metrics, train_predictions = self.dataset_metrics(train_data)
            test_metrics, test_predictions = self.dataset_metrics(test_data)

            train_acc = train_metrics["accuracy"]
            test_acc = test_metrics["accuracy"]
            train_mse = train_metrics.get("mse", 0)
            test_mse = test_metrics.get("mse", 0)

            if self.problem_type == "classification":
                metric_key = next(iter(evals_result["train"]))
                self.final_train_loss = float(evals_result["train"][metric_key][-1])
                self.final_test_loss = float(evals_result["validate"][metric_key][-1])
                displayed_loss = self.final_train_loss
            else:
                self.final_train_loss = float(train_mse)
                self.final_test_loss = float(test_mse)
                displayed_loss = self.final_train_loss

            self.losses.append(self.final_train_loss)

            self.add_epoch_values(
                self.train_mse_per_epoch,
                self.test_mse_per_epoch,
                self.train_acc_per_epoch,
                self.test_acc_per_epoch,
                self.train_precision_per_epoch,
                self.test_precision_per_epoch,
                self.train_recall_per_epoch,
                self.test_recall_per_epoch,
                train_metrics,
                test_metrics,
                train_mse,
                test_mse,
                train_acc,
                test_acc,
            )

            self.print_epoch_metrics(
                self.model_id,
                i,
                num_local_round,
                displayed_loss,
                train_acc,
                test_acc,
                self.problem_type,
                train_metrics,
                test_metrics,
                train_mse,
                test_mse,
            )

        if train_method == "bagging":
            start = self.model.num_boosted_rounds() - num_local_round
            self.model = self.model[start:self.model.num_boosted_rounds()]

        return self.model
    
    def evaluate(self, test_data):
        if self.model is None:
            raise RuntimeError("build_model or set_parameters must be called before evaluate")

        eval_result = self.model.eval_set(
            evals=[(test_data, "valid")],
            iteration=self.model.num_boosted_rounds() - 1,
        )
        metric_value = float(eval_result.rsplit(":", 1)[-1])
        return self.eval_metric, metric_value

    def dataset_metrics(self, data):
        labels = data.get_label()
        predictions = self.task.predictions(self.model.predict(data))
        metrics = self.task.metrics(labels, predictions, self.accuracy_tolerance)
        return metrics, predictions

    def client_metadata(
        self,
        train_data,
        test_data,
        train_indices,
        test_indices,
        partitions_file,
        seed,
        test_fraction,
        global_round,
    ):
        train_metrics, train_predictions = self.dataset_metrics(train_data)
        test_metrics, test_predictions = self.dataset_metrics(test_data)

        train_acc = train_metrics["accuracy"]
        test_acc = test_metrics["accuracy"]
        train_mse = train_metrics.get("mse", 0)
        test_mse = test_metrics.get("mse", 0)

        metadata = {
            "created on": str(datetime.now()),
            "model id": int(self.model_id),
            "round number": int(global_round),
            "partitions file": partitions_file,
            "problem type": self.problem_type,
        }

        self.task.add_task_report_metadata(
            metadata,
            class_labels=self.class_labels,
            accuracy_tolerance=self.accuracy_tolerance,
        )

        metadata.update({
            "train accuracy": float(train_acc),
            "test accuracy": float(test_acc),
            "train loss": float(self.final_train_loss),
            "test loss": float(self.final_test_loss),
            "train indices": train_indices,
            "test indices": test_indices,
            "train predictions": train_predictions,
            "test predictions": test_predictions,
            "hyperparameters": {
                **self.params,
                "seed": seed,
                "test_fraction": test_fraction,
                "test fraction": test_fraction,
            },
        })

        if self.problem_type == "regression":
            metadata.update({
                "train mean squared error": float(train_mse),
                "test mean squared error": float(test_mse),
            })

        self.task.add_report_metrics(metadata, train_metrics, test_metrics)
        self.add_epoch_report_metrics(
            metadata,
            self.train_acc_per_epoch,
            self.test_acc_per_epoch,
            self.losses,
            problem_type=self.problem_type,
            train_mse=self.train_mse_per_epoch,
            test_mse=self.test_mse_per_epoch,
            train_precision=self.train_precision_per_epoch if self.problem_type == "classification" else None,
            test_precision=self.test_precision_per_epoch if self.problem_type == "classification" else None,
            train_recall=self.train_recall_per_epoch if self.problem_type == "classification" else None,
            test_recall=self.test_recall_per_epoch if self.problem_type == "classification" else None,
        )
        return metadata, train_acc, test_acc

    def save_client_output(
        self,
        train_data,
        test_data,
        train_indices,
        test_indices,
        partitions_file,
        seed,
        test_fraction,
        global_round,
        output_dir,
    ):
        metadata, train_acc, test_acc = self.client_metadata(
            train_data,
            test_data,
            train_indices,
            test_indices,
            partitions_file,
            seed,
            test_fraction,
            global_round,
        )
        self.save_model(
            metadata,
            f"xgboost_client_{self.model_id}",
            global_round,
            output_dir,
        )
        return train_acc, test_acc

    def predict(self, model, data):
        return self.task.predictions(model.predict(data))

    def save_model(self, metadata, name, round_number, output_dir):
        self.model.save_model(self.model_path(name, round_number, output_dir))
        self.save_artifacts(metadata, name, round_number, output_dir)

    def load_model(self, path):
        self.build_model(self.params)
        self.model.load_model(path)
        return self.model

    def save_raw_parameters(self, path):
        if self.model is None:
            raise RuntimeError("No model has been built or trained")
        Path(path).write_bytes(bytes(self.model.save_raw("json")))

####################################################################################################
#                                                                                                  #
#                                       Helper Model Class                                         #
#                                                                                                  #
####################################################################################################

class _CNNNet(nn.Module):
    def __init__(self, features: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 12, kernel_size=14)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(12, 10, kernel_size=10)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(10, 8, kernel_size=8)
        self.relu3 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        flatten_len = self._calculate_flatten_size(features)
        self.layer_norm1 = nn.LayerNorm([8, flatten_len // 8])
        self.fc1 = nn.Linear(in_features=flatten_len, out_features=48)
        self.fc2 = nn.Linear(in_features=48, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.layer_norm2 = nn.LayerNorm(16)
        self.output = nn.Linear(in_features=16, out_features=output_dim)

    def _calculate_flatten_size(self, features: int):
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.zeros(1, 1, features).to(device)
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.pool(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        # Convert to tensor if input is an ndarray
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(
                next(self.parameters()).device
            )
        x = x.float()
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool(x)
        x = self.layer_norm1(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.layer_norm2(x)
        x = self.output(x)
        return x
