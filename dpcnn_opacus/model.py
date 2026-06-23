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
#                                           Base Class                                             #
#                                                                                                  #
####################################################################################################

class BaseModel:
    def __init__(self, model_id, output_dir):
        self.model_id = model_id
        self.output_dir = output_dir
        self.model = None

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

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, train_loader, test_loader, criterion, problem_type, accuracy_tolerance):
        self.model.eval()
        device = next(self.model.parameters()).device
        self.model.to(device)
        train = self.evaluate_loader(train_loader, criterion, None, problem_type, accuracy_tolerance)
        test = self.evaluate_loader(test_loader, criterion, None, problem_type, accuracy_tolerance)

        print(f"\nClient {self.model_id} | Final Results:")
        if problem_type == "classification":
            self.print_final_classification_metrics("Train", train)
            self.print_final_classification_metrics("Test", test)
        else:
            self.print_final_regression_metrics("Train", train)
            self.print_final_regression_metrics("Test", test)

        return (
            train["accuracy"], test["accuracy"], train["loss"], test["loss"],
            train["mse"], test["mse"], train["predictions"], test["predictions"],
            train, test
        )

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

    @staticmethod
    def task_predictions(outputs, problem_type):
        if problem_type == "classification":
            if isinstance(outputs, torch.Tensor):
                return torch.argmax(outputs, dim=1)
            return np.argmax(outputs, axis=1) if outputs.ndim == 2 else np.rint(outputs)

        return outputs.squeeze() if isinstance(outputs, torch.Tensor) else outputs

    def task_loss_and_predictions(self, criterion, outputs, labels, problem_type):
        if problem_type == "classification":
            labels = labels.long()
            predictions = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)   # raw logits for CE loss
        else:
            labels = labels.float()
            predictions = outputs.squeeze()
            loss = criterion(predictions, labels)  # numeric values for regression loss

        return loss, labels, predictions

    def metrics_from_predictions(self, labels, predictions, problem_type, accuracy_tolerance):
        predictions = self.task_predictions(predictions, problem_type)
        return self.task_metrics(labels, predictions, problem_type, accuracy_tolerance)

    def evaluate_loader(self, loader, criterion, device, problem_type, accuracy_tolerance):
        total_loss = 0
        processed_batches = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                inputs, labels = self.unpack_batch(batch, device)
                if inputs.shape[0] < 2:
                    continue

                outputs = self.model(inputs)
                loss, labels, predictions = self.task_loss_and_predictions(
                    criterion,
                    outputs,
                    labels,
                    problem_type,
                )
                total_loss += loss.item()
                processed_batches += 1
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = self.task_metrics(all_labels, all_preds, problem_type, accuracy_tolerance)
        metrics["loss"] = total_loss / processed_batches if processed_batches else 0
        metrics["predictions"] = all_preds
        return metrics

    def compute_test_metrics(self, test_loader, criterion, device, problem_type, accuracy_tolerance):
        metrics = self.evaluate_loader(test_loader, criterion, device, problem_type, accuracy_tolerance)
        return metrics["mse"], metrics["accuracy"], metrics["loss"], (
            metrics if problem_type == "classification" else {}
        )

    @classmethod
    def task_metrics(cls, labels, predictions, problem_type, accuracy_tolerance):
        if problem_type == "classification":
            metrics = cls.get_classification_metrics(labels, predictions)
            metrics.update({"mae": 0, "mse": 0, "rmse": 0, "r2": 0, "mean": 0, "error_mean": 0})
            return metrics
        return cls.get_regression_metrics(labels, predictions, accuracy_tolerance)

    def print_final_classification_metrics(self, name, metrics):
        print(
            f"Client {self.model_id} | {name}: Accuracy: {metrics['accuracy']:.2f}, "
            f"Loss: {metrics['loss']:.4f}"
        )

    def print_final_regression_metrics(self, name, metrics):
        print(
            f"Client {self.model_id} | {name}: Accuracy: {metrics['accuracy']:.2f}, "
            f"Loss: {metrics['loss']:.4f}, MAE: {metrics['mae']:.4f}, "
            f"MSE: {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}"
        )
        print(f"Client {self.model_id} | R^2 Value is: {metrics['r2']:.4f}")
        print(
            f"Client {self.model_id} | RMSE for {name.lower()} set is: "
            f"{metrics['rmse']:.4f} & mean is {metrics['mean']:.4f}"
        )
        print(
            f"Client {self.model_id} | "
            f"This is {metrics['error_mean']:.2f}% of the mean pheno data"
        )

    @staticmethod
    def get_classification_metrics(labels, predictions):
        if len(labels) == 0:
            return {
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
            }

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision_macro": precision_score(
                labels, predictions, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                labels, predictions, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        }

    @staticmethod
    def format_classification_metrics(name, metrics):
        return (
            f"{name} Macro Precision: {metrics['precision_macro']:.2f}, "
            f"{name} Macro Recall: {metrics['recall_macro']:.2f}, "
            f"{name} Macro F1: {metrics['f1_macro']:.2f}"
        )

    @staticmethod
    def add_classification_report_metrics(metadata, train_metrics, test_metrics):
        metadata.update({
            "train precision macro": float(train_metrics["precision_macro"]),
            "train recall macro": float(train_metrics["recall_macro"]),
            "test precision macro": float(test_metrics["precision_macro"]),
            "test recall macro": float(test_metrics["recall_macro"]),
        })

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
    def add_epoch_report_metrics(
        metadata,
        train_acc,
        test_acc,
        train_mse,
        test_mse,
        losses,
        train_precision=None,
        test_precision=None,
        train_recall=None,
        test_recall=None,
        problem_type=None,
    ):
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
        if train_precision is not None:
            metadata.update({
                "train precision macro per epoch": np.array(train_precision),
                "test precision macro per epoch": np.array(test_precision),
                "train recall macro per epoch": np.array(train_recall),
                "test recall macro per epoch": np.array(test_recall),
            })

    @staticmethod
    def add_epoch_values(
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
        problem_type,
    ):
        train_mse.append(epoch_train_mse)
        test_mse.append(epoch_test_mse)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        if problem_type == "classification":
            train_precision.append(train_metrics["precision_macro"])
            test_precision.append(test_metrics["precision_macro"])
            train_recall.append(train_metrics["recall_macro"])
            test_recall.append(test_metrics["recall_macro"])

    @staticmethod
    def get_regression_metrics(labels, predictions, accuracy_tolerance):
        if len(labels) == 0:
            return {
                "accuracy": 0.0,
                "mae": 0.0,
                "mse": 0.0,
                "rmse": 0.0,
                "r2": 0.0,
                "mean": 0.0,
                "error_mean": 0.0,
            }

        labels = np.asarray(labels, dtype=float)
        predictions = np.asarray(predictions, dtype=float)
        mse = mean_squared_error(labels, predictions)
        rmse = mse**0.5
        mean = np.mean(labels)
        return {
            "accuracy": np.mean(np.abs(predictions - labels) <= accuracy_tolerance),
            "mae": mean_absolute_error(labels, predictions),
            "mse": mse,
            "rmse": rmse,
            "r2": r2_score(labels, predictions),
            "mean": mean,
            "error_mean": ((rmse / mean) * 100) if mean else 0,
        }

    @staticmethod
    def print_epoch_metrics(model_id, epoch, epochs, epoch_loss, train_accuracy, test_accuracy, problem_type, train_metrics, test_class_metrics,
                            mse, epoch_test_mse, epsilon_spent=None):
        message = (f"Client {model_id} | "
                   f"Epoch {epoch + 1}/{epochs}, "
                   f"{f'Loss: {epoch_loss:.4f}, ' if epoch_loss is not None else ''}"
                   f"Train Acc: {train_accuracy:.2f}, "
                   f"Test Acc: {test_accuracy:.2f}, ")
        if problem_type == "classification":
            message += (f"{BaseModel.format_classification_metrics('Train', train_metrics)}, "
                        f"{BaseModel.format_classification_metrics('Test', test_class_metrics)}")
        else:
            message += (f"Train MSE: {mse:.4f}, Test MSE: {epoch_test_mse:.4f}, "
                        f"MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
        if epsilon_spent is not None:
            message += f", ε: {epsilon_spent:.2f}"
        print(message)

####################################################################################################
#                                                                                                  #
#                                         Callable Classes                                         #
#                                                                                                  #
####################################################################################################

class CNNModel(BaseModel):
    model_extension = ".torch"

    def build_model(self, num_features, output_dim):
        self.model = _CNNNet(num_features, output_dim=output_dim)
        return self.model
    
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

    def fit(
        self,
        train_loader,
        test_loader,
        epochs,
        optimizer,
        criterion,
        device,
        problem_type,
        accuracy_tolerance,
    ):
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

        for epoch in range(epochs):
            epoch_loss = 0
            train_labels, train_preds = [], []
            for i, data in enumerate(train_loader):
                inputs, labels = self.unpack_batch(data, device)
                if inputs.shape[0] < 2:
                    continue
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss, labels, pred_classes = self.task_loss_and_predictions(
                    criterion,
                    outputs,
                    labels,
                    problem_type,
                )
                if problem_type == "classification":
                    outputs = pred_classes.float()
                else:
                    outputs = pred_classes
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_labels.extend(labels.cpu().numpy())
                train_preds.extend(outputs.detach().cpu().numpy())
            
            train_metrics = (
                self.get_classification_metrics(train_labels, train_preds)
                if problem_type == "classification"
                else self.get_regression_metrics(train_labels, train_preds, accuracy_tolerance)
            )
            train_accuracy = train_metrics["accuracy"]
            mse = train_metrics.get("mse", 0)

            self.model.eval()
            (
                epoch_test_mse,
                epoch_test_acc,
                _,
                test_class_metrics,
            ) = self.compute_test_metrics(
                test_loader,
                criterion,
                device,
                problem_type,
                accuracy_tolerance,
            )
            self.model.train()
            self.add_epoch_values(
                train_mse, test_mse, train_acc, test_acc,
                train_precision, test_precision, train_recall, test_recall,
                train_metrics, test_class_metrics, mse, epoch_test_mse,
                train_accuracy, epoch_test_acc, problem_type,
            )
            losses.append(epoch_loss)

            self.print_epoch_metrics(self.model_id, epoch, epochs, epoch_loss, train_accuracy,
                                     epoch_test_acc, problem_type, train_metrics,
                                     test_class_metrics, mse, epoch_test_mse)

        return train_mse, test_mse, train_acc, test_acc, train_precision, test_precision, train_recall, test_recall, losses

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

class DPCNNModel(BaseModel):
    model_extension = ".torch"

    def build_model(self, num_features, output_dim):
        self.model = _CNNNet(num_features, output_dim=output_dim)
        return self.model
    
    def build_optimizer(self, optimizer_name, learning_rate, weight_decay):
        if self.model is None:
            raise RuntimeError("build_model must be called before build_optimizer")

        if optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

        if optimizer_name == "adamax":
            return optim.Adamax(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

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

    def fit(
        self,
        train_loader,
        test_loader,
        epochs,
        optimizer,
        criterion,
        delta,
        privacy_engine,
        device,
        problem_type,
        accuracy_tolerance,
    ):
        self.model.train()
        train_mse = []
        test_mse = []
        train_acc = []
        test_acc = []
        train_precision = []
        test_precision = []
        train_recall = []
        test_recall = []
        eps = []
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            train_labels, train_preds = [], []
            for i, data in enumerate(train_loader):
                inputs, labels = self.unpack_batch(data, device)
                if inputs.shape[0] < 2:
                    continue
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss, labels, pred_classes = self.task_loss_and_predictions(
                    criterion,
                    outputs,
                    labels,
                    problem_type,
                )
                if problem_type == "classification":
                    outputs = pred_classes.float()
                else:
                    outputs = pred_classes
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_labels.extend(labels.cpu().numpy())
                train_preds.extend(outputs.detach().cpu().numpy())

            train_metrics = (
                self.get_classification_metrics(train_labels, train_preds)
                if problem_type == "classification"
                else self.get_regression_metrics(train_labels, train_preds, accuracy_tolerance)
            )
            train_accuracy = train_metrics["accuracy"]
            mse = train_metrics.get("mse", 0)

            # Compute test MSE per epoch
            self.model.eval()
            (
                epoch_test_mse,
                epoch_test_acc,
                _,
                test_class_metrics,
            ) = self.compute_test_metrics(
                test_loader,
                criterion,
                device,
                problem_type,
                accuracy_tolerance,
            )
            self.model.train()
            epsilon_spent, _ = privacy_engine.accountant.get_privacy_spent(
                delta=delta
            )
            self.add_epoch_values(
                train_mse, test_mse, train_acc, test_acc,
                train_precision, test_precision, train_recall, test_recall,
                train_metrics, test_class_metrics, mse, epoch_test_mse,
                train_accuracy, epoch_test_acc, problem_type,
            )
            losses.append(epoch_loss)
            eps.append(epsilon_spent)

            self.print_epoch_metrics(self.model_id, epoch, epochs, epoch_loss, train_accuracy,
                                     epoch_test_acc, problem_type, train_metrics,
                                     test_class_metrics, mse, epoch_test_mse, epsilon_spent)

        return train_mse, test_mse, train_acc, test_acc, train_precision, test_precision, train_recall, test_recall, losses, eps

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

class XGBoostModel(BaseModel):
    model_extension = ".ubj"
    def __init__(
        self,
        model_id,
        output_dir,
        params,
        problem_type,
        class_labels,
        accuracy_tolerance,
    ):
        super().__init__(model_id=model_id, output_dir=output_dir)
        self.params = params.copy()
        self.problem_type = problem_type
        self.class_labels = class_labels
        self.accuracy_tolerance = accuracy_tolerance

        self.train_loss_per_epoch = []
        self.test_loss_per_epoch = []
        self.final_train_loss = 0.0
        self.final_test_loss = 0.0

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

            self.loss_metric_name = self.eval_metric
        else:
            self.objective = self.params.get("objective")
            self.eval_metric = self.params.get("eval_metric", "rmse")
            self.loss_metric_name = "mse"

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

            train_acc, train_mse, train_mae, train_rmse = self.regression_metrics(train_data)
            test_acc, test_mse, _, _ = self.regression_metrics(test_data)
            train_metrics = {"mae": train_mae, "rmse": train_rmse}
            test_metrics = {}

            if self.problem_type == "classification":
                train_metrics = self.get_classification_metrics(
                    train_data.get_label(),
                    self.task_predictions(self.model.predict(train_data), self.problem_type),
                )
                test_metrics = self.get_classification_metrics(
                    test_data.get_label(),
                    self.task_predictions(self.model.predict(test_data), self.problem_type),
                )

                metric_key = next(iter(evals_result["train"]))
                self.final_train_loss = float(evals_result["train"][metric_key][-1])
                self.final_test_loss = float(evals_result["validate"][metric_key][-1])
                self.losses.append(self.final_train_loss)
                displayed_loss = self.final_train_loss
            else:
                self.final_train_loss = float(train_mse)
                self.final_test_loss = float(test_mse)
                self.losses.append(self.final_train_loss)
                displayed_loss = self.final_train_loss

            self.add_epoch_values(
                self.train_mse_per_epoch, self.test_mse_per_epoch,
                self.train_acc_per_epoch, self.test_acc_per_epoch,
                self.train_precision_per_epoch, self.test_precision_per_epoch,
                self.train_recall_per_epoch, self.test_recall_per_epoch,
                train_metrics, test_metrics, train_mse, test_mse,
                train_acc, test_acc, self.problem_type,
            )
            self.print_epoch_metrics(
                self.model_id, i, num_local_round,
                displayed_loss,
                train_acc, test_acc, self.problem_type,
                train_metrics, test_metrics, train_mse, test_mse
            )

        if train_method == "bagging":
            start = self.model.num_boosted_rounds() - num_local_round
            self.model = self.model[start:self.model.num_boosted_rounds()]

        return self.model
    
    def evaluate(self, test_data):
        if self.model is None:
            raise RuntimeError("build_model or set_parameters must be called before evaluate")

        return self._evaluate_loss(test_data, "valid")

    def regression_metrics(self, data):
        labels = data.get_label()
        predictions = self.task_predictions(self.model.predict(data), self.problem_type)
        metrics = self.task_metrics(
            labels,
            predictions,
            self.problem_type,
            self.accuracy_tolerance,
        )
        return metrics["accuracy"], metrics["mse"], metrics["mae"], metrics["rmse"]

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
        train_acc, _train_mse, _train_mae, _train_rmse = self.regression_metrics(train_data)
        test_acc, _test_mse, _test_mae, _test_rmse = self.regression_metrics(test_data)
        train_predictions = self.task_predictions(self.model.predict(train_data), self.problem_type)
        test_predictions = self.task_predictions(self.model.predict(test_data), self.problem_type)

        metadata = {
            "created on": str(datetime.now()),
            "model id": int(self.model_id),
            "round number": int(global_round),
            "partitions file": partitions_file,
        }
        self.add_task_report_metadata(
            metadata,
            self.problem_type,
            self.class_labels,
            self.accuracy_tolerance,
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
                "train mean squared error": float(_train_mse),
                "test mean squared error": float(_test_mse),
            })
        if self.problem_type == "classification":
            self.add_classification_report_metrics(
                metadata,
                self.get_classification_metrics(train_data.get_label(), train_predictions),
                self.get_classification_metrics(test_data.get_label(), test_predictions),
            )
        self.add_epoch_report_metrics(
            metadata,
            self.train_acc_per_epoch,
            self.test_acc_per_epoch,
            self.train_mse_per_epoch,
            self.test_mse_per_epoch,
            self.losses,
            train_precision=self.train_precision_per_epoch if self.problem_type == "classification" else None,
            test_precision=self.test_precision_per_epoch if self.problem_type == "classification" else None,
            train_recall=self.train_recall_per_epoch if self.problem_type == "classification" else None,
            test_recall=self.test_recall_per_epoch if self.problem_type == "classification" else None,
            problem_type=self.problem_type,
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
        return self.task_predictions(model.predict(data), self.problem_type)

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
