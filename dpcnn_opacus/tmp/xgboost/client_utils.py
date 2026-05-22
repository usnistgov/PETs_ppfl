from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json
from logging import INFO
import numpy as np
import xgboost as xgb
from xgboost.core import DMatrix
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import flwr as fl
from flwr.common.logger import log
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
from report import Report

def predict_labels(model, data):
    predictions = model.predict(data)
    if predictions.ndim == 2:
        return np.argmax(predictions, axis=1)
    if predictions.dtype.kind == "f" and predictions.min() >= 0 and predictions.max() <= 1:
        return np.rint(predictions)
    return np.rint(predictions)


def save_xgb_output(model, metadata, name, round_number, output_dir):
    output_path = (
        Path(output_dir).absolute()
        if output_dir is not None
        else Path(__file__).parent
    )
    output_path.mkdir(parents=True, exist_ok=True)

    out_name = (
        f"{name}_round_{round_number}" if round_number is not None else name
    )
    metadata_path = output_path / f"{out_name}_meta.npz"
    report_path = output_path / f"{out_name}.json"

    np.savez(metadata_path, **metadata, allow_pickle=True)
    report = Report(metadata)
    report.save_to_file(report_path)
    print(f"Model metadata saved to {metadata_path}")


def regression_metrics(model, data):
    labels = data.get_label()
    predictions = model.predict(data)
    rounded_predictions = np.rint(predictions)
    accuracy = accuracy_score(labels, rounded_predictions)
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = mse**0.5
    return accuracy, mse, mae, rmse


def print_epoch_metrics(model_id, epoch, epochs, model, train_data, test_data):
    train_acc, train_mse, train_mae, train_rmse = regression_metrics(
        model, train_data
    )
    test_acc, test_mse, _, _ = regression_metrics(model, test_data)
    print(
        f"Model {model_id} | "
        f"Epoch {epoch}/{epochs}, Loss: {train_mse:.4f}, "
        f"Train Acc: {train_acc:.2f}, "
        f"Test Acc: {test_acc:.2f}, "
        f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}, "
        f"MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}"
    )


class EpochLogger(xgb.callback.TrainingCallback):
    def __init__(self, model_id, epochs, train_data, test_data):
        self.model_id = model_id
        self.epochs = epochs
        self.train_data = train_data
        self.test_data = test_data

    def after_iteration(self, model, epoch, evals_log):
        print_epoch_metrics(
            self.model_id,
            epoch + 1,
            self.epochs,
            model,
            self.train_data,
            self.test_data,
        )
        return False


class XgbClient(fl.client.Client):
    def __init__(
        self,
        client_id: int,
        train_data: DMatrix,
        test_data: DMatrix,
        train_indices: List[int],
        test_indices: List[int],
        num_local_round: int,
        params: Dict[str, Any],
        train_method: str,
        data_partitions_file: Optional[Path] = None,
        test_data_fraction: float = 0.2,
        output_dir: Optional[Path] = None,
    ):
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.num_local_round = num_local_round
        self.params = params
        self.train_method = train_method
        self.data_partitions_file = data_partitions_file
        self.test_data_fraction = test_data_fraction
        self.output_dir = output_dir

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_data, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees
        # for server aggregation
        # Cyclic: return the entire model
        bst = (
            bst_input[
                bst_input.num_boosted_rounds()
                - self.num_local_round: bst_input.num_boosted_rounds()
            ]
            if self.train_method == "bagging"
            else bst_input
        )

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_data,
                num_boost_round=self.num_local_round,
                evals=[
                    (self.test_data, "validate"),
                    (self.train_data, "train"),
                ],
                verbose_eval=False,
                callbacks=[
                    EpochLogger(
                        self.client_id,
                        self.num_local_round,
                        self.train_data,
                        self.test_data,
                    )
                ],
            )
        else:
            bst = xgb.Booster(params=self.params)
            if not ins.parameters.tensors:
                return self.fit(
                    FitIns(
                        parameters=Parameters(tensor_type="", tensors=[]),
                        config={"global_round": "1"},
                    )
                )
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            for i in range(self.num_local_round):
                bst.update(self.train_data, bst.num_boosted_rounds())
                print_epoch_metrics(
                    self.client_id,
                    i + 1,
                    self.num_local_round,
                    bst,
                    self.train_data,
                    self.test_data,
                )

            bst = (
                bst[
                    bst.num_boosted_rounds()
                    - self.num_local_round: bst.num_boosted_rounds()
                ]
                if self.train_method == "bagging"
                else bst
            )

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        # Use model to find train and test data accuracy
        predictions = predict_labels(bst, self.train_data)
        train_acc = accuracy_score(self.train_data.get_label(), predictions)

        predictions = predict_labels(bst, self.test_data)
        test_acc = accuracy_score(self.test_data.get_label(), predictions)

        print(
            f"client {self.client_id} Train accuracy: {train_acc * 100:.2f}%"
        )
        print(f"client {self.client_id} Test accuracy: {test_acc * 100:.2f}%")

        partitions_path = (
            Path(self.data_partitions_file).name
            if self.data_partitions_file
            else 'none'
        )
        metadata = {
            'created on': str(datetime.now()),
            'model id': int(self.client_id),
            'partitions file': partitions_path,
            'train accuracy': float(train_acc),
            'test accuracy': float(test_acc),
            'train loss': 0,
            'test loss': 0,
            'train indices': self.train_indices,
            'test indices': self.test_indices,
            'hyperparameters': {
                **self.params,
                "seed": self.params['random_state'],
                "test_fraction": self.test_data_fraction,
            },
        }

        metadata['hyperparameters'] = json.dumps(metadata['hyperparameters'])
        save_xgb_output(
            bst,
            metadata,
            f'xgb_{self.client_id}',
            global_round,
            self.output_dir,
        )

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=len(self.test_indices),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        if not ins.parameters.tensors:
            return EvaluateRes(
                status=Status(
                    code=Code.OK,
                    message="No model parameters available for evaluation.",
                ),
                loss=0.0,
                num_examples=0,
                metrics={self.params.get("eval_metric", "metric"): 0.0},
            )
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.test_data, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        metric_name, metric_value = eval_results.split("\t")[1].split(":")
        metric_name = metric_name.split("-")[-1]
        metric_value = round(float(metric_value), 4)

        global_round = ins.config["global_round"]
        log(INFO, f"{metric_name} = {metric_value} at round {global_round}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=len(self.test_indices),
            metrics={metric_name: metric_value},
        )
