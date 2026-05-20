from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json
from logging import INFO
import numpy as np
import xgboost as xgb
from xgboost.core import DMatrix
from sklearn.metrics import accuracy_score
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

from utils import save_xgb


def predict_labels(model, data):
    predictions = model.predict(data)
    if predictions.ndim == 2:
        return np.argmax(predictions, axis=1)
    if predictions.dtype.kind == "f" and predictions.min() >= 0 and predictions.max() <= 1:
        return np.rint(predictions)
    return np.rint(predictions)

def configure_objective_from_labels(params, labels):
    params = params.copy()
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    integer_labels = np.all(np.equal(labels, labels.astype(int)))
    nonnegative_labels = np.all(labels >= 0)

    if len(unique_labels) <= 2 and set(unique_labels.astype(int)) <= {0, 1}:
        params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})
    elif integer_labels and nonnegative_labels:
        params.update(
            {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "num_class": int(np.max(labels)) + 1,
            }
        )
    else:
        params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})

    return params

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
            bst = self._local_boost(bst)

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
        save_xgb(bst, metadata, f'xgb_{self.client_id}', global_round)

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
                metrics={"AUC": 0.0},
            )
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.test_data, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        global_round = ins.config["global_round"]
        log(INFO, f"AUC = {auc} at round {global_round}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=len(self.test_indices),
            metrics={"AUC": auc},
        )
