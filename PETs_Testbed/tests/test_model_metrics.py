# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software

import numpy as np
import pytest

from model import BaseModel, XGBoostModel


class DummyData:
    def __init__(self, labels):
        self.labels = np.array(labels)

    def get_label(self):
        return self.labels


class DummyBooster:
    def __init__(self, predictions):
        self.predictions = np.array(predictions)

    def predict(self, data):
        return self.predictions


def test_metrics_from_predictions_supports_regression():
    model = BaseModel(model_id=0, output_dir=None, problem_type="regression")

    metrics = model.task.metrics(
        labels=np.array([1.0, 2.0, 3.0]),
        predictions=np.array([1.0, 2.2, 4.0]),
        accuracy_tolerance=0.25,
    )

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["mse"] == pytest.approx((0.0 + 0.04 + 1.0) / 3)
    assert metrics["mae"] == pytest.approx((0.0 + 0.2 + 1.0) / 3)


def test_metrics_from_predictions_supports_classification_probabilities():
    model = BaseModel(model_id=0, output_dir=None, problem_type="classification")

    metrics = model.task.metrics(
        labels=np.array([0, 1, 1, 0]),
        predictions=np.array([0, 1, 1, 0]),
        accuracy_tolerance=0.0,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision_macro"] == pytest.approx(1.0)
    assert metrics["recall_macro"] == pytest.approx(1.0)


def test_epoch_report_metrics_are_problem_specific():
    regression = {}
    BaseModel.add_epoch_report_metrics(
        regression,
        train_acc=[0.1],
        test_acc=[0.2],
        train_mse=[1.0],
        test_mse=[2.0],
        losses=[3.0],
        problem_type="regression",
    )

    assert "train mse per epoch" in regression
    assert "train precision macro per epoch" not in regression

    classification = {}
    BaseModel.add_epoch_report_metrics(
        classification,
        train_acc=[0.1],
        test_acc=[0.2],
        train_mse=[],
        test_mse=[],
        losses=[3.0],
        train_precision=[0.4],
        test_precision=[0.5],
        train_recall=[0.6],
        test_recall=[0.7],
        problem_type="classification",
    )

    assert "train mse per epoch" not in classification
    assert classification["train precision macro per epoch"].tolist() == [0.4]
    assert classification["test recall macro per epoch"].tolist() == [0.7]


def test_xgboost_data_metrics_uses_shared_regression_metrics():
    model = XGBoostModel(
        model_id=0,
        output_dir=None,
        params={},
        problem_type="regression",
        class_labels=None,
        accuracy_tolerance=0.6,
    )
    model.model = DummyBooster([1.0, 2.5])

    metrics, _ = model.dataset_metrics(DummyData([1.0, 2.0]))

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["mse"] == pytest.approx(0.125)


def test_xgboost_data_metrics_uses_shared_classification_metrics():
    model = XGBoostModel(
        model_id=0,
        output_dir=None,
        params={},
        problem_type="classification",
        class_labels=[0, 1],
        accuracy_tolerance=0.0,
    )
    model.model = DummyBooster([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])

    metrics, _ = model.dataset_metrics(DummyData([0, 1, 0]))

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision_macro"] == pytest.approx(1.0)
    assert metrics["recall_macro"] == pytest.approx(1.0)
