# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software
#
# This file was edited with the assistance of Claude Code (Anthropic, model
# Claude Opus 4.8). The assistant proposed and wrote unit tests for the server
# aggregation and history helpers in accordance with the author's instructions.
# All content has been reviewed and verified by the authors.

import pytest

import server
from model import CNNModel, DPCNNModel
from server import (
    get_torch_model_class,
    update_global_history,
    weighted_average_metrics,
)


# --------------------------------------------------------------------------- #
# weighted_average_metrics                                                    #
# --------------------------------------------------------------------------- #
def test_weighted_average_weights_by_example_count():
    result = weighted_average_metrics([(2, {"accuracy": 0.5}), (3, {"accuracy": 1.0})])
    # (2*0.5 + 3*1.0) / 5 = 0.8
    assert result["accuracy"] == pytest.approx(0.8)


def test_weighted_average_zero_total_returns_empty():
    # Entries with no examples are dropped, leaving nothing to average.
    assert weighted_average_metrics([(0, {"accuracy": 1.0})]) == {}


def test_weighted_average_drops_empty_metric_dicts():
    assert weighted_average_metrics([(5, {})]) == {}


def test_weighted_average_handles_missing_keys_as_zero():
    result = weighted_average_metrics([(2, {"a": 1.0}), (3, {"b": 1.0})])
    # a present only in first client (weight 2/5), b only in second (3/5).
    assert result["a"] == pytest.approx(0.4)
    assert result["b"] == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# get_torch_model_class                                                       #
# --------------------------------------------------------------------------- #
def test_get_torch_model_class_dispatch():
    assert get_torch_model_class("dpcnn") is DPCNNModel
    assert get_torch_model_class("cnn") is CNNModel
    assert get_torch_model_class("anything_else") is CNNModel


# --------------------------------------------------------------------------- #
# update_global_history (mutates module-level round history lists)            #
# --------------------------------------------------------------------------- #
_HISTORY_ATTRS = ["loss_rounds", "accuracy_rounds", "mse_rounds",
                  "precision_rounds", "recall_rounds"]


@pytest.fixture
def reset_history():
    """Snapshot, clear, and restore the module-level history lists."""
    saved = {name: list(getattr(server, name)) for name in _HISTORY_ATTRS}
    for name in _HISTORY_ATTRS:
        getattr(server, name).clear()
    yield
    for name in _HISTORY_ATTRS:
        lst = getattr(server, name)
        lst.clear()
        lst.extend(saved[name])


def test_update_global_history_regression(reset_history):
    update_global_history("regression", loss=1.0, accuracy=0.5, mse=2.0)
    assert server.loss_rounds == [1.0]
    assert server.accuracy_rounds == [0.5]
    assert server.mse_rounds == [2.0]
    # Classification-only histories stay untouched.
    assert server.precision_rounds == []
    assert server.recall_rounds == []


def test_update_global_history_classification(reset_history):
    update_global_history(
        "classification", loss=1.0, accuracy=0.5, mse=2.0,
        classification_metrics={"precision_macro": 0.3, "recall_macro": 0.4},
    )
    assert server.precision_rounds == [0.3]
    assert server.recall_rounds == [0.4]
