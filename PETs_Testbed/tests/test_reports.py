# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software
#
# This file was edited with the assistance of Claude Code (Anthropic, model
# Claude Opus 4.8). The assistant added characterization tests for the
# NumpyEncoder fallback path in accordance with the author's instructions. All
# content has been reviewed and verified by the authors.

import json

import numpy as np
import pytest

from report import Report


def test_report_requires_dictionary_input():
    with pytest.raises(TypeError, match="Input must be a dictionary object"):
        Report(["not", "a", "dict"])


def test_report_saves_plain_dictionary_as_json(tmp_path):
    report_path = tmp_path / "report.json"
    metadata = {
        "model id": 3,
        "train accuracy": 0.75,
        "hyperparameters": {"epochs": 5, "optimizer": "sgd"},
    }

    Report(metadata).save_to_file(report_path)

    assert json.loads(report_path.read_text(encoding="utf-8")) == metadata


def test_report_serializes_numpy_scalars_and_arrays(tmp_path):
    report_path = tmp_path / "numpy_report.json"
    metadata = {
        "model id": np.int64(7),
        "test mse": np.float64(1.25),
        "losses per epoch": np.array([3.0, 2.0, 1.0]),
        "confusion": np.array([[1, 2], [3, 4]]),
    }

    Report(metadata).save_to_file(report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))

    assert saved == {
        "model id": 7,
        "test mse": 1.25,
        "losses per epoch": [3.0, 2.0, 1.0],
        "confusion": [[1, 2], [3, 4]],
    }


def test_report_unserializable_value_becomes_null_without_raising(tmp_path):
    # Characterization: NumpyEncoder.default swallows the TypeError for an
    # unencodable object and returns None, so the value is written as JSON null
    # rather than propagating an error. Documents current behavior.
    report_path = tmp_path / "weird.json"

    Report({"weird": object()}).save_to_file(report_path)

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == {"weird": None}


def test_report_numpy_bool_is_not_encoded(tmp_path):
    # Characterization / known gap: np.bool_ is neither np.integer nor
    # np.floating, so it falls through to the error path and serializes as null
    # instead of a JSON boolean.
    report_path = tmp_path / "bool.json"

    Report({"flag": np.bool_(True)}).save_to_file(report_path)

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == {"flag": None}
