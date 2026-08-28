# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software
#
# This file was edited with the assistance of Claude Code (Anthropic, model
# Claude Opus 4.8). The assistant proposed and wrote unit tests for the client
# data-conversion helpers in accordance with the author's instructions. All
# content has been reviewed and verified by the authors.

import numpy as np

from client import empty_evaluate_res, loader_to_dmatrix


def test_empty_evaluate_res_shape():
    res = empty_evaluate_res()
    assert res.loss == 0.0
    assert res.num_examples == 0
    assert res.metrics == {"accuracy": 0.0, "mse": 0.0}


def test_loader_to_dmatrix_concatenates_batches():
    # A loader yields (features, labels) batches of differing sizes.
    loader = [
        (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([0.0, 1.0])),
        (np.array([[5.0, 6.0]], dtype=np.float32), np.array([1.0])),
    ]
    dmatrix = loader_to_dmatrix(loader)

    assert dmatrix.num_row() == 3
    assert dmatrix.num_col() == 2
    np.testing.assert_array_equal(dmatrix.get_label(), np.array([0.0, 1.0, 1.0], dtype=np.float32))


def test_loader_to_dmatrix_flattens_2d_labels():
    # Labels arriving as (n, 1) are reshaped to a flat vector.
    loader = [
        (np.array([[1.0], [2.0]], dtype=np.float32), np.array([[7.0], [8.0]], dtype=np.float32)),
    ]
    dmatrix = loader_to_dmatrix(loader)

    assert dmatrix.num_row() == 2
    np.testing.assert_array_equal(dmatrix.get_label(), np.array([7.0, 8.0], dtype=np.float32))
