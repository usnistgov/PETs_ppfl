# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software
#
# This file was edited with the assistance of Claude Code (Anthropic, model
# Claude Opus 4.8). The assistant proposed pytest fixtures for synthetic,
# file-backed test data in accordance with the author's instructions. All
# content has been reviewed and verified by the authors.

import os

# On macOS, torch and xgboost each ship their own libomp; importing torch before
# xgboost (as client.py does) can double-load OpenMP and segfault the moment
# xgboost enters a parallel region (e.g. DMatrix construction). Pinning the pool
# to a single thread sidesteps that parallel region. This must run before any
# test module imports torch or xgboost, which is why it lives at conftest import
# time. It only affects the test process, not production code.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pickle  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run t46 end-to-end smoke test. Skipped by default.",
    )


# All feature/label file suffixes the loaders expect to discover in a data dir.
# Mirrors dataset.NPY_SUFFIXES, kept local so the fixtures do not import the
# module under test.
_FEATURE_SUFFIXES = ("tt_vcf", "ho_vcf", "pub_vcf")
_LABEL_SUFFIXES = ("tt_pheno", "ho_pheno", "pub_pheno")


@pytest.fixture
def synthetic_split():
    """Return two small, distinct feature/label arrays for concatenation tests.

    The two arrays have different lengths (3 and 2 rows) so that logical row
    indices exercise the cross-array boundary in IndexedArrayDataset.
    """
    features_a = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    features_b = np.array([[6.0, 7.0], [8.0, 9.0]], dtype=np.float32)
    labels_a = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    labels_b = np.array([13.0, 14.0], dtype=np.float32)
    return {
        "features_list": [features_a, features_b],
        "labels_list": [labels_a, labels_b],
        # Global row -> label, for assertions.
        "expected_labels": [10.0, 11.0, 12.0, 13.0, 14.0],
        # Global row -> first feature value, for assertions.
        "expected_feature0": [0.0, 2.0, 4.0, 6.0, 8.0],
    }


@pytest.fixture
def npy_data_dir(tmp_path):
    """Create a data directory populated with all required *.npy files.

    Feature arrays have two columns; label arrays are stored 2-D (n, 1) to
    mirror the on-disk phenotype layout. Rows are deterministic. Returns a dict
    with the directory path and the arrays that were written.
    """
    prefix = "T"
    features = {}
    labels = {}
    rng = np.random.default_rng(0)

    for suffix in _FEATURE_SUFFIXES:
        arr = rng.random((6, 3), dtype=np.float64).astype(np.float32)
        np.save(tmp_path / f"{prefix}_{suffix}.npy", arr)
        features[suffix] = arr

    for suffix in _LABEL_SUFFIXES:
        # Two classes / numeric labels, stored 2-D to match real phenotype files.
        arr = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
        np.save(tmp_path / f"{prefix}_{suffix}.npy", arr)
        labels[suffix] = arr

    return {"dir": tmp_path, "features": features, "labels": labels}


@pytest.fixture
def dat_data_dir(tmp_path):
    """Create a directory of pickled .dat NumPy arrays for conversion tests."""
    float64_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    with (tmp_path / "S_tt_vcf.dat").open("wb") as fh:
        pickle.dump(float64_arr, fh)

    int_arr = np.array([[1], [0]], dtype=np.int64)
    with (tmp_path / "S_tt_pheno.dat").open("wb") as fh:
        pickle.dump(int_arr, fh)

    # A non-array pickle that must be skipped rather than converted.
    with (tmp_path / "S_notes.dat").open("wb") as fh:
        pickle.dump({"not": "an array"}, fh)

    return {"dir": tmp_path, "float64_arr": float64_arr, "int_arr": int_arr}
