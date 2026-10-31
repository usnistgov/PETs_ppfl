# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software

from collections import Counter

import numpy as np
import pytest

from client import create_dataloaders
from dataset import train_test_indices_split
from server import build_test_data


CLASS_LABELS = [
    pytest.param(["case", "control", "rare"], id="strings"),
    pytest.param([1.1, 1.9, 1.5], id="fractional-numbers"),
    pytest.param(["01", "1", "2"], id="numeric-strings"),
]


def classification_labels(classes):
    return np.asarray([classes[0]] * 8 + [classes[1]] * 16 + [classes[2]])


@pytest.mark.parametrize("classes", CLASS_LABELS)
def test_predefined_partition_split_preserves_class_identity(classes):
    labels = classification_labels(classes)
    indices = np.arange(len(labels))

    train, test = train_test_indices_split(
        labels.reshape(-1, 1), indices, 0.25, 42, "classification"
    )

    assert set(train).isdisjoint(test)
    assert sorted(train + test) == indices.tolist()
    assert 24 in train and 24 not in test
    assert Counter(labels[test]) == {classes[0]: 2, classes[1]: 4}


def write_synthetic_arrays(directory, classes):
    train_labels = classification_labels(classes)
    holdout_labels = np.asarray([classes[0], classes[1]] * 4)
    for split, labels in [("tt", train_labels), ("ho", holdout_labels),
                          ("pub", np.asarray([], dtype=train_labels.dtype))]:
        features = np.arange(len(labels) * 128, dtype=np.float32).reshape(-1, 128)
        np.save(directory / f"synthetic_{split}_vcf.npy", features)
        np.save(directory / f"synthetic_{split}_pheno.npy", labels)


@pytest.mark.parametrize("classes", CLASS_LABELS)
def test_client_loads_and_encodes_predefined_classification_partition(
    tmp_path, capsys, classes
):
    write_synthetic_arrays(tmp_path, classes)
    partition_path = tmp_path / "partitions.npz"
    np.savez(partition_path, client_0=np.arange(25))

    _, _, train_loader, test_loader, train, test = create_dataloaders(
        client_id=0,
        data_partitions_file=partition_path,
        data_directory=tmp_path,
        num_partitions=1,
        partitioner_type="uniform",
        test_fraction=0.25,
        seed=42,
        batch_size=8,
        problem_type="classification",
        class_labels=classes,
    )

    train_encoded = [int(label) for _, batch in train_loader for label in batch]
    test_encoded = [int(label) for _, batch in test_loader for label in batch]
    assert Counter(train_encoded) == {0: 6, 1: 12, 2: 1}
    assert Counter(test_encoded) == {0: 2, 1: 4}
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(25))
    output = capsys.readouterr().out
    assert f"class {classes[0]}: 6 records" in output
    assert f"class {classes[1]}: 12 records" in output
    assert f"class {classes[2]}: 1 records" in output


@pytest.mark.parametrize("classes", CLASS_LABELS)
@pytest.mark.parametrize("model_type", ["cnn", "xgboost"])
def test_server_holdout_preserves_raw_label_counts_and_encodes_once(
    tmp_path, capsys, classes, model_type
):
    write_synthetic_arrays(tmp_path, classes)

    num_features, holdout = build_test_data({
        "data_dir": tmp_path,
        "model_type": model_type,
        "problem_type": "classification",
        "class_labels": classes,
        "batch_size": 8,
    })

    if model_type == "xgboost":
        encoded = holdout.get_label().tolist()
    else:
        encoded = [int(label) for _, batch in holdout for label in batch]
    assert num_features == 128
    assert encoded == [0, 1] * 4
    output = capsys.readouterr().out
    assert f"class {classes[0]}: 4 records" in output
    assert f"class {classes[1]}: 4 records" in output


def test_integer_classification_split_keeps_existing_indices():
    labels = np.asarray([1] * 12 + [2] * 12 + [3])

    train, test = train_test_indices_split(
        labels.reshape(-1, 1), np.arange(25), 0.25, 42, "classification"
    )

    assert train == [9, 16, 13, 2, 5, 20, 15, 11, 8, 1, 0, 22, 18, 21, 10,
                     4, 17, 12, 24]
    assert test == [6, 23, 19, 14, 3, 7]


def test_regression_split_still_keeps_singleton_extreme_in_training():
    labels = np.asarray([0.0] * 12 + [10.0] * 12 + [100.0])

    train, test = train_test_indices_split(
        labels.reshape(-1, 1), np.arange(25), 0.25, 42, "regression"
    )

    assert 24 in train and 24 not in test
    assert Counter(labels[test]) == {0.0: 3, 10.0: 3}
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(25))
