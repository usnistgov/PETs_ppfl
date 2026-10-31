# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software
# Tests added 2026-10-31 for batch-size validation of NIST testbed inputs.

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from utils import validate_data_size


def write_features(path, rows):
    features = np.zeros((rows, 4), dtype=np.float32)
    with path.open("wb") as stream:
        if path.suffix == ".dat":
            pickle.dump(features, stream)
        else:
            np.save(stream, features)


@pytest.mark.parametrize("extension", [".dat", ".npy"])
@pytest.mark.parametrize("suffix", ["_tt_vcf", "_ho_vcf"])
def test_rejects_batch_larger_than_feature_rows(tmp_path, extension, suffix):
    write_features(tmp_path / f"sample{suffix}{extension}", rows=4)

    with pytest.raises(ValueError, match=r"batch_size=5.*num_rows=4"):
        validate_data_size(tmp_path, batch_size=5)


@pytest.mark.parametrize("extension", [".dat", ".npy"])
@pytest.mark.parametrize("suffix", ["_tt_vcf", "_ho_vcf"])
def test_accepts_batch_equal_to_feature_rows_without_conversion(tmp_path, extension, suffix):
    path = tmp_path / f"sample{suffix}{extension}"
    write_features(path, rows=4)
    original = path.read_bytes()

    validate_data_size(tmp_path, batch_size=4)

    assert list(tmp_path.iterdir()) == [path]
    assert path.read_bytes() == original


@pytest.mark.parametrize("legacy_suffix", ["_tt_vcf", "_ho_vcf"])
def test_mixed_formats_still_check_the_legacy_array(tmp_path, legacy_suffix):
    for suffix in ["_tt_vcf", "_ho_vcf"]:
        extension = ".dat" if suffix == legacy_suffix else ".npy"
        rows = 3 if suffix == legacy_suffix else 8
        write_features(tmp_path / f"sample{suffix}{extension}", rows)

    with pytest.raises(ValueError, match=r"batch_size=4.*num_rows=3"):
        validate_data_size(tmp_path, batch_size=4)


@pytest.mark.parametrize("suffix", ["_tt_vcf", "_ho_vcf"])
def test_existing_npy_takes_precedence_over_legacy_input(tmp_path, suffix):
    write_features(tmp_path / f"sample{suffix}.npy", rows=4)
    # The loader uses the existing converted array, without opening this file.
    (tmp_path / f"sample{suffix}.dat").write_bytes(b"unused legacy input")

    validate_data_size(tmp_path, batch_size=4)


@pytest.mark.parametrize("extension", [".dat", ".npy"])
@pytest.mark.parametrize("suffix", ["_tt_vcf", "_ho_vcf"])
def test_ambiguous_feature_inputs_are_rejected(tmp_path, extension, suffix):
    for prefix in ["first", "second"]:
        write_features(tmp_path / f"{prefix}{suffix}{extension}", rows=4)

    with pytest.raises(ValueError, match="Multiple files found"):
        validate_data_size(tmp_path, batch_size=4)


@pytest.mark.parametrize("extension", [".dat", ".npy"])
@pytest.mark.parametrize("batch_size", [8, 9])
def test_check_only_applies_the_same_batch_limit(tmp_path, extension, batch_size):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for suffix in ["_tt_vcf", "_ho_vcf"]:
        write_features(data_dir / f"sample{suffix}{extension}", rows=8)
    before = {path.name: path.read_bytes() for path in data_dir.iterdir()}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "model_type": "cnn",
        "num_clients": 1,
        "num_partitions": 1,
        "data_dir": str(data_dir),
        "output_dir": str(tmp_path),
        "batch_size": batch_size,
        "problem_type": "regression",
    }))
    run_py = Path(__file__).resolve().parents[1] / "run.py"

    result = subprocess.run(
        [sys.executable, str(run_py), "--check_only", "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == (0 if batch_size == 8 else 1), result.stdout + result.stderr
    if batch_size == 8:
        assert "Parameters validated. Ending script" in result.stdout
    else:
        assert "batch_size=9" in result.stdout
        assert "num_rows=8" in result.stdout
    assert {path.name: path.read_bytes() for path in data_dir.iterdir()} == before
