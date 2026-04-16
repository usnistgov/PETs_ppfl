# test_cli_config_regression.py
import copy
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import pytest


RUN_PY = Path(
    os.environ.get("RUN_PY", Path(__file__).resolve().parents[0] / "run.py")
).resolve()


COMMON_BASE_CONFIG: Dict[str, Any] = {
    "model_type": "dpcnn",
    "num_cpus": 2,
    "num_gpus": 0,
    "num_rounds": 1,
    "min_fit_clients": 1,
    "min_available_clients": 1,
    "min_evaluate_clients": 1,
    "data_partitions_file": "",
    "partitioner_type": "uniform",
    "num_partitions": 1,
    "partition_id": 0,
    "client_id": 0,
    "seed": 1,
    "epochs": 1,
    "batch_divisor": 1,
    "n_models": 1,
    "test_fraction": 0.2,
    "opacus_secure_mode": False,
    "epsilon": 1.0,
    "delta": 0.0,
    "max_grad_norm": 1.0,
    "output_dir": "../reports",
    "data_dir": "../data/Oil_binned5",
}

BASE_CONFIG: Dict[str, Any] = {
    **COMMON_BASE_CONFIG,
    "model_type": "dpcnn",
    "learning_rate": 0.01,
    "weight_decay": 0.001,
    "optimizer": "sgd",
    "accuracy_tolerance": 0.0,
}

CNN_BASE_CONFIG: Dict[str, Any] = {
    **COMMON_BASE_CONFIG,
    "model_type": "cnn",
    "learning_rate": 0.01,
    "weight_decay": 0.001,
    "optimizer": "sgd",
}

XGBOOST_BASE_CONFIG: Dict[str, Any] = {
    **COMMON_BASE_CONFIG,
    "model_type": "xgboost",
    "train_method": "bagging",
    "centralised_eval": True,
    "scaled_lr": True,
}


@dataclass(frozen=True)
class Case:
    name: str
    config: Optional[Dict[str, Any]] = None
    raw_config_text: Optional[str] = None
    create_config_file: bool = True
    config_filename: str = "config.json"
    cli_args: Sequence[str] = field(default_factory=list)
    allowed_exit_codes: Set[int] = field(default_factory=lambda: {0})
    stdout_must_match: Sequence[str] = field(default_factory=list)
    stdout_must_not_match: Sequence[str] = field(default_factory=list)


def _write_config(tmp_path: Path, case: Case) -> Path:
    cfg_path = tmp_path / case.config_filename
    if case.raw_config_text is not None:
        cfg_path.write_text(case.raw_config_text, encoding="utf-8")
        return cfg_path
    assert case.config is not None, "Case.config is None but raw_config_text is also None"
    cfg_path.write_text(json.dumps(case.config, indent=2), encoding="utf-8")
    return cfg_path


def _run(cmd: List[str], capture: bool = True) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True)
    return subprocess.run(cmd, text=True)


def _assert_patterns(text: str, must: Sequence[str], must_not: Sequence[str], stream_name: str) -> None:
    for pat in must:
        assert re.search(pat, text, re.IGNORECASE | re.MULTILINE), (
            f"Expected {stream_name} to match /{pat}/\n"
            f"--- {stream_name} ---\n{text}\n"
            f"----------------------\n"
        )
    for pat in must_not:
        assert not re.search(pat, text, re.IGNORECASE | re.MULTILINE), (
            f"Expected {stream_name} to NOT match /{pat}/\n"
            f"--- {stream_name} ---\n{text}\n"
            f"----------------------\n"
        )


def _base_with(**overrides: Any) -> Dict[str, Any]:
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update(overrides)
    return cfg


def _cnn_base_with(**overrides: Any) -> Dict[str, Any]:
    cfg = copy.deepcopy(CNN_BASE_CONFIG)
    cfg.update(overrides)
    return cfg


def _xgb_base_with(**overrides: Any) -> Dict[str, Any]:
    cfg = copy.deepcopy(XGBOOST_BASE_CONFIG)
    cfg.update(overrides)
    return cfg


def _base_without(*keys: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(BASE_CONFIG)
    for key in keys:
        cfg.pop(key, None)
    return cfg


CASES: List[Any] = [
    # A. Config/CLI precedence & parsing
    pytest.param(
        Case(
            name="1. Config-only load succeeds",
            config=_base_with(),
            cli_args=[],
            allowed_exit_codes={0},
        ),
        id="t01",
    ),
    pytest.param(
        Case(
            name="2. CLI overrides config (single field)",
            config=_base_with(num_rounds=10),
            cli_args=["--num_rounds", "11"],
            allowed_exit_codes={0},
        ),
        id="t02",
    ),
    pytest.param(
        Case(
            name="3. CLI overrides config (multiple fields)",
            config=_base_with(epochs=2, optimizer="sgd", opacus_secure_mode=False),
            cli_args=["--epochs", "3", "--optimizer", "adamax", "--opacus_secure_mode", "True"],
            allowed_exit_codes={0},
        ),
        id="t03",
    ),
    pytest.param(
        Case(
            name="4. Repeated CLI flags precedence",
            config=_base_with(),
            cli_args=["--seed", "1", "--seed", "2"],
            allowed_exit_codes={0},
        ),
        id="t04"
    ),
    pytest.param(
        Case(
            name="5. Unknown CLI argument",
            config=_base_with(),
            cli_args=["--unknown_flag", "1"],
            allowed_exit_codes={2, 1},
            stdout_must_match=[r"unknown|unrecognized|no such option|invalid option"],
        ),
        id="t05",
    ),
    pytest.param(
        Case(
            name="6. Unknown config key",
            config={**_base_with(), "unknown_key": 123},
            cli_args=[],
            allowed_exit_codes={1},
            stdout_must_match=[r"unknown_key|unknown|unexpected|additional"],
        ),
        id="t06"
    ),
    pytest.param(
        Case(
            name="7. Missing config file path",
            config=None,
            create_config_file=False,
            config_filename="does_not_exist.json",
            cli_args=[],
            allowed_exit_codes={2, 1},
            stdout_must_match=[r"not found|no such file|cannot open|does_not_exist|not find"],
        ),
        id="t07",
    ),
    pytest.param(
        Case(
            name="8. Invalid config syntax",
            raw_config_text='{ "num_rounds": 1, }',
            config=None,
            cli_args=[],
            allowed_exit_codes={2, 1},
            stdout_must_match=[r"error|parsing|config"],
        ),
        id="t08",
    ),
    pytest.param(
        Case(
            name="9. Empty config file",
            raw_config_text="{}",
            config=None,
            cli_args=[],
            allowed_exit_codes={1},
            stdout_must_match=[r"missing|required|model_type|default"],
        ),
        id="t09"
    ),
    pytest.param(
        Case(
            name="10. Type coercion rules (CLI)",
            config=_base_with(),
            cli_args=["--opacus_secure_mode", "true", "--learning_rate", "1e-3"],
            allowed_exit_codes={0, 2, 1},
        ),
        id="t10"
    ),
    pytest.param(
        Case(
            name="11. Whitespace/quoting handling for paths",
            config=_base_with(data_partitions_file=""),
            cli_args=["--data_partitions_file", "a b/parts.json"],
            allowed_exit_codes={1},
        ),
        id="t11"
    ),

    # B. Per-parameter validation examples
    pytest.param(Case("12a. num_rounds accepts min", _base_with(num_rounds=1)), id="t12a"),
    pytest.param(Case("12b. num_rounds accepts max", _base_with(num_rounds=100)), id="t12b"),
    pytest.param(Case("12c. num_rounds rejects below min", _base_with(num_rounds=0), allowed_exit_codes={1, 2}), id="t12c"),
    pytest.param(Case("12d. num_rounds rejects above max", _base_with(num_rounds=101), allowed_exit_codes={1, 2}), id="t12d"),
    pytest.param(Case("12e. num_rounds rejects wrong type", _base_with(num_rounds=1.5), allowed_exit_codes={1, 2}), id="t12e"),

    pytest.param(Case("13a. min_fit_clients boundary valid", _base_with(min_fit_clients=1)), id="t13a"),
    pytest.param(Case("13b. min_fit_clients boundary invalid", _base_with(min_fit_clients=0), allowed_exit_codes={1, 2}), id="t13b"),

    pytest.param(Case("14a. min_available_clients boundary valid", _base_with(min_available_clients=100)), id="t14a"),
    pytest.param(Case("14b. min_available_clients boundary invalid", _base_with(min_available_clients=101), allowed_exit_codes={1, 2}), id="t14b"),

    pytest.param(Case("15a. min_evaluate_clients boundary valid", _base_with(min_evaluate_clients=1)), id="t15a"),
    pytest.param(Case("15b. min_evaluate_clients boundary invalid", _base_with(min_evaluate_clients=0), allowed_exit_codes={1, 2}), id="t15b"),

    pytest.param(Case("16a. num_partitions boundary valid", _base_with(num_partitions=100)), id="t16a"),
    pytest.param(Case("16b. num_partitions boundary invalid", _base_with(num_partitions=0), allowed_exit_codes={1, 2}), id="t16b"),

    pytest.param(Case("17a. partition_id boundary valid", _base_with(partition_id=0)), id="t17a"),
    pytest.param(Case("17b. partition_id boundary invalid (below)", _base_with(partition_id=-1), allowed_exit_codes={1, 2}), id="t17b"),

    pytest.param(Case("18a. client_id boundary valid", _base_with(client_id=100)), id="t18a"),
    pytest.param(Case("18b. client_id boundary invalid (above)", _base_with(client_id=101), allowed_exit_codes={1, 2}), id="t18b"),

    pytest.param(Case("19a. seed boundary valid", _base_with(seed=1000)), id="t19a"),
    pytest.param(Case("19b. seed boundary invalid", _base_with(seed=0), allowed_exit_codes={1, 2}), id="t19b"),

    pytest.param(Case("20a. epochs boundary valid", _base_with(epochs=100)), id="t20a"),
    pytest.param(Case("20b. epochs boundary invalid", _base_with(epochs=101), allowed_exit_codes={1, 2}), id="t20b"),

    pytest.param(Case("21a. batch_divisor min valid", _base_with(batch_divisor=1)), id="t21a"),
    pytest.param(Case("21b. batch_divisor invalid (0)", _base_with(batch_divisor=0), allowed_exit_codes={1, 2}), id="t21b"),

    pytest.param(Case("22a. n_models boundary valid", _base_with(n_models=100)), id="t22a"),
    pytest.param(Case("22b. n_models boundary invalid", _base_with(n_models=0), allowed_exit_codes={1, 2}), id="t22b"),

    pytest.param(Case("23a. test_fraction valid interior", _base_with(test_fraction=0.9999)), id="t23a"),
    pytest.param(Case("23b. test_fraction invalid at 0", _base_with(test_fraction=0), allowed_exit_codes={1, 2}), id="t23b"),
    pytest.param(Case("23c. test_fraction invalid at 1", _base_with(test_fraction=1), allowed_exit_codes={1, 2}), id="t23c"),

    pytest.param(Case("24a. learning_rate valid interior", _base_with(learning_rate=0.5)), id="t24a"),
    pytest.param(Case("24b. learning_rate invalid at 0", _base_with(learning_rate=0), allowed_exit_codes={1, 2}), id="t24b"),

    pytest.param(Case("25a. weight_decay valid interior", _base_with(weight_decay=0.09999)), id="t25a"),
    pytest.param(Case("25b. weight_decay invalid at 0.1", _base_with(weight_decay=0.1), allowed_exit_codes={1, 2}), id="t25b"),

    pytest.param(Case("26a. epsilon valid interior", _base_with(epsilon=49.9999)), id="t26a"),
    pytest.param(Case("26b. epsilon invalid at 50", _base_with(epsilon=50), allowed_exit_codes={1, 2}), id="t26b"),

    pytest.param(Case("27a. delta valid boundaries", _base_with(delta=1)), id="t27a"),
    pytest.param(Case("27b. delta invalid (>1)", _base_with(delta=1.1), allowed_exit_codes={1, 2}), id="t27b"),

    pytest.param(Case("28a. max_grad_norm valid boundaries", _base_with(max_grad_norm=0)), id="t28a"),
    pytest.param(Case("28b. max_grad_norm invalid (negative)", _base_with(max_grad_norm=-0.01), allowed_exit_codes={1, 2}), id="t28b"),

    pytest.param(Case("29a. accuracy_tolerance valid boundaries", _base_with(accuracy_tolerance=1)), id="t29a"),
    pytest.param(Case("29b. accuracy_tolerance invalid (>1)", _base_with(accuracy_tolerance=2), allowed_exit_codes={1, 2}), id="t29b"),

    pytest.param(Case("30a. partitioner_type accepts allowed value", _base_with(partitioner_type="exponential")), id="t30a"),
    pytest.param(Case("30b. partitioner_type rejects bad enum", _base_with(partitioner_type="log"), allowed_exit_codes={1, 2}), id="t30b"),

    pytest.param(Case("31a. optimizer accepts allowed value", _base_with(optimizer="adamax")), id="t31a"),
    pytest.param(Case("31b. optimizer rejects bad enum", _base_with(optimizer="adam"), allowed_exit_codes={1, 2}), id="t31b"),

    pytest.param(Case("52a. lower one bad value", _base_with(min_available_clients=2)), id="t52a"),
    pytest.param(Case("52b. lower two bad values", _base_with(min_evaluate_clients=2, min_fit_clients=2)), id="t52b"),
    pytest.param(Case("52c. reject too high num clients", _base_with(min_evaluate_clients=2, min_fit_clients=2)), id="t52c"),
    pytest.param(Case("52d. lower one value and reject too high num clients", _base_with(min_available_clients=2, min_evaluate_clients=3, min_fit_clients=2, num_partitions=1), allowed_exit_codes={1, 2}), id="t52d"),
    pytest.param(Case("52e. lower two values and reject too high num clients", _base_with(min_available_clients=2, min_evaluate_clients=3, min_fit_clients=3, num_partitions=1), allowed_exit_codes={1, 2}), id="t52e"),
    pytest.param(Case("51a. num_cpus invalid at 0", _base_with(num_cpus=0), allowed_exit_codes={1, 2}), id="t51a"),
    pytest.param(Case("51b. num_gpus valid at 1", _base_with(num_gpus=1)), id="t51b"),

    pytest.param(Case("53. bad data directory path", _base_with(data_dir="foo_bar"), allowed_exit_codes={1, 2}), id="t53"),

    pytest.param(
        Case(
            "32a. data_partitions_file accepts string",
            _base_with(data_partitions_file="parts.json"),
            allowed_exit_codes={1},
        ),
        id="t32a"
    ),
    pytest.param(Case("32b. data_partitions_file rejects non-string", _base_with(data_partitions_file=123), allowed_exit_codes={1, 2}), id="t32b"),

    pytest.param(Case("33a. opacus_secure_mode accepts boolean", _base_with(opacus_secure_mode=True)), id="t33a"),
    pytest.param(Case("33b. opacus_secure_mode rejects non-boolean", _base_with(opacus_secure_mode="yes"), allowed_exit_codes={1, 2}), id="t33b"),

    # C. Cross-parameter interactions
    pytest.param(
        Case(
            "34. num_partitions lowers min_*_clients",
            _base_with(num_partitions=5, min_fit_clients=10, min_available_clients=8, min_evaluate_clients=7),
            allowed_exit_codes={1},
            stdout_must_match=[r"<=|num_partitions"],
        ),
        id="t34"
    ),
    pytest.param(
        Case(
            "35. partition_id must be < num_partitions (if enforced)",
            _base_with(num_partitions=10, partition_id=11),
            allowed_exit_codes={1, 2},
            stdout_must_match=[r"<=|num_partitions"]
        ),
        id="t35"
    ),
    pytest.param(
        Case(
            "36. client_id uniqueness in local sim (if enforced)",
            _base_with(n_models=2, client_id=1),
            cli_args=["--client_id", "1"],
            allowed_exit_codes={0},
        ),
        id="t36"
    ),
    pytest.param(
        Case(
            "37. data_partitions_file provided; partitioner_type invalid",
            _base_with(data_partitions_file="parts.json", partitioner_type="log"),
            allowed_exit_codes={0, 1, 2},
        ),
        id="t37",
        marks=pytest.mark.skip(reason="No data partitions file to test with yet"),
    ),
    pytest.param(
        Case(
            "38. Missing partition info",
            _base_without("data_partitions_file", "partitioner_type"),
            allowed_exit_codes={0},
        ),
        id="t38"
    ),
    pytest.param(
        Case(
            "39. batch_divisor vs dataset_size runtime behavior",
            _base_with(batch_divisor=1_000_000),
            allowed_exit_codes={1, 2},
        ),
        id="t39"
    ),
    pytest.param(
        Case(
            "40. DP gating: secure_mode true requires DP params (if required)",
            _base_with(opacus_secure_mode=True, epsilon=None),
            allowed_exit_codes={1, 2},
        ),
        id="t40"
    ),
    pytest.param(
        Case(
            "41. Optimizer selection sanity",
            _base_with(optimizer="adamax"),
            allowed_exit_codes={0},
        ),
        id="t41",
    ),

    # D. Error-message regression
    pytest.param(
        Case(
            "42. Error messages include parameter + expected + received",
            _base_with(learning_rate=2),
            allowed_exit_codes={1, 2},
            stdout_must_match=[r"learning_rate", r"1|one|range|exclusive|between|<|>"],
        ),
        id="t42",
    ),
    pytest.param(
        Case(
            "43. Non-zero exit on validation failure",
            _base_with(epochs=0),
            allowed_exit_codes={1, 2},
        ),
        id="t43",
    ),
    pytest.param(
        Case(
            "44. Multiple invalid params reported (aggregation)",
            _base_with(epochs=0, learning_rate=0, optimizer="adam"),
            allowed_exit_codes={1, 2},
            stdout_must_match=[r"epochs", r"learning_rate", r"optimizer"],
        ),
        id="t44"
    ),

    # E. End-to-end scenarios
    pytest.param(
        Case(
            "45. Determinism via seed (run twice)",
            _base_with(seed=123),
            allowed_exit_codes={0},
        ),
        id="t45",
        marks=pytest.mark.skip(reason="Needs a defined deterministic artifact/output to compare across runs"),
    ),
    pytest.param(
        Case("46. Minimal valid run (E2E smoke)", _base_with(num_cpus=12, delta=1e-5), allowed_exit_codes={0}),
        id="t46"
    ),
    pytest.param(
        Case(
            "47. Maximal valid bounds (validation)",
            _base_with(
                num_cpus=100,
                num_gpus=100,
                num_rounds=100,
                min_fit_clients=100,
                min_available_clients=100,
                min_evaluate_clients=100,
                num_partitions=100,
                partition_id=100,
                client_id=100,
                seed=1000,
                epochs=100,
                batch_divisor=1,
                n_models=100,
                test_fraction=0.9999,
                learning_rate=0.9999,
                weight_decay=0.09999,
                optimizer="adamax",
                opacus_secure_mode=False,
                epsilon=49.999,
                delta=1,
                max_grad_norm=100,
                accuracy_tolerance=1,
            ),
            allowed_exit_codes={0},
        ),
        id="t47"
    ),
    pytest.param(
        Case(
            "48. Typical FL config (representative)",
            _base_with(
                num_rounds=10,
                min_fit_clients=5,
                min_available_clients=5,
                min_evaluate_clients=5,
                num_partitions=20,
                epochs=3,
                batch_divisor=32,
                n_models=1,
                test_fraction=0.2,
                learning_rate=0.01,
                weight_decay=0.0005,
                optimizer="sgd",
            ),
            allowed_exit_codes={0},
        ),
        id="t48"
    ),
    pytest.param(
        Case(
            "49. With partitions file",
            _base_with(data_partitions_file="./fixtures/partitions.json"),
            allowed_exit_codes={0, 1, 2},
        ),
        id="t49",
        marks=pytest.mark.skip(reason="Requires a real partitions file"),
    ),
    pytest.param(
        Case(
            "50. Without partitions file; partitioner type uniform",
            _base_without("data_partitions_file"),
            allowed_exit_codes={0},
        ),
        id="t50_uniform",
        marks=pytest.mark.skip(reason="Requires a real partitions file"),
    ),

    # F. [NEW] Regressions added for schema changes
    pytest.param(
        Case(
            "[NEW] 54a. model_type is required",
            config=_base_without("model_type"),
            allowed_exit_codes={1, 2},
            stdout_must_match=[r"model_type|required|missing"],
        ),
        id="t54a",
    ),
    pytest.param(
        Case(
            "[NEW] 54b. model_type rejects bad enum",
            config=_base_with(model_type="transformer"),
            allowed_exit_codes={1, 2},
            stdout_must_match=[r"model_type|xgboost|dpcnn|cnn|enum"],
        ),
        id="t54b",
    ),
    pytest.param(
        Case(
            "[NEW] 55a. xgboost accepts train_method cyclic",
            config=_xgb_base_with(train_method="cyclic"),
            allowed_exit_codes={0},
        ),
        id="t55a",
    ),
    pytest.param(
        Case(
            "[NEW] 55b. xgboost rejects bad train_method enum",
            config=_xgb_base_with(train_method="invalid_method"),
            allowed_exit_codes={1, 2},
            stdout_must_match=[r"train_method|bagging|cyclic|enum"],
        ),
        id="t55b",
    ),
    pytest.param(
        Case(
            "[NEW] 56a. cnn config-only load succeeds",
            config=_cnn_base_with(),
            allowed_exit_codes={0},
        ),
        id="t56a",
    ),
    pytest.param(
        Case(
            "[NEW] 56b. xgboost config-only load succeeds",
            config=_xgb_base_with(),
            allowed_exit_codes={0},
        ),
        id="t56b",
    ),
]


@pytest.mark.parametrize("case", CASES)
def test_run_py_regressions(case: Case, tmp_path: Path, request: pytest.FixtureRequest) -> None:
    is_t46 = request.node.callspec.id == "t46"

    if is_t46 and not request.config.getoption("--run-e2e"):
        pytest.skip("t46 is skipped by default. Use --run-t46 to enable it.")

    cmd = [sys.executable, str(RUN_PY)]
    if not is_t46:
        cmd += ["--check_only", "True"]

    if case.create_config_file:
        cfg_path = _write_config(tmp_path, case)
        cmd += ["--config", str(cfg_path)]
    else:
        cmd += ["--config", str(tmp_path / case.config_filename)]

    cmd += list(case.cli_args)

    proc = _run(cmd, capture=not is_t46)

    assert proc.returncode in case.allowed_exit_codes, (
        f"Case '{case.name}' failed.\n"
        f"Command: {' '.join(cmd)}\n"
        f"Allowed exit codes: {sorted(case.allowed_exit_codes)}\n"
        f"Actual exit code: {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
    )

    if not is_t46:
        _assert_patterns(proc.stdout, case.stdout_must_match, case.stdout_must_not_match, "stdout")
