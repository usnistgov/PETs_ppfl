# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software
#
# This file was edited with the assistance of Claude Code (Anthropic, models
# Claude Opus 4.8 and Claude Opus 5). The assistant proposed and wrote direct
# unit tests for the schema-driven configuration engine in utils.py, and added
# characterization tests pinning the _strtobool replacement to the behavior of
# the removed distutils.util.strtobool, in accordance with the author's
# instructions. All content has been reviewed and verified by the authors.

import pytest

from utils import (
    UnknownParameterError,
    _coerce_cli_value,
    _strtobool,
    _get_validation_errors,
    _prune_inactive_model_params,
    _sync_enabled_flags,
    _to_layered_config,
    apply_defaults,
    validate_data_size,
    validate_dir_path,
    validate_file_path,
)


# --------------------------------------------------------------------------- #
# _coerce_cli_value                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, schema, expected",
    [
        ("true", {"type": "boolean"}, True),
        ("false", {"type": "boolean"}, False),
        ("5", {"type": "integer"}, 5),
        ("1e-3", {"type": "number"}, 0.001),
        ("[1, 2, 3]", {"type": "array"}, [1, 2, 3]),
        ('{"a": 1}', {"type": "object"}, {"a": 1}),
        ("plain", {"type": "string"}, "plain"),
    ],
)
def test_coerce_cli_value_by_type(raw, schema, expected):
    assert _coerce_cli_value(raw, schema) == expected


def test_coerce_cli_value_bad_number_falls_back_to_raw():
    # An uncoercible value is returned untouched so schema validation can report
    # a precise, field-level type error later.
    assert _coerce_cli_value("not_a_number", {"type": "integer"}) == "not_a_number"


def test_coerce_cli_value_non_string_passthrough():
    assert _coerce_cli_value(42, {"type": "integer"}) == 42


def test_coerce_cli_value_resolves_type_through_combinators():
    schema = {"oneOf": [{"type": "integer"}]}
    assert _coerce_cli_value("7", schema) == 7


# --------------------------------------------------------------------------- #
# _strtobool                                                                  #
# --------------------------------------------------------------------------- #
# Characterization: _strtobool replaced distutils.util.strtobool, which was
# removed from the standard library in Python 3.12. These cases were captured
# from the behavior of the original function and must not drift -- the rejected
# values in particular, because _coerce_cli_value depends on the ValueError to
# fall back to the raw string.
@pytest.mark.parametrize(
    "raw", ["y", "yes", "t", "true", "on", "1", "True", "TRUE", "Yes"]
)
def test_strtobool_accepts_truthy_vocabulary(raw):
    assert _strtobool(raw) is True


@pytest.mark.parametrize(
    "raw", ["n", "no", "f", "false", "off", "0", "False", "FALSE", "Off"]
)
def test_strtobool_accepts_falsy_vocabulary(raw):
    assert _strtobool(raw) is False


@pytest.mark.parametrize("raw", ["maybe", "", "2", "-1", " true", "true "])
def test_strtobool_rejects_everything_else(raw):
    # Note that surrounding whitespace is *not* stripped, matching the original.
    with pytest.raises(ValueError):
        _strtobool(raw)


def test_strtobool_rejection_makes_coerce_fall_back_to_raw():
    assert _coerce_cli_value("maybe", {"type": "boolean"}) == "maybe"


# --------------------------------------------------------------------------- #
# apply_defaults                                                              #
# --------------------------------------------------------------------------- #
def test_apply_defaults_fills_scalar_and_nested_defaults():
    schema = {
        "type": "object",
        "properties": {
            "a": {"default": 5},
            "nested": {"type": "object", "properties": {"c": {"default": 7}}},
        },
    }
    instance = {}
    apply_defaults(schema, instance)
    assert instance == {"a": 5, "nested": {"c": 7}}


def test_apply_defaults_does_not_override_supplied_values():
    schema = {"type": "object", "properties": {"a": {"default": 5}}}
    instance = {"a": 1}
    apply_defaults(schema, instance)
    assert instance["a"] == 1


def test_apply_defaults_selects_oneof_branch_by_discriminator():
    schema = {
        "oneOf": [
            {"properties": {"model_type": {"const": "cnn"}, "lr": {"default": 0.1}}},
            {"properties": {"model_type": {"const": "xgb"}, "eta": {"default": 0.3}}},
        ]
    }
    instance = {"model_type": "cnn"}
    apply_defaults(schema, instance)
    assert instance["lr"] == 0.1
    assert "eta" not in instance


# --------------------------------------------------------------------------- #
# _to_layered_config                                                          #
# --------------------------------------------------------------------------- #
def test_to_layered_config_nests_flat_keys_by_schema():
    schema = {
        "type": "object",
        "properties": {
            "num_rounds": {"type": "integer"},
            "federated": {"type": "object", "properties": {"num_clients": {"type": "integer"}}},
        },
    }
    flat = {"num_rounds": 5, "num_clients": 3}
    layered = _to_layered_config(flat, schema)
    assert layered["num_rounds"] == 5
    assert layered["federated"]["num_clients"] == 3


# --------------------------------------------------------------------------- #
# _prune_inactive_model_params                                                #
# --------------------------------------------------------------------------- #
def _two_model_schema():
    return {
        "allOf": [
            {
                "oneOf": [
                    {"properties": {"model_type": {"const": "cnn"},
                                    "model_params": {"properties": {"learning_rate": {}, "batch_size": {}}}}},
                    {"properties": {"model_type": {"const": "xgboost"},
                                    "model_params": {"properties": {"eta": {}, "batch_size": {}}}}},
                ]
            }
        ]
    }


def test_prune_drops_other_models_params_but_keeps_unknown():
    cfg = {
        "model_type": "cnn",
        "problem_type": "regression",
        "model_params": {"learning_rate": 0.1, "eta": 0.5, "batch_size": 8, "custom": 9},
    }
    pruned = _prune_inactive_model_params(cfg, _two_model_schema())
    params = pruned["model_params"]
    assert "learning_rate" in params   # active for cnn
    assert "batch_size" in params      # shared
    assert "custom" in params          # unknown -> preserved
    assert "eta" not in params         # belongs to xgboost only


def test_prune_removes_task_specific_keys():
    cfg = {"model_type": "cnn", "problem_type": "classification",
           "accuracy_tolerance": 0.1, "class_labels": [0, 1], "model_params": {}}
    pruned = _prune_inactive_model_params(cfg, _two_model_schema())
    assert "accuracy_tolerance" not in pruned  # regression-only key
    assert "class_labels" in pruned            # classification key retained


# --------------------------------------------------------------------------- #
# _sync_enabled_flags                                                         #
# --------------------------------------------------------------------------- #
def _enabled_flag_schema():
    return {
        "type": "object",
        "properties": {
            "dp": {
                "type": "object",
                "properties": {
                    "dp_enabled": {"type": "boolean"},
                    "epsilon": {"type": "number"},
                },
            }
        },
    }


def test_sync_enabled_flag_inferred_true_when_group_populated():
    cfg = {"dp": {"epsilon": 1.0}}
    _sync_enabled_flags(cfg, _enabled_flag_schema())
    assert cfg["dp"]["dp_enabled"] is True


def test_sync_enabled_flag_inferred_false_when_group_empty():
    cfg = {"dp": {}}
    _sync_enabled_flags(cfg, _enabled_flag_schema())
    assert cfg["dp"]["dp_enabled"] is False


def test_sync_enabled_flag_respects_explicit_value():
    cfg = {"dp": {"dp_enabled": False, "epsilon": 1.0}}
    _sync_enabled_flags(cfg, _enabled_flag_schema())
    assert cfg["dp"]["dp_enabled"] is False


# --------------------------------------------------------------------------- #
# _get_validation_errors                                                      #
# --------------------------------------------------------------------------- #
def test_validation_errors_normalizes_missing_required():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    errors = _get_validation_errors(schema, {})
    assert ("n", None, "Missing required parameter") in errors


def test_validation_errors_normalizes_unknown_property():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "additionalProperties": False}
    errors = _get_validation_errors(schema, {"n": 1, "bogus": 2})
    paths = [p for p, _, _ in errors]
    assert "bogus" in paths
    assert any(msg == "Unknown parameter" for _, _, msg in errors)


def test_validation_errors_reports_type_mismatch():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    errors = _get_validation_errors(schema, {"n": "x"})
    assert any(path == "n" for path, _, _ in errors)


# --------------------------------------------------------------------------- #
# UnknownParameterError                                                       #
# --------------------------------------------------------------------------- #
def test_unknown_parameter_error_includes_suggestion():
    err = UnknownParameterError([("epochz", "epochs")], where="CLI")
    assert "epochz" in str(err)
    assert "did you mean 'epochs'" in str(err)
    assert err.parameters == ["epochz"]


def test_unknown_parameter_error_without_suggestion():
    err = UnknownParameterError([("mystery", None)])
    assert "mystery" in str(err)
    assert "did you mean" not in str(err)


# --------------------------------------------------------------------------- #
# path / data-size validation                                                 #
# --------------------------------------------------------------------------- #
def test_validate_file_path_ok_and_missing(tmp_path):
    existing = tmp_path / "f.txt"
    existing.write_text("x")
    validate_file_path(str(existing))  # should not raise
    with pytest.raises(FileNotFoundError):
        validate_file_path(str(tmp_path / "missing.txt"))


def test_validate_dir_path_ok_and_missing(tmp_path):
    validate_dir_path(str(tmp_path))  # should not raise
    with pytest.raises(FileNotFoundError):
        validate_dir_path(str(tmp_path / "no_such_dir"))


def test_validate_data_size_rejects_oversized_batch_npy(npy_data_dir):
    # Fixture writes 6-row feature files.
    with pytest.raises(ValueError, match="Batch size"):
        validate_data_size(str(npy_data_dir["dir"]), batch_size=8)


def test_validate_data_size_accepts_fitting_batch_npy(npy_data_dir):
    validate_data_size(str(npy_data_dir["dir"]), batch_size=4)  # should not raise


def test_validate_data_size_falls_back_to_dat(dat_data_dir):
    # No .npy files exist yet; the 2-row S_tt_vcf.dat must still be honored.
    with pytest.raises(ValueError, match="Batch size"):
        validate_data_size(str(dat_data_dir["dir"]), batch_size=8)
