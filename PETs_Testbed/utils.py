# This Software (PETs Testbed) is being made available as a public service by the
# National Institute of Standards and Technology (NIST), an Agency of the United
# States Department of Commerce. This software was developed in part by employees of
# NIST and in part by NIST contractors. Copyright in portions of this software that
# were developed by NIST contractors has been licensed or assigned to NIST. Pursuant
# to Title 17 United States Code Section 105, works of NIST employees are not
# subject to copyright protection in the United States. However, NIST may hold
# international copyright in software created by its employees and domestic
# copyright (or licensing rights) in portions of software that were assigned or
# licensed to NIST. To the extent that NIST holds copyright in this software, it is
# being made available under the Creative Commons Attribution 4.0 International
# license (CC BY 4.0). The disclaimers of the CC BY 4.0 license apply to all parts
# of the software developed or licensed by NIST.
#
# ACCESS THE FULL CC BY 4.0 LICENSE HERE:
# https://creativecommons.org/licenses/by/4.0/legalcode

import argparse
import numpy as np
from collections import Counter
import torch
import json
from copy import deepcopy
from distutils.util import strtobool
from pathlib import Path
from typing import Any, Dict, Tuple, List, Set,  Union
from xgboost import XGBClassifier, Booster
import os
import pickle
import re
from report import Report

from jsonschema.validators import validator_for

###
#   get_device()
#   purpose: Return the default torch device, preferring CUDA when available and falling back to CPU otherwise.
###
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

###
#   validate_data_size(data_path, batch_size)
#   purpose: Validate that the discovered train/holdout dataset files are large enough for the requested batch divisor.
###
def validate_data_size(data_path, batch_size):
    suffixes = ["_tt_vcf", "_ho_vcf"]
    for suffix in suffixes:
        npy_matches = [f for f in os.listdir(data_path) if f.endswith(f"{suffix}.npy")]
        if len(npy_matches) > 1:
            raise ValueError(f"Multiple files found in {data_path} matching {suffix}.npy: {npy_matches}")

        if npy_matches:
            file_path = os.path.relpath(os.path.join(data_path, npy_matches[0]))
            num_data_rows = np.load(file_path, mmap_mode="r").shape[0]
            if batch_size > num_data_rows:
                raise ValueError(f"Batch divisor (batch_divisor={batch_size}) is greater than train/test dataset size (num_rows={num_data_rows})")
            continue



def print_binned_counts(dataset: np.ndarray, indices: List[int] | np.ndarray, num_bins: int = 10):
    """
    Count the occurrences of binned labels for a given indices in the dataset
    and print the counts with the bin ranges

    Args:
        dataset: np.ndarray
            The dataset to use for counting binned labels.
        indices: List[int] | np.ndarray
            The indices to use for counting binned labels.
        num_bins: int
            The number of bins to use for binning the labels.
    Returns:
        None
    """
    # Get labels from the dataset for given indices
    labels = dataset[indices, -1]
    # Create bins for the selected labels
    bins = np.linspace(np.min(labels), np.max(labels), num_bins + 1)
    binned_labels = np.digitize(labels, bins) - 1
    # Count occurrences of each bin
    binned_counts = Counter(binned_labels)
    # Print binned label counts with ranges
    for bin_idx, count in sorted(binned_counts.items()):
        if bin_idx < len(bins) - 1:
            print(
                f"{bins[bin_idx]:.2f} - {bins[bin_idx + 1]:.2f}: "
                f"{count} records"
            )

###
#   configure_warning_logging(output_dir)
#   purpose: Route Python warnings into a warnings.log file under output_dir instead of printing them to the terminal.
###
def configure_warning_logging(output_dir):
    import logging
    import warnings
    import threading
    from pathlib import Path

    logging.captureWarnings(True)
    warnings.simplefilter("default")

    log_file = Path(output_dir) / "PETs_warnings.log"

    warning_logger = logging.getLogger("py.warnings")
    warning_logger.setLevel(logging.WARNING)
    warning_logger.propagate = False

    if not any(isinstance(h, logging.FileHandler) for h in warning_logger.handlers):
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(process)d %(levelname)s %(message)s"
        ))
        warning_logger.addHandler(handler)

    stderr_copy = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    pattern = re.compile(r"\[[^\]]+\s[EW]\s\d+\s\d+\]")

    def forward_stderr():
        with (
            os.fdopen(read_fd, "r", buffering=1, errors="replace") as src,
            os.fdopen(stderr_copy, "w", buffering=1, errors="replace") as term,
            open(log_file, "a", buffering=1) as log,
        ):
            for line in src:
                term.write(line)
                term.flush()
                if pattern.search(line):
                    log.write(line)
                    log.flush()

    threading.Thread(target=forward_stderr, daemon=True).start()

###
#   _is_object_schema(sch)
#   purpose: Return True when the given schema behaves like a JSON object schema, based on type or properties.
###
def _is_object_schema(sch: dict) -> bool:
    return isinstance(sch, dict) and (sch.get("type") == "object" or "properties" in sch)

###
#   _has_any_default(sch)
#   purpose: Recursively check whether a schema or any nested applicable subschema defines a default value.
###
def _has_any_default(sch: dict) -> bool:
    if not isinstance(sch, dict):
        return False
    if "default" in sch:
        return True
    if _is_object_schema(sch):
        for subschema in sch.get("properties", {}).values():
            if _has_any_default(subschema):
                return True
    for k in ("allOf", "oneOf", "anyOf"):
        for subschema in sch.get(k, []) if isinstance(sch.get(k), list) else []:
            if _has_any_default(subschema):
                return True
    return False

###
#   _select_oneof_branch(oneof_list, instance)
#   purpose: Select the matching oneOf branch using model_type as a discriminator from the current instance.
###
def _select_oneof_branch(oneof_list, instance):
    mt = instance.get("model_type")
    for opt in oneof_list:
        const = opt.get("properties", {}).get("model_type", {}).get("const")
        if const == mt:
            return opt
    return None

###
#   apply_defaults(schema, instance)
#   purpose: Recursively apply schema defaults to a config instance, including nested object properties.
###
def apply_defaults(schema: dict, instance):
    # Layer: allOf
    for sub in schema.get("allOf", []):
        apply_defaults(sub, instance)

    # Choose: oneOf (by model_type discriminator)
    if "oneOf" in schema:
        chosen = _select_oneof_branch(schema["oneOf"], instance)
        if chosen is not None:
            apply_defaults(chosen, instance)

    # Apply defaults for object properties (even if "type":"object" is omitted)
    if _is_object_schema(schema) and isinstance(instance, dict):
        for prop, prop_schema in schema.get("properties", {}).items():
            if prop not in instance:
                if isinstance(prop_schema, dict) and "default" in prop_schema:
                    instance[prop] = deepcopy(prop_schema["default"])
                elif _is_object_schema(prop_schema) and _has_any_default(prop_schema):
                    instance[prop] = {}
                else:
                    continue

            if isinstance(instance.get(prop), dict):
                apply_defaults(prop_schema, instance[prop])

    return instance

###
#   _schema_leaf_map(schema)
#   purpose: Build a mapping from leaf property paths to their corresponding schema definitions.
###
def _schema_leaf_map(schema: Dict[str, Any]) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    leaf_paths = {}
    for path, prop_schema in _iter_schema_leaf_args(schema):
        if path not in leaf_paths:
            leaf_paths[path] = prop_schema
    return leaf_paths

###
#   _build_schema_path_index(schema)
#   purpose: Build lookup tables for resolving schema leaf paths by leaf name or dotted path.
###
def _build_schema_path_index(schema: Dict[str, Any]):
    by_name: Dict[str, Tuple[str, ...]] = {}
    by_dot: Dict[str, Tuple[str, ...]] = {}
    name_to_paths: Dict[str, Set[Tuple[str, ...]]] = {}

    for path in _schema_leaf_map(schema).keys():
        by_dot[".".join(path)] = path
        name_to_paths.setdefault(path[-1], set()).add(path)

    for name, paths in name_to_paths.items():
        if len(paths) == 1:
            by_name[name] = next(iter(paths))

    return by_name, by_dot

###
#   _model_param_keys_by_model_type(schema)
#   purpose: Collect the model_params keys declared for each model_type branch.
###
def _model_param_keys_by_model_type(schema: Dict[str, Any]) -> Dict[str, Set[str]]:
    keys_by_model: Dict[str, Set[str]] = {}

    for sub in schema.get("allOf", []) or []:
        for branch in sub.get("oneOf", []) or []:
            props = branch.get("properties", {})
            model_type = props.get("model_type", {}).get("const")
            model_params = props.get("model_params", {})
            param_props = model_params.get("properties", {})
            if isinstance(model_type, str) and isinstance(param_props, dict):
                keys_by_model[model_type] = set(param_props.keys())

    return keys_by_model

###
#   _prune_inactive_model_params(cfg, schema)
#   purpose: Drop known model_params that do not apply to the selected model_type.
###
def _prune_inactive_model_params(cfg: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    if cfg.get("problem_type") != "classification":
        cfg.pop("class_labels", None)
    if cfg.get("problem_type") != "regression":
        cfg.pop("accuracy_tolerance", None)

    model_type = cfg.get("model_type")
    model_params = cfg.get("model_params")
    if not isinstance(model_type, str) or not isinstance(model_params, dict):
        return cfg

    keys_by_model = _model_param_keys_by_model_type(schema)
    active_keys = keys_by_model.get(model_type)
    if active_keys is None:
        return cfg

    all_known_keys = set().union(*keys_by_model.values()) if keys_by_model else set()
    pruned_params = {}
    for key, value in model_params.items():
        if key in active_keys or key not in all_known_keys:
            pruned_params[key] = value

    cfg["model_params"] = pruned_params
    return cfg

###
#   _flatten_config_items(data, prefix=())
#   purpose: Flatten a nested config dictionary into dotted-path key/value pairs.
###
def _flatten_config_items(data: Dict[str, Any], prefix: Tuple[str, ...] = ()):
    for key, value in data.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            yield from _flatten_config_items(value, path)
        else:
            yield ".".join(path), value

###
#   _top_level_object_schemas(schema, instance)
#   purpose: Collect top-level object-valued schema groups that apply to the current config instance.
###
def _top_level_object_schemas(schema: Dict[str, Any], instance: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    top_objects: Dict[str, Dict[str, Any]] = {}

    for sch in _iter_applicable_schemas(schema, instance):
        props = sch.get("properties", {})
        if isinstance(props, dict):
            for key, prop_schema in props.items():
                if key not in top_objects and _is_object_schema(prop_schema):
                    top_objects[key] = prop_schema

    return top_objects

###
#   _sync_enabled_flags(cfg, schema)
#   purpose: Infer or validate group-level *_enabled flags based on the presence of other keys in each group.
###
def _sync_enabled_flags(cfg: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    top_objects = _top_level_object_schemas(schema, cfg)

    for group_name, group_schema in top_objects.items():
        group_cfg = cfg.get(group_name)
        if group_cfg is None:
            continue
        if not isinstance(group_cfg, dict):
            continue

        child_props: Dict[str, Any] = {}
        for sch in _iter_applicable_schemas(group_schema, group_cfg):
            props = sch.get("properties", {})
            if isinstance(props, dict):
                child_props.update(props)

        enabled_keys = [
            key
            for key, prop_schema in child_props.items()
            if isinstance(prop_schema, dict)
            and prop_schema.get("type") == "boolean"
            and key.endswith("_enabled")
        ]

        if len(enabled_keys) == 1:
            enabled_key = enabled_keys[0]
            if enabled_key not in group_cfg:
                cfg[group_name][enabled_key] = any(k != enabled_key for k in group_cfg.keys())

    return cfg

###
#   _to_layered_config(flat_cfg, schema)
#   purpose: Convert a flat or partially nested config into a nested structure aligned to schema paths.
###
def _to_layered_config(flat_cfg: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    by_name, by_dot = _build_schema_path_index(schema)

    # Keep top-level object groups present, like federated/dp/model_params
    for group_name in _top_level_object_schemas(schema, flat_cfg).keys():
        cfg[group_name] = {}

    for flat_key, value in _flatten_config_items(flat_cfg):
        if flat_key in by_dot:
            _set_nested(cfg, by_dot[flat_key], value)
        elif "." not in flat_key and flat_key in by_name:
            _set_nested(cfg, by_name[flat_key], value)
        else:
            _set_nested(cfg, tuple(flat_key.split(".")), value)

    return cfg

###
# _validate_paths(paths, check_func, kind)
#   input: paths (string or iterable of strings), check_func (function like os.path.exists or os.path.isdir), kind (description for error message ("file", "directory", etc.))
#   output: none
#   purpose:  Internal helper to validate one or many paths.
###
def _validate_paths(paths, check_func, kind: str) -> None:

    # Normalize to a list of paths
    if not isinstance(paths, list):
        paths = [paths]

    failed = [p for p in paths if not check_func(p)]

    if failed:
        plural = "path" if len(failed) == 1 else "paths"
        raise FileNotFoundError(f"Could not find {kind} {plural} {failed}. Please check it and try again.")

###
#   validate_file_path(test_path)
#   input: A string representing a file path
#   output: none
#   purpose: to test if a single file path exists. If it does, no noticable action occurs. If it does not, an error is raised.
###
def validate_file_path(test_path: str) -> None:
    _validate_paths(test_path, os.path.exists, "file")

###
#   validate_dir_path(test_path)
#   input: A string representing a directory path
#   output: none
#   purpose: to test if a directory exists. If it does, no noticable action occurs. If it does not, an error is raised.
###
def validate_dir_path(test_path: str) -> None:
    _validate_paths(test_path, os.path.isdir, "directory")

###
#   _set_nested(cfg, path, value)
#   purpose: Set a value inside a nested dictionary, creating intermediate dictionaries as needed.
###
def _set_nested(cfg: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur = cfg
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value

###
#   _iter_applicable_schemas(schema, instance)
#   purpose: Yield the base schema and any allOf/selected oneOf subschemas that apply to the instance.
###
def _iter_applicable_schemas(schema: Dict[str, Any], instance: Any) -> List[Dict[str, Any]]:
    # Base schema applies
    schemas = [schema]

    # allOf: all apply
    for sub in schema.get("allOf", []) or []:
        schemas.extend(_iter_applicable_schemas(sub, instance))

    # oneOf: only chosen applies (using your discriminator helper)
    if "oneOf" in schema:
        chosen = _select_oneof_branch(schema["oneOf"], instance)
        if chosen is not None:
            schemas.extend(_iter_applicable_schemas(chosen, instance))

    return schemas

###
#   _iter_schema_layers(schema)
#   purpose: Iterate through the schema and its allOf layers in declaration order.
###
def _iter_schema_layers(schema: Dict[str, Any]):
    yield schema
    for sub in schema.get("allOf", []) or []:
        yield from _iter_schema_layers(sub)

###
#   _schema_key_order(schema)
#   purpose: Derive a stable property-printing order from the schema and its layered definitions.
###
def _schema_key_order(schema: Dict[str, Any]) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    for sch in _iter_schema_layers(schema):
        props = sch.get("properties", {})
        if isinstance(props, dict):
            for key in props.keys():
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)

    return ordered

###
#   _property_schema(schema, key)
#   purpose: Gather all schema fragments that define a given property and combine them into one schema view.
###
def _property_schema(schema: Dict[str, Any], key: str) -> Dict[str, Any]:
    prop_schemas = []

    for sch in _iter_schema_layers(schema):
        props = sch.get("properties", {})
        if isinstance(props, dict) and key in props and isinstance(props[key], dict):
            prop_schemas.append(props[key])

    if not prop_schemas:
        return {}

    return {"allOf": prop_schemas}

###
#   _print_config_by_schema(data, schema, indent=0)
#   purpose: Print config values in schema-defined key order, recursively formatting nested objects.
###
def _print_config_by_schema(data: Dict[str, Any], schema: Dict[str, Any], indent: int = 0) -> None:
    pad = "  " * indent
    ordered_keys = _schema_key_order(schema)

    for key in data.keys():
        if key not in ordered_keys and not key.startswith("_"):
            ordered_keys.append(key)

    non_dict_keys = [key for key in ordered_keys if key in data and not isinstance(data[key], dict)]
    dict_keys = [key for key in ordered_keys if key in data and isinstance(data[key], dict)]

    for key in non_dict_keys:
        if not "_enabled" in key:
            print(f"{pad}{key}={data[key]}")

    if indent == 0 and dict_keys:
        print()

    for key in dict_keys:
        print(f"{pad}{key}={{")
        child_schema = _property_schema(schema, key)
        _print_config_by_schema(data[key], child_schema, indent + 1)
        print(f"{pad}}}")
        if indent == 0:
            print()

###
#   _iter_schema_leaf_args(schema, prefix=())
#   purpose: Recursively iterate over leaf argument paths and their schemas from a nested JSON schema.
###
def _iter_schema_leaf_args(schema: Dict[str, Any], prefix: Tuple[str, ...] = ()):
    if not isinstance(schema, dict):
        return

    props = schema.get("properties", {})
    if isinstance(props, dict):
        for key, prop_schema in props.items():
            path = prefix + (key,)
            if _is_object_schema(prop_schema):
                yield from _iter_schema_leaf_args(prop_schema, path)
            else:
                yield path, prop_schema

    for sub in schema.get("allOf", []) or []:
        yield from _iter_schema_leaf_args(sub, prefix)

    for sub in schema.get("oneOf", []) or []:
        yield from _iter_schema_leaf_args(sub, prefix)

###
#   _schema_without_combinators(sch)
#   purpose: Return a copy of a schema with combinator keywords like allOf/oneOf/anyOf removed.
###
def _schema_without_combinators(sch: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in sch.items() if k not in ("allOf", "oneOf", "anyOf")}

###
#   _effective_schema(schema, instance)
#   purpose: Build the effective validation schema for an instance by combining all applicable schema parts.
###
def _effective_schema(schema: Dict[str, Any], instance: Any) -> Dict[str, Any]:
    parts = []

    for sch in _iter_applicable_schemas(schema, instance):
        cleaned = _schema_without_combinators(sch)
        if cleaned:
            parts.append(cleaned)

    if not parts:
        return {}

    if len(parts) == 1:
        return parts[0]

    return {"allOf": parts}

###
#   _validation_instance(cfg)
#   purpose: Create a copy of the config suitable for schema validation by removing non-schema runtime fields.
###
def _validation_instance(cfg: Dict[str, Any]) -> Dict[str, Any]:
    instance = deepcopy(cfg)
    instance.pop("check_only", None)
    return instance

###
#   _get_validation_errors(schema, instance)
#   purpose: Validate an instance against a schema and return normalized, user-friendly error tuples.
###
def _get_validation_errors(schema: Dict[str, Any], instance: Dict[str, Any]) -> List[Tuple[str, Any, str]]:
    Validator = validator_for(schema)
    Validator.check_schema(schema)
    validator = Validator(schema)

    errors = []
    for err in sorted(validator.iter_errors(instance), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in err.path)

        if err.validator == "additionalProperties":
            bad_keys = re.findall(r"'([^']+)'", err.message)
            if bad_keys:
                for bad_key in bad_keys:
                    full_path = f"{path}.{bad_key}" if path else bad_key
                    errors.append((full_path, None, "Unknown parameter"))
            else:
                errors.append((path, None, err.message))

        elif err.validator == "required":
            m = re.search(r"'([^']+)' is a required property", err.message)
            missing_key = m.group(1) if m else None
            if missing_key is not None:
                full_path = f"{path}.{missing_key}" if path else missing_key
                errors.append((full_path, None, "Missing required parameter"))
            else:
                errors.append((path, None, err.message))

        else:
            errors.append((path, err.instance, err.message))

    return errors

###
#   _coerce_cli_value(raw_value, sch)
#   purpose: Convert a CLI string value into the schema-expected Python type when possible.
###
def _coerce_cli_value(raw_value: str, sch: Dict[str, Any]) -> Any:
    if not isinstance(raw_value, str) or not isinstance(sch, dict):
        return raw_value

    t = sch.get("type")

    try:
        if t == "boolean":
            return bool(strtobool(raw_value))
        if t == "integer":
            return int(raw_value)
        if t == "number":
            return float(raw_value)
        if t == "array" or t == "object":
            return json.loads(raw_value)
    except (ValueError, TypeError, json.JSONDecodeError):
        return raw_value

    for k in ("allOf", "oneOf", "anyOf"):
        for sub in sch.get(k, []) if isinstance(sch.get(k), list) else []:
            coerced = _coerce_cli_value(raw_value, sub)
            if coerced is not raw_value or sub.get("type") in ("boolean", "integer", "number", "array", "object"):
                return coerced

    return raw_value

def save_xgb(
    model: Union[XGBClassifier, Booster],
    metadata: Dict[str, any],
    name: str,
    round_number: int | None = None,
    output_dir: str | Path | None = None,
):
    out_name = (
        f"{name}_round_{round_number}" if round_number is not None else name
    )
    parent_path = Path(output_dir).absolute() if output_dir is not None else Path(__file__).parent
    parent_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(parent_path, f"{out_name}.ubj")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    metadata_path_name = f"{out_name}_meta.npz"
    metadata_path = Path(parent_path, metadata_path_name)
    np.savez(metadata_path, **metadata, allow_pickle=True)
    report = Report(metadata)
    report.save_to_file(Path(parent_path, f"{out_name}.json"))
    print(f"Model metadata saved to {metadata_path}")

class UnknownParameterError(ValueError):
    """Exception raised for unknown parameters."""
    def __init__(self, issues, where="config"):
        self.issues = issues
        self.parameters = [name for name, _ in issues]

        details = []
        for name, suggestion in issues:
            if suggestion:
                details.append(f"'{name}' (did you mean '{suggestion}'?)")
            else:
                details.append(f"'{name}'")

        msg = f"Unknown parameter name(s) in {where}: " + ", ".join(details)
        super().__init__(msg)

class ConfigArgs(argparse.Namespace):
    def __init__(self, schema: Dict[str, Any] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._print_schema = schema or {}

    def print(self):
        data = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        _print_config_by_schema(data, self._print_schema)

class ConfigPipeline:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(exit_on_error=False)
        self.args = None
        self._cli_dest_to_path = {}
        self._schema_args_added = False
        self._cli_dest_to_schema = {}

        # "meta" args
        base_dir = Path(__file__).resolve().parent.parent
        default_config = base_dir / "configs" / "config.json"
        default_schema = base_dir / "schemas" / "configuration-schema.json"
        self.parser.add_argument("--config", default=str(default_config), type=str)
        self.parser.add_argument("--schema", default=str(default_schema), type=str)
        self.parser.add_argument("--check_only", action="store_true")

    ###
    #   _load_json(path)
    #   purpose: Load and return a JSON file from disk as a Python dictionary.
    ###
    def _load_json(self, path: str) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    ###
    #   _add_schema_cli_arguments(schema)
    #   purpose: Dynamically add CLI arguments for schema leaf properties and track their path mappings.
    ###
    def _add_schema_cli_arguments(self, schema: Dict[str, Any]) -> None:
        if self._schema_args_added:
            return

        leaf_paths = _schema_leaf_map(schema)

        leaf_name_counts = Counter(path[-1] for path in leaf_paths.keys())

        for path, prop_schema in leaf_paths.items():
            # Use plain leaf name when unique, otherwise dotted path
            if leaf_name_counts[path[-1]] == 1:
                arg_name = path[-1]
            else:
                arg_name = ".".join(path)

            dest = "__".join(path)

            self.parser.add_argument(f"--{arg_name}", dest=dest, default=None, type=str)
            self._cli_dest_to_path[dest] = path
            self._cli_dest_to_schema[dest] = prop_schema

        self._schema_args_added = True

    ###
    #   _apply_cli_overrides(cfg, args)
    #   purpose: Apply CLI-provided values onto the nested config, coercing values using schema metadata.
    ###
    def _apply_cli_overrides(self, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        d = vars(args)

        for dest, path in self._cli_dest_to_path.items():
            if d.get(dest) is not None:
                value = _coerce_cli_value(d[dest], self._cli_dest_to_schema[dest])
                _set_nested(cfg, path, value)

        # carry check_only (not part of schema, but you use it)
        cfg["check_only"] = bool(d.get("check_only", False))
        return cfg

    def parse(self) -> argparse.Namespace:
        try:
            # first pass: only meta args (config, schema, check_only) are known yet
            meta_args, _ = self.parser.parse_known_args()

            # validate and load paths
            validate_file_path([meta_args.config, meta_args.schema])
            schema = self._load_json(meta_args.schema)
            raw_cfg = self._load_json(meta_args.config)

            # build CLI args from schema, then parse again
            self._add_schema_cli_arguments(schema)
            args, unknown = self.parser.parse_known_args()
            if unknown:
                bad_cli_params = [elem for elem in unknown if elem.startswith("--")]
                if bad_cli_params:
                    issues = [
                        (elem[2:].split("=", 1)[0].split(".")[-1], None)
                        for elem in bad_cli_params
                    ]
                    raise UnknownParameterError(issues, where="CLI")
                raise UnknownParameterError(
                    [(elem.split("=", 1)[0].split(".")[-1], None) for elem in unknown],
                    where="CLI",
                )
            
            # layer + apply schema defaults
            cfg = _to_layered_config(raw_cfg, schema)
            cfg = self._apply_cli_overrides(cfg, args)
            cfg = _sync_enabled_flags(cfg, schema)
            cfg = apply_defaults(schema=schema, instance=cfg)
            cfg = _prune_inactive_model_params(cfg, schema)

            validation_cfg = _validation_instance(cfg)
            effective_schema = _effective_schema(schema, validation_cfg)

            # validate final config
            validation_errors = _get_validation_errors(effective_schema, validation_cfg)
            if validation_errors:
                msg = "An error occured during configuration validation. Please check the following issue(s):\n"
                for path, bad_value, err_msg in validation_errors:
                    path = path.split(".")[-1]
                    if bad_value is None:
                        msg += f"  {path} -> {err_msg}\n"
                    else:
                        msg += f"  {path}={bad_value!r} -> {err_msg}\n"
                raise ValueError(msg.rstrip())

            # path checks / postprocessing (same logic you already had)
            if cfg.get("model_params", {}).get("data_partitions_file", "") not in ("", None):
                validate_file_path(cfg["model_params"]["data_partitions_file"])
            #validate_dir_path(cfg["output_dir"])
            validate_dir_path(cfg["data_dir"])

            # ensure batch_size size aligns with data size
            validate_data_size(cfg["data_dir"], cfg["model_params"]["batch_size"])

            # your equalization logic
            if cfg["model_params"]["partition_id"] > cfg["num_partitions"]:
                raise ValueError("partition_id must be <= num_partitions.")

            if cfg.get("federated", {}).get("federated_enabled", False):
                if cfg["federated"]["num_clients"] > cfg["num_partitions"]:
                    raise ValueError("num_clients must be <= num_partitions.")

            if cfg.get("dp", {}).get("dp_enabled", False) and cfg["dp"].get("opacus_secure_mode", False):
                cfg["dp"]["opacus_secure_mode"] = False
        except UnknownParameterError as e:
            print(e)
            for elem in e.parameters:
                name = elem[2:] if elem.startswith("--") else elem
                name = name.split("=", 1)[0]
                name = name.split(".")[-1]
                print(f"{name}\t", end="")
            print()
            exit(1)
        except FileNotFoundError as e:
            print(e)
            exit(1)
        except json.JSONDecodeError as e:
            print(f"An error occured while parsing the config file.", end = "")
            print(f"Please check the following issue(s): {e.message}.")
            exit(1)
        except ValueError as e:
            print(e)
            exit(1)
        

        self.args = ConfigArgs(schema=effective_schema, **cfg)
        return self.args
