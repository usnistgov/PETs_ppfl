import argparse
import types
from typing import List, Dict, Any
import numpy as np
from collections import Counter
import torch
import json
from copy import deepcopy
from distutils.util import strtobool
from pathlib import Path
from typing import Any, Dict, Tuple, List, Set
import os
import pickle

from jsonschema import validate, ValidationError

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def validate_data_size(data_path, batch_divisor):
    pattern = ["_tt_vcf.dat", "_ho_vcf.dat"]
    for pat in pattern: 
        matches = [f for f in os.listdir(data_path) if f.endswith(pat)]
        if not matches:
            raise FileNotFoundError(f"No file found in {data_path} matching {pat}")
        if len(matches) > 1:
            raise ValueError(f"Multiple files found in {data_path} matching {pat}: {matches}")

        file_path = os.path.relpath(os.path.join(data_path, matches[0]))
        with open(file_path, "rb") as f:
            num_data_rows = pickle.load(f).shape[0]
            if batch_divisor > num_data_rows:
                raise ValueError(f"Batch divisor (batch_divisor={batch_divisor}) is greater than train/test dataset size (num_rows={num_data_rows})")


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
def _iter_subschemas(sch: Dict[str, Any]):
    if not isinstance(sch, dict):
        return

    for k in ("allOf", "oneOf", "anyOf"):
        for sub in sch.get(k, []) if isinstance(sch.get(k), list) else []:
            yield sub
            yield from _iter_subschemas(sub)

def _is_object_schema(sch: dict) -> bool:
    return isinstance(sch, dict) and (sch.get("type") == "object" or "properties" in sch)

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

def _select_oneof_branch(oneof_list, instance):
    mt = instance.get("model_type")
    for opt in oneof_list:
        const = opt.get("properties", {}).get("model_type", {}).get("const")
        if const == mt:
            return opt
    return None

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

def _build_schema_path_index(schema: Dict[str, Any]):
    by_name: Dict[str, Tuple[str, ...]] = {}
    by_dot: Dict[str, Tuple[str, ...]] = {}
    name_to_paths: Dict[str, Set[Tuple[str, ...]]] = {}
    seen_paths: Set[Tuple[str, ...]] = set()

    for path, _ in _iter_schema_leaf_args(schema):
        if path in seen_paths:
            continue
        seen_paths.add(path)
        by_dot[".".join(path)] = path
        name_to_paths.setdefault(path[-1], set()).add(path)

    for name, paths in name_to_paths.items():
        if len(paths) == 1:
            by_name[name] = next(iter(paths))

    return by_name, by_dot

def _flatten_config_items(data: Dict[str, Any], prefix: Tuple[str, ...] = ()):
    for key, value in data.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            yield from _flatten_config_items(value, path)
        else:
            yield ".".join(path), value

def _top_level_object_schemas(schema: Dict[str, Any], instance: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    top_objects: Dict[str, Dict[str, Any]] = {}

    for sch in _iter_applicable_schemas(schema, instance):
        props = sch.get("properties", {})
        if isinstance(props, dict):
            for key, prop_schema in props.items():
                if key not in top_objects and _is_object_schema(prop_schema):
                    top_objects[key] = prop_schema

    return top_objects

def _sync_enabled_flags(cfg: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    top_objects = _top_level_object_schemas(schema, cfg)

    for group_name, group_schema in top_objects.items():
        cfg.setdefault(group_name, {})
        group_cfg = cfg.get(group_name)

        if not isinstance(group_cfg, dict):
            continue

        child_props: Dict[str, Any] = {}
        for sch in _iter_applicable_schemas(group_schema, group_cfg):
            props = sch.get("properties", {})
            if isinstance(props, dict):
                child_props.update(props)

        enabled_keys = [
            key for key, prop_schema in child_props.items()
            if isinstance(prop_schema, dict)
            and prop_schema.get("type") == "boolean"
            and key.endswith("_enabled")
        ]

        if len(enabled_keys) == 1:
            enabled_key = enabled_keys[0]
            cfg[group_name][enabled_key] = any(k != enabled_key for k in group_cfg.keys())

    return cfg

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
            # preserve unknown keys so find_unknown_fields() can report them
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

def _set_nested(cfg: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur = cfg
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value

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

def _ordered_keys_at_level(applicable: List[Dict[str, Any]]) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    for sch in applicable:
        props = sch.get("properties")
        if isinstance(props, dict):
            for key in props.keys():
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)

    return ordered

def _allowed_keys_at_level(applicable: List[Dict[str, Any]]) -> Set[str]:
    return set(_ordered_keys_at_level(applicable))

def find_unknown_fields(schema: Dict[str, Any], instance: Any, path: str = "") -> List[str]:
    """
    Returns dot-paths for keys present in instance that are not present in the
    relevant schema portions (allOf layers + chosen oneOf branch).
    """
    if not isinstance(instance, dict):
        return []

    applicable = _iter_applicable_schemas(schema, instance)
    allowed_here = _allowed_keys_at_level(applicable)
    allowed_here.update(["check_only"])

    bad: List[str] = []
    for k, v in instance.items():
        p = f"{path}.{k}" if path else k
        if k not in allowed_here:
            bad.append(p)
            continue

        # Recurse if value is an object and schema for that key looks object-ish in any applicable layer
        if isinstance(v, dict):
            # merge the property schemas for k from all applicable layers using allOf
            child_schema = _property_schema(schema, instance, k)
            if child_schema:
                bad.extend(find_unknown_fields(child_schema, v, p))

    return bad

def _schema_key_order(schema: Dict[str, Any], instance: Dict[str, Any]) -> List[str]:
    return _ordered_keys_at_level(_iter_applicable_schemas(schema, instance))

def _property_schema(schema: Dict[str, Any], instance: Dict[str, Any], key: str) -> Dict[str, Any]:
    prop_schemas = []

    for sch in _iter_applicable_schemas(schema, instance):
        props = sch.get("properties", {})
        if isinstance(props, dict) and key in props and isinstance(props[key], dict):
            prop_schemas.append(props[key])

    if not prop_schemas:
        return {}

    return {"allOf": prop_schemas}

def _print_config_by_schema(data: Dict[str, Any], schema: Dict[str, Any], indent: int = 0) -> None:
    pad = "  " * indent
    ordered_keys = _schema_key_order(schema, data)

    for key in data.keys():
        if key not in ordered_keys and not key.startswith("_"):
            ordered_keys.append(key)

    non_dict_keys = [key for key in ordered_keys if key in data and not isinstance(data[key], dict)]
    dict_keys = [key for key in ordered_keys if key in data and isinstance(data[key], dict)]

    for key in non_dict_keys:
        if not "enabled" in key:
            print(f"{pad}{key}={data[key]}")

    if indent == 0 and dict_keys:
        print()

    for key in dict_keys:
        print(f"{pad}{key}={{")
        child_schema = _property_schema(schema, data, key)
        _print_config_by_schema(data[key], child_schema, indent + 1)
        print(f"{pad}}}")
        if indent == 0:
            print()

def _argparse_type_from_schema(sch: Dict[str, Any]):
    if not isinstance(sch, dict):
        return str

    t = sch.get("type")
    if t == "boolean":
        return strtobool
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "array" or t == "object":
        return json.loads
    if t == "string":
        return str

    for sub in _iter_subschemas(sch):
        sub_t = _argparse_type_from_schema(sub)
        if sub_t is not str:
            return sub_t

    return str

def _schema_choices(sch: Dict[str, Any]):
    if not isinstance(sch, dict):
        return None

    if "enum" in sch and isinstance(sch["enum"], list):
        return sch["enum"]

    for sub in _iter_subschemas(sch):
        choices = _schema_choices(sub)
        if choices is not None:
            return choices

    return None

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

class UnknownParameterError(ValueError):
    """Exception raised for unknown parameters."""
    def __init__(self, parameters, message="Unknown parameter name(s). Please check the spelling of the inputs above and try again."):
        self.parameters = parameters
        super().__init__(message)

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

        # "meta" args
        self.parser.add_argument("--config", default="config.json", type=str)
        self.parser.add_argument("--schema", default="configuration-schema.json", type=str)
        self.parser.add_argument("--check_only", default=False, type=strtobool)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def _add_schema_cli_arguments(self, schema: Dict[str, Any]) -> None:
        if self._schema_args_added:
            return

        leaf_paths = {}
        for path, prop_schema in _iter_schema_leaf_args(schema):
            if path not in leaf_paths:
                leaf_paths[path] = prop_schema

        leaf_name_counts = Counter(path[-1] for path in leaf_paths.keys())

        for path, prop_schema in leaf_paths.items():
            # Use plain leaf name when unique, otherwise dotted path
            if leaf_name_counts[path[-1]] == 1:
                arg_name = path[-1]
            else:
                arg_name = ".".join(path)

            dest = "__".join(path)

            kwargs = {
                "dest": dest,
                "default": None,
                "type": _argparse_type_from_schema(prop_schema),
            }

            choices = _schema_choices(prop_schema)
            if choices is not None and all(not isinstance(c, (dict, list)) for c in choices):
                kwargs["choices"] = choices

            self.parser.add_argument(f"--{arg_name}", **kwargs)
            self._cli_dest_to_path[dest] = path

        self._schema_args_added = True

    def _apply_cli_overrides(self, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        d = vars(args)

        for dest, path in self._cli_dest_to_path.items():
            if d.get(dest) is not None:
                _set_nested(cfg, path, d[dest])

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
                raise UnknownParameterError(unknown, message="Unknown parameter name(s) in CLI. Please check the spelling of the inputs above and try again.")

            # layer + apply schema defaults
            cfg = _to_layered_config(raw_cfg, schema)
            cfg = self._apply_cli_overrides(cfg, args)
            cfg = _sync_enabled_flags(cfg, schema)
            cfg = apply_defaults(schema=schema, instance=cfg)

            # validate final config
            validate(instance=cfg, schema=schema)

            bad = find_unknown_fields(schema, cfg)
            if bad:
                raise UnknownParameterError(bad, message="Unknown parameter name(s) in config. Please check the spelling of the inputs above and try again.")

            # path checks / postprocessing (same logic you already had)
            if cfg.get("model_params", {}).get("data_partitions_file", "") not in ("", None):
                validate_file_path(cfg["model_params"]["data_partitions_file"])
            validate_dir_path(cfg["output_dir"])
            validate_dir_path(cfg["data_dir"])

            # ensure batch_divisor size aligns with data size
            validate_data_size(cfg["data_dir"], cfg["model_params"]["batch_divisor"])

            # your equalization logic
            if cfg["model_params"]["partition_id"] > cfg["model_params"]["num_partitions"]:
                raise ValueError("partition_id must be <= num_partitions.")

            if cfg.get("federated", {}).get("federated_enabled", False):
                fed = cfg["federated"]
                print(f"Normalizing min_available_clients, min_evaluate_clients, and min_fit_clients to their minimum value.")
                if not (fed["min_available_clients"] == fed["min_evaluate_clients"] == fed["min_fit_clients"]):
                    min_val = min(fed["min_available_clients"], fed["min_evaluate_clients"], fed["min_fit_clients"])
                    fed["min_available_clients"] = min_val
                    fed["min_evaluate_clients"] = min_val
                    fed["min_fit_clients"] = min_val

                if fed["min_fit_clients"] > cfg["model_params"]["num_partitions"]:
                    raise ValueError("min_*_clients must be <= num_partitions.")

            if cfg.get("dp", {}).get("dp_enabled", False) and cfg["dp"].get("opacus_secure_mode", False):
                cfg["dp"]["opacus_secure_mode"] = False
        except UnknownParameterError as e:
            print(e)
            for elem in e.parameters:
                if "--" in elem:
                    print(f"{elem[2:]}\t", end="")
                else:
                    print(f"{elem}\t", end="")
            print()
            exit(1)
        except FileNotFoundError as e:
            print(e)
            exit(1)
        except json.JSONDecodeError as e:
            print(f"An error occured while parsing the config file.", end = "")
            print(f"Please check the following issue(s): {e.message}.")
            exit(1)
        except ValidationError as e:
            print(f"An error occured during configuration validation.", end = "")
            print(f"Please check the following issue(s): {e.message}.")
            exit(1)
        except ValueError as e:
            print(e)
            exit(1)
        

        self.args = ConfigArgs(schema=schema, **cfg)
        return self.args
