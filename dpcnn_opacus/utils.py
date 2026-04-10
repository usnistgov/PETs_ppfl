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

from jsonschema import validate, ValidationError

TOP_KEYS = {"model_type", "num_cpus", "num_gpus", "output_dir", "data_dir"}
FED_KEYS = {"num_rounds", "min_fit_clients", "min_available_clients", "min_evaluate_clients", "n_models", "federated_enabled"}
DP_KEYS = {"opacus_secure_mode", "epsilon", "delta", "max_grad_norm", "dp_enabled"}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


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


def centralized_args_parser():
    """
    Parse arguments to define hyperparameter settings for centralized training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed used for train/test splitting (default = 42).",
    )
    parser.add_argument(
        "--test-fraction",
        default=0.2,
        type=float,
        help="Test fraction for train/test splitting (default = 0.2).",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="Number of training epochs (default = 20).",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.003,
        type=float,
        help="Learning rate (default = 0.003).",
    )
    parser.add_argument(
        "--batch-divisor",
        default=40,
        type=int,
        help="Divisor to determine batch size (default = 40).",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.0001,
        type=float,
        help="Weight decay constant (default = 0.0001).",
    )
    parser.add_argument(
        "--accuracy-tolerance",
        default=0.1,
        type=float,
        help="Error tolerance to declare prediction as correct "
        "(default = 0.1). This is used for computing the "
        "accuracy of the model.",
    )
    parser.add_argument(
        "--data-partitions-file",
        default=None,
        type=str,
        help=f"Path to the data partitions file (default = {None}).\n"
        "If not used, then the data is split into n_models equal parts."
        "If data-partitions file is provided, "
        "then the number of models (n_models) "
        "to train is equal to the number of "
        "data partitions available in the file. In the data partitions, "
        "each key is a client id and value for each key is a list of "
        "dataset indices to be used for that client.",
    )
    parser.add_argument(
        "--n-models",
        default=1,
        type=int,
        help="Number of models to train (default = 1). "
        "Data is split into n_models equal parts."
        "This is used when the clustered-indices file is not found.",
    )
    parser.add_argument(
        "--optimizer",
        default="sgd",
        type=str,
        choices=["sgd", "adamax"],
        help="Optimizer to use sgd or adamax (default = sgd).",
    )
    parser.add_argument(
        "--opacus-secure-mode",
        default=False,
        type=bool,
        help="Use Opacus secure mode. It is set to false by default for "
        "faster experimentation (default = False).",
    )
    parser.add_argument(
        "--epsilon",
        default=1.0,
        type=float,
        help="Privacy parameter: epsilon (default = 1.0).",
    )
    parser.add_argument(
        "--delta",
        default=1e-5,
        type=float,
        help="Privacy parameter: delta (default = 1e-5).",
    )
    parser.add_argument(
        "--max-grad-norm",
        default=1.0,
        type=float,
        help="Privacy parameter: max grad norm (default = 1.0)."
        "This clips the gradients to be under this value before "
        "applying noise.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Output directory to save the trained models and metadata."
        "This should be a relative path from the parent directory of "
        "the centralized_train.py script.",
    )
    args = parser.parse_args()
    return args


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


def _to_layered_config(flat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    # Top-level
    for k in TOP_KEYS:
        if k in flat_cfg:
            cfg[k] = flat_cfg[k]

    # Ensure required model_type exists if caller forgot
    cfg.setdefault("model_type", "dpcnn")

    # Federated group
    fed: Dict[str, Any] = {"federated_enabled": False}
    for k in FED_KEYS:
        if k in flat_cfg:
            fed[k] = flat_cfg[k]
    if len(fed) > 1 or "federated_enabled" in flat_cfg:
        cfg["federated"] = fed
    else:
        # Keep federated present for tests that validate its parameters
        cfg["federated"] = fed

    # DP group
    dp: Dict[str, Any] = {"dp_enabled": False}
    for k in DP_KEYS:
        if k in flat_cfg:
            dp[k] = flat_cfg[k]
    if len(dp) > 1 or "dp_enabled" in flat_cfg:
        cfg["dp"] = dp
    else:
        # Keep dp present for tests that validate its parameters
        cfg["dp"] = dp

    # Model params group
    mp: Dict[str, Any] = {}
    for k, v in flat_cfg.items():
        if k in TOP_KEYS or k in FED_KEYS or k in DP_KEYS:
            continue
        mp[k] = v
    cfg["model_params"] = mp

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

def _allowed_keys_at_level(applicable: List[Dict[str, Any]]) -> Set[str]:
    allowed: Set[str] = set()
    for sch in applicable:
        props = sch.get("properties")
        if isinstance(props, dict):
            allowed.update(props.keys())
    return allowed

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
            prop_schemas = []
            for sch in applicable:
                props = sch.get("properties")
                if isinstance(props, dict) and k in props:
                    prop_schemas.append(props[k])
            if prop_schemas:
                bad.extend(find_unknown_fields({"allOf": prop_schemas}, v, p))

    return bad

class UnknownParameterError(ValueError):
    """Exception raised for unknown parameters."""
    def __init__(self, parameters, message="Unknown parameter name(s). Please check the spelling of the inputs above and try again."):
        self.parameters = parameters
        super().__init__(message)

class ConfigPipeline:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(exit_on_error=False)

        # "meta" args
        self.parser.add_argument("--config", default="config.json", type=str)
        self.parser.add_argument("--schema", default="configuration-schema.json", type=str)
        self.parser.add_argument("--check_only", default=False, type=strtobool)

        # top-level overrides
        self.parser.add_argument("--model_type", type=str)
        self.parser.add_argument("--num_cpus", type=int)
        self.parser.add_argument("--num_gpus", type=int)
        self.parser.add_argument("--output_dir", type=str)
        self.parser.add_argument("--data_dir", type=str)

        # federated overrides
        self.parser.add_argument("--federated_enabled", type=strtobool)
        self.parser.add_argument("--num_rounds", type=int)
        self.parser.add_argument("--min_fit_clients", type=int)
        self.parser.add_argument("--min_available_clients", type=int)
        self.parser.add_argument("--min_evaluate_clients", type=int)
        self.parser.add_argument("--n_models", type=int)

        # dp overrides
        self.parser.add_argument("--dp_enabled", type=strtobool)
        self.parser.add_argument("--opacus_secure_mode", type=strtobool)
        self.parser.add_argument("--epsilon", type=float)
        self.parser.add_argument("--delta", type=float)
        self.parser.add_argument("--max_grad_norm", type=float)

        # model_params overrides (flat CLI, nested in config)
        self.parser.add_argument("--data_partitions_file", type=str)
        self.parser.add_argument("--partitioner_type", type=str)
        self.parser.add_argument("--num_partitions", type=int)
        self.parser.add_argument("--partition_id", type=int)
        self.parser.add_argument("--client_id", type=int)
        self.parser.add_argument("--seed", type=int)
        self.parser.add_argument("--epochs", type=int)
        self.parser.add_argument("--batch_divisor", type=int)
        self.parser.add_argument("--test_fraction", type=float)   # NOTE: maps to model_params.test_frac
        self.parser.add_argument("--learning_rate", type=float)
        self.parser.add_argument("--weight_decay", type=float)
        self.parser.add_argument("--optimizer", type=str)
        self.parser.add_argument("--accuracy_tolerance", type=float)
        self.parser.add_argument("--train_method", type=str)
        self.parser.add_argument("--centralised_eval", type=strtobool)
        self.parser.add_argument("--scaled_lr", type=strtobool)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def _apply_cli_overrides(self, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        d = vars(args)

        # top-level
        for k in ("model_type", "num_cpus", "num_gpus", "output_dir"):
            if d.get(k) is not None:
                cfg[k] = d[k]

        # federated
        for k in ("federated_enabled", "num_rounds", "min_fit_clients", "min_available_clients", "min_evaluate_clients", "n_models"):
            if d.get(k) is not None:
                _set_nested(cfg, ("federated", k), d[k])

        # dp
        for k in ("dp_enabled", "opacus_secure_mode", "epsilon", "delta", "max_grad_norm"):
            if d.get(k) is not None:
                _set_nested(cfg, ("dp", k), d[k])

        # model_params
        model_params_keys = (
            "data_partitions_file",
            "partitioner_type",
            "num_partitions",
            "partition_id",
            "client_id",
            "seed",
            "epochs",
            "batch_divisor",
            "learning_rate",
            "weight_decay",
            "optimizer",
            "accuracy_tolerance",
            "train_method",
            "centralised_eval",
            "scaled_lr"
            # add more here as CLI grows
        )

        for k in model_params_keys:
            if d.get(k) is not None:
                _set_nested(cfg, ("model_params", k), d[k])

        # carry check_only (not part of schema, but you use it)
        cfg["check_only"] = bool(d.get("check_only", False))
        return cfg

    def parse(self) -> argparse.Namespace:
        try:
            args, unknown = self.parser.parse_known_args()
            if unknown:
                raise UnknownParameterError(unknown, message="Unknown parameter name(s) in CLI. Please check the spelling of the inputs above and try again.")

            # validate paths exist
            validate_file_path([args.config, args.schema])

            schema = self._load_json(args.schema)
            raw_cfg = self._load_json(args.config)

            # layer + apply schema defaults
            cfg = _to_layered_config(raw_cfg)
            cfg = apply_defaults(schema=schema, instance=cfg)

            # override with CLI
            cfg = self._apply_cli_overrides(cfg, args)

            bad = find_unknown_fields(schema, cfg)
            if bad:
                raise UnknownParameterError(bad, message="Unknown parameter name(s) in config. Please check the spelling of the inputs above and try again.")

            # validate final config
            validate(instance=cfg, schema=schema)

            # path checks / postprocessing (same logic you already had)
            if cfg.get("model_params", {}).get("data_partitions_file", "") not in ("", None):
                validate_file_path(cfg["model_params"]["data_partitions_file"])
            validate_dir_path(cfg["output_dir"])

            # your equalization logic
            if cfg.get("federated", {}).get("federated_enabled", False):
                fed = cfg["federated"]
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

        return argparse.Namespace(**cfg)
