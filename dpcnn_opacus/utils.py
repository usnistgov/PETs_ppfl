import argparse
import types
from typing import List, Dict, Any
import numpy as np
from collections import Counter
import torch
import json
from jsonschema import validate, ValidationError
import argparse
from pathlib import Path
import os
from distutils.util import strtobool

class UnknownParameterError(ValueError):
    """Exception raised for unknown parameters."""
    def __init__(self, parameters, message="Unknown parameter name(s). Please check the spelling of the inputs above and try again."):
        self.parameters = parameters
        super().__init__(message)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def print_binned_counts(
    dataset: np.ndarray, indices: List[int] | np.ndarray, num_bins: int = 10
):
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


def flower_args_parser():
    """
    Parse arguments to define hyperparameter settings for centralized training.
    """
    parser = argparse.ArgumentParser()

    # server arguments
    parser.add_argument(
        "--num-rounds",
        default=3,
        type=int,
        help="Number of training rounds (default = 3).",
    )
    parser.add_argument(
        "--min-fit-clients",
        default=1,
        type=int,
        help="Minimum number of fit clients (default = 1).",
    )
    parser.add_argument(
        "--min-evaluate-clients",
        default=1,
        type=int,
        help="Minimum number of evaluation clients (default = 1).",
    )
    parser.add_argument(
        "--min-available-clients",
        default=1,
        type=int,
        help="Minimum number of available clients (default = 1).",
    )
    parser.add_argument(
        "--num-partitions",
        default=1,
        type=int,
        help="Number of partitions (default = 1)."
        "This is used when the data-partitions file is not provided.",
    )
    # client arguments
    parser.add_argument(
        "--partitioner-type",
        default="uniform",
        type=str,
        choices=["uniform", "linear", "square", "exponential"],
        help="Partitioner types (default = 'uniform').",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="Number of training epochs (default = 20).",
    )
    parser.add_argument(
        "--batch-divisor",
        default=5,
        type=int,
        help="Divisor to determine batch size (default = 5).",
    )
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
        "--learning-rate",
        default=0.003,
        type=float,
        help="Learning rate (default = 0.003).",
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
        "--output-dir",
        default=None,
        type=str,
        help="Output directory to save the trained models and metadata."
        "This should be a relative path from the parent directory of "
        "the centralized_train.py script.",
    )
    args = parser.parse_args()
    return args

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
#   override_cli(defaults)
#   input: a dictionary with string keys representing the values loaded in from the configuration json files
#   output: a dictionary with string keys that contain the configuration json file defaults, updated based on command line input
#   purpose: to allow the command line arguments to supersede the configuration json file
###
def override_cli(defaults: Dict[str, Any]):

    parser = argparse.ArgumentParser(exit_on_error=False)

    #Checks if this should only validate parameters or if it should run the testbed
    parser.add_argument("--check_only", default=False, type=strtobool)

    parser.add_argument("--num_gpus", type=int)
    parser.add_argument("--num_cpus", type=int)

    parser.add_argument("--num_rounds", type=int)
    parser.add_argument("--min_fit_clients", type=int)
    parser.add_argument("--min_available_clients", type=int)
    parser.add_argument("--min_evaluate_clients", type=int)
    parser.add_argument("--num_partitions", type=int)
    
    parser.add_argument("--data_partitions_file", type=str)
    parser.add_argument("--partitioner_type", type=str)
    parser.add_argument("--partition_id", type=int)
    parser.add_argument("--client_id", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_divisor", type=int)
    parser.add_argument("--n_models", type=int)
    parser.add_argument("--test_fraction", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--accuracy_tolerance", type=float)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--opacus_secure_mode", type=strtobool)
    parser.add_argument("--output_dir", type=str)

    args, unknown = parser.parse_known_args()

    if "--config" in unknown:
        remove_index = unknown.index("--config")
        unknown.pop(remove_index); unknown.pop(remove_index) #removes key and value pair
    
    if len(unknown) > 0:
        raise UnknownParameterError(unknown)

    for key, value in vars(args).items():
        if value is not None:
            defaults[key] = value

    # For pretty prints
    defaults["opacus_secure_mode"]=bool(defaults["opacus_secure_mode"])
    defaults["check_only"]=bool(defaults["check_only"])
    return defaults

###
#   validate_config_file(config_path, schema_path)
#   input: Two strings: one representing the configuration path and one representing the json schema path
#   output: A dictionary representing the input json file
#   purpose: Validate and load then configuration file and its schema. Then, apply the schema to the configuration file to validate parameter bounds
###
def validate_config_file(config_path: str, schema_path:str):
    validate_file_path([config_path, schema_path])

    with Path(config_path).open("r", encoding="utf-8") as f:
        config = json.load(f)
    with Path(schema_path).open("r", encoding="utf-8") as s:
        schema = json.load(s)

    try:
        validate(instance=config, schema=schema) #checks for missing fields

        defaults = override_cli(config) #override config file with command line inputs

        validate(instance=defaults, schema=schema) #validate with the schema only after the command line arguments are loaded in

    except ValidationError as err:  #adds in the offending parameter name
        param = ".".join(map(str, err.absolute_path)) or "<root>"
        raise ValueError(f"Invalid parameter '{param}': {err.message}") from err

    return defaults

###
#   json_args_parser(config_path, schema_path)
#   input: Two strings: one representing the configuration path and one representing the json schema path
#   output: An argparse parser
#   purpose: Load and validate all parameters from the configuration file. Also validate that passed paths exist.
###
def json_args_parser(schema_path="configuration-schema.json"):
    try:
        print()
        #parses the configuration path separately from everything else so that can be loaded first
        config_args = argparse.ArgumentParser(exit_on_error=False)
        config_args.add_argument("--config", default="config.json", type=str)
        args, _ = config_args.parse_known_args()

        defaults = validate_config_file(args.config, schema_path)

        parser = argparse.Namespace(**defaults)
        print("Configuration file validated against JSON schema")

        if not parser.data_partitions_file == "":
            validate_file_path(parser.data_partitions_file)
            print("Data partitions file successfully validated")
        
        validate_dir_path(parser.output_dir)
        print("Output directory successfully validated")

        ## Handles a current issue with opacus_secure_mode
        if parser.opacus_secure_mode:
            print("Warning: \"opacus_secure_mode\" not behaving as expected. Reverting back to opacus_secure_mode=false.")
            parser.opacus_secure_mode=False
            #Needs the torchcsprng package, but there are issues installing that for python3.10.
            #To do: investigate further
        print()

    except ValidationError as e:
        print(f"Configuration parameters failed validation. This could be due to a missing parameter or out-of-bounds value.")
        print(e.message)
        exit(1)
    except UnknownParameterError as e:
        print(f"An unknown parameter was encountered in the command line. Please check the spelling and try again for:")
        for elem in e.parameters:
            if "--" in elem:
                print(f"{elem}\t", end="")
        print()
        exit(1)
    except FileNotFoundError as e:
        print(e)
        print(1)
    except ValueError as e:
        print(f"Parameter value error occured. Please check the following issue and try again\n{e}")
        exit(1)
    except argparse.ArgumentError as e:
        print(f"Configuration parameters failed validation. Please check the following issue and try again.\n{e}")
        exit(1)
    except SystemExit as e:
        print(f"An error occured, likely related to command line inputs. Please check the spelling of the inputs above and try again.")
        exit(1)
    except OSError as e:
        print(f"Ran in to an unexpected error. Please try again.\n{e.message}")
        exit(1)
    return parser
