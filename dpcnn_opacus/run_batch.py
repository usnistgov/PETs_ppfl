import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
from jsonschema import ValidationError
from tqdm import tqdm


def load_json_schema(file_path: Path) -> dict:
    """ Helper function to load and parse a JSON file

    Args:
        file_path: path to the JSON file

    Returns:
        Parsed JSON content as a dictionary

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if the file contains an invalid JSON
        RuntimeError: if an unexpected error occurs while reading the file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            schema = json.load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Schema file not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in schema file: {file_path}\n{e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error occured when reading schema file: {e}") from e

    return schema


def print_parameter_constraints(parameter_choice: str, parameter_requirements: dict) -> None:
    """ Print the validation constraints for a parameter

    Args:
        parameter_choice: name of the parameter being inspected
        parameter_requirements: metadata for the parameter
    """
    print(f"\n Parameter {parameter_choice} has the following contraints:")

    # Show the user relevant constraint fields based on parameter type
    if parameter_requirements["type"] == "number":
        print(
            f"\t Parameter {parameter_choice} has a type of number.",
            f"\n\t Exclusive minimum value is {parameter_requirements['exclusiveMinimum']}.",
            f"\n\t Exclusive maximum value is {parameter_requirements['exclusiveMaximum']}.",
        )
    elif parameter_requirements["type"] == "integer":
        print(
            f"\t Parameter {parameter_choice} has a type of integer.",
            f"\n\t Minimum value is {parameter_requirements['minimum']}.",
            f"\n\t Maximum value is {parameter_requirements['maximum']}.",
        )
    elif parameter_requirements["type"] == "string":
        print(
            f"\t Parameter {parameter_choice} has a type of string.",
            f"\n\t Accepted strings are {parameter_requirements['enum']}.",
        )


def get_parameters(schema: dict, batch_experimentation: dict) -> dict:
    """ Prompts the user for experiment settings and validate their selections

    1. Prompts the user to choose a model type
    2. Lists the available parameters for that model
    3. Prompts for values either as a range or as a comma-separated list
    4. Validates those values by calling the run.py

    Args:
        schema: configuration schema in JSON format.
        batch_experimentation: batch experimentation settings in JSON format

    Returns:
        Dictionary containing:
            - model_chpoce: selected model type
            - parameter_choice: selected parameter name
            - raw_parameters: raw values entered by the user or generated using the user inputs
            - valid_parameters: subset of values validated successfully

    Raises:
        KeyError: if the schema does not contain the expected model metadata
    """
    blacklist = batch_experimentation["parameter_blacklist"]

    print("\n Welcome to Batch Experimentation! \n")

    # Display the supported model types from the configuration schema
    print(" These are the model types that you can train:")
    model_types = schema["properties"]["model_type"]["enum"]
    for model_type in model_types:
        print(f"\t {model_type}")

    model_choice = input("\n Which model type would you like to train? ")
    if model_choice not in model_types:
        print(f" Invalid selection. {model_choice} is not a valid model type. Aborting.")
        sys.exit(1)

    # Display the parameters available to experiment for the chosen model type
    print("\n These are the parameters that you can run experiments on:")
    parameter_options = None
    for model_option in schema["allOf"][1]["oneOf"]:
        if model_option["properties"]["model_type"]["const"] == model_choice:
            parameter_options = model_option["properties"]["model_params"]["properties"]
            break

    if parameter_options:
        for parameter in parameter_options:
            if parameter not in blacklist:
                print(f"\t {parameter}")
    else:
        raise KeyError(" Unexpected error occured when obtaining parameter options")

    parameter_choice = input("\n Which parameter would you like to run an experiment with? ")
    if parameter_choice not in parameter_options:
        print(f" Invalid selection. {parameter_choice} is not a valid parameter. Aborting.")
        sys.exit(1)
    if parameter_choice in blacklist:
        print(
            f" Invalid selection. {parameter_choice} is a valid parameter but is not a parameter that can be run as an experiment. Aborting."
        )
        sys.exit(1)

    print(f"\n How would you like to specify experiment values for {parameter_choice}?")
    print("\t 1. Regular intervals (range)")
    print("\t 2. Manually specify values")

    choice = input("\n Enter 1 or 2: ").strip()
    parameter_requirements = parameter_options[parameter_choice]
    parameters: List[Any] = []

    if choice == "1":
        print_parameter_constraints(parameter_choice, parameter_requirements)

        # Range mode generates candidate values based on user inputs
        try:
            start_raw = input("\n Enter start value: ")
            end_raw = input("\n Enter end value: ")
            step_raw = input("\n Enter step size: ")

            # Convert inputs before comparing them.
            if parameter_requirements["type"] == "integer":
                start = int(start_raw)
                end = int(end_raw)
                step = int(step_raw)
            elif parameter_requirements["type"] == "number":
                start = float(start_raw)
                end = float(end_raw)
                step = float(step_raw)
            else:
                print(" Range mode is only supported for integer and number parameters. Aborting.")
                sys.exit(1)

            if end < start:
                print("\n Invalid input. The end value must be after the start value. Aborting.")
                sys.exit(1)

            # Validate the start/end bounds before generating the full range
            valid_range_values = validate_parameters(model_choice, parameter_choice, [start, end])
            if valid_range_values == [start, end]:
                current = start
                while current <= end:
                    parameters.append(current)
                    current += step
            else:
                print("\n Invalid range. Aborting.")
                sys.exit(1)

            print(f"\n Parameter values before validation: {parameters}")
        except ValueError as e:
            print(f"Invalid input: {e}. Aborting.")
            sys.exit(1)

    elif choice == "2":
        print_parameter_constraints(parameter_choice, parameter_requirements)

        # Manual mode accepts a comma-separated list of candidate values
        raw_parameters = input(f"\n Enter experiment values for {parameter_choice} separated by commas: ")
        parameters = [item.strip() for item in raw_parameters.split(",")]
    else:
        print("Invalid choice. Aborting.")
        sys.exit(1)

    print("\n Validating parameter values...")
    valid_parameters = validate_parameters(model_choice, parameter_choice, parameters)

    return_json = {
        "model_choice": model_choice,
        "parameter_choice": parameter_choice,
        "raw_parameters": parameters,
        "valid_parameters": valid_parameters,
    }

    return return_json


def validate_parameters(model_choice: str, parameter_choice: str, parameters: list) -> list:
    """ Validate candidate parameter values using run.py in check_only mode

    Args:
        model_choice: name of the model being tested
        parameter_choice: name of the parameter being tested
        parameters: candidate values to validate

    Returns:
        List of valid parameters
    """
    # Validate the default configuration without any CLI override
    result = subprocess.run(["python3", "run.py", f"--model_type={model_choice}", "--check_only"], capture_output=False, text=False, check=True)
    if result.returncode != 0:
        print(
            " Unexpected error occurred during validation of your configuration file. Please check your configuration file and try again. Aborting."
        )
        sys.exit(1)

    # Validate each candidate value by overriding the chosen CLI argument
    valid_parameters = []
    for parameter in parameters:
        try:
            result = subprocess.run(
                ["python3", "run.py", f"--model_type={model_choice}", f"--{parameter_choice}={parameter}", "--check_only"],
                capture_output=True,
                text=True,
                check=True
            )
            if result.returncode == 0:
                valid_parameters.append(parameter)
            else:
                print(f"\n {parameter} is not a valid value for {parameter_choice}. Skipping.")
        except subprocess.CalledProcessError:
            # ``subprocess.run`` only raises CalledProcessError when ``check=True`` is used,
            # but this branch is kept for defensive error handling.
            print(
                f" Unexpected error occurred during the validation of {parameter_choice} = {parameter}. Skipping."
            )
            continue

    if not valid_parameters:
        print(" No valid parameter values provided. Aborting.")
        sys.exit(1)

    return valid_parameters


def run_experiments(parameters: list) -> None:
    """ Executes run.py for each validated parameter

    Args:
        model_choice: name of the model being run
        parameter_choice: name of the parameter being swept
        parameters: validated values
    """
    model_choice = parameters["model_choice"]
    parameter_choice = parameters["parameter_choice"]
    raw_parameters = parameters["raw_parameters"]
    valid_parameters = parameters["valid_parameters"]

    print("\n Starting experiments...\n")

    # Create folder in output_dir to hold 

    current_dir = Path(__file__).resolve().parent
    config_path = current_dir / "config.json"

    config = load_json_schema(config_path)

    try:
        output_dir = config["output_dir"]
    except:
        schema_path = current_dir / "configuration-schema.json"
        schema = load_json_schema(schema_path)
        output_dir = schema["properties"]["output_dir"]["default"]

    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

    appended_output_dir = f"{output_dir}/batch-experiment-{timestamp}" 
    directory = Path(appended_output_dir)
    directory.mkdir(parents=True, exist_ok=False)

    print(f"Created directory: {directory}")

    # Create experiment-inputs.json file

    experiment_inputs_path = directory / "experiment_inputs.json"
    experiment_inputs_json = {
        "initiated_on": timestamp,
        "parameter": parameter_choice,
        "raw_parameter_values": raw_parameters,
        "validated_parameter_values": valid_parameters
    }

    with open(experiment_inputs_path, "w") as f:
        json.dump(experiment_inputs_json, f, indent=4)

    # tqdm provides a simple progress bar for the experiment sweep.
    for value in tqdm(valid_parameters, desc=" Running experiments", unit="exp"):
        tqdm.write(f" Running {parameter_choice} = {value}")

        try:
            subprocess.run(["python3", "run.py", f"--model_type={model_choice}", f"--{parameter_choice}={value}", f"--output_dir={appended_output_dir}"])
            tqdm.write(f" Finished {parameter_choice} = {value}")
        except subprocess.CalledProcessError as e:
            tqdm.write(f" Experiment failed for {parameter_choice} = {value}")
            tqdm.write(f" Error: {e}")
            continue


def main() -> None:
    current_dir = Path(__file__).resolve().parent

    # Load the main schema that describes the supported model configurations.
    schema_path = current_dir / "configuration-schema.json"
    schema = load_json_schema(schema_path)

    # Load the batch-experimentation settings, including any parameter blacklist.
    batch_experimentation_path = current_dir / "batch_experimentation.json"
    batch_experimentation = load_json_schema(batch_experimentation_path)

    parameters = get_parameters(schema, batch_experimentation)

    print("\n Final parameter values after validation:")
    for parameter in parameters["valid_parameters"]:
        print(f" - {parameter}")

    confirm = input("\n Proceed with experiments? (y/n): ").strip().lower()
    if confirm != "y":
        print("\n Aborted.")
        sys.exit(0)

    run_experiments(parameters)
    print("\n All experiments completed.")


if __name__ == "__main__":
    main()
