# Privacy-Enhancing Technologies (PETs) Testbed

Welcome to the NIST genomics PETs testbed (beta version). This testbed aims to provide you with tools that help you evaluate the efficacy of different data privacy technologies on your genomics data.

# Table of Contents
1. [Currently Supported Capabilities](#current)
2. [Currently In-Progress Capabilities](#future_supported)
3. [Setting Up the Testbed](#setup)
4. [Running the Testbed](#running)
5. [Modifying the Parameters](#params)

## Currently Supported Capabilities <a name="current"></a>
Throughout the beta development process, these capabilities are expected to expand. Feedback on additional features and discovered issues is welcome and encouraged.

### Data
- Testing using _any*_ dataset provided in the `data/` folder

### Privacy and Training
- Applying differential privacy (DP) to data prior to training through the Opacus framework
- Centralized training for models (CNN models only)
- Federated training (CNN models only)

### User Interaction
- Modifying model and privacy parameters through the provided `config.json` file
- Modifying model and privacy parameters through command-line arguments
- Validation of provided parameters against the provided JSON schema
- Validation of provided file paths against the currently visible directory
- Generation of machine-readable reports, with some values printed out to the terminal

_\* Please note that bring-your-own data partitions files have not been tested yet. There may be unexpected behaviors. Also note that users are expected to have already preprocessed their data prior to uploading it to the `data/` folder._

## Currently In-Progress Capabilities <a name="future_supported"></a>
- Generation of human-readable and machine-readable reports
- Improvement of report organization
- Support for the `opacus_secure_mode` parameter being `true` in DP

## Setting Up the Testbed <a name="setup"></a>

1. Download the zip file containing the data and code.
   1. If using the provided test data, ensure the following files exist in the `data/Oil_binned5` directory. The included files will be:
      1. `Oil_QTL_ho_pheno.dat`
      2. `Oil_QTL_ho_vcf.dat`
      3. `Oil_QTL_ohe_map.dat`
      4. `Oil_QTL_ohe.dat`
      5. `Oil_QTL_pheno_bins.dat`
      6. `Oil_QTL_tt_pheno.dat`
      7. `Oil_QTL_tt_vcf.dat`
   2. If using your own data:
      - Please ensure that all data files are of type `.dat`. It is also expected that the endings of the data match the provided test data. For example, this testbed assumes the endings of the data files are:
         1. `_ho_pheno.dat`
         2. `_ho_vcf.dat`
         3. `_ohe.dat`
         4. `_tt_pheno.dat`
         5. `_tt_vcf.dat`
      - Please also remember to update the `data_dir` variable either through the command line or through the provided configuration file.
2. Transfer the files to the beta machine.
3. Download or validate a Python 3.10 installation (other installations result in errors with the `torch` library) via the `python -V` or `python3 -V` commands.
   1. Run the command `python -V` to check the installation version.
   2. If the wrong version of Python, or no Python installation, is installed, try:
      1. Running the `setup.sh` bash script, which downloads Python 3.10 from the deadsnakes repository
   3. Run the command `python3.10.exe -V` to confirm a successful installation.
4. `cd` into the directory where the downloaded project is located. You should see folders for `data`, `dpcnn_opacus`, etc. Create a virtual environment with the command `python3.10.exe -m venv .venv`.
5. Confirm that the `.venv` directory is created by running `ls -la`.
6. Activate the virtual environment. On success, `(.venv)` should be prepended to your shell.
   1. On Linux: `source ./.venv/bin/activate`
7. Install packages with the commands `python -m pip install --upgrade pip` and then `pip install -r requirements.txt` while in the home codebase directory.
   1. Note that if there are errors with the `torch` package, you may need to uninstall the packages and reinstall them using `pip uninstall -y -r .\requirements.txt`
8. Change directories to the `dpcnn_opacus` directory (note that this is necessary due to relative pathing).
9. Run the command `python run.py`.
   1. Note that this will use the `config.json` file for input parameters. These inputs are validated through the `configuration-schema.json` file.

## Running the Testbed <a name="running"></a>

### Hello World for the testbed
Once you have your environment set up, the most basic way to run the testbed is to navigate into the `dpcnn_opacus` folder and simply run `python3.10 run.py`. If `python3.10` is the only Python version in your environment, you may be able to run `python run.py` instead. This will load the default parameter values from the schema and get a simple version of federated learning with differential privacy running.

### Adjusting testbed parameters from the configuration file
One way to modify the parameters of the testbed is to edit the `config.json` file to have new values. To do this, simply open the JSON file and add the field for the parameter you want to modify. You do not need to worry about layering your configuration file like the schema. Please enter parameters as key-value pairs, such as `"epochs": 20` or `"data_dir": "../data/my_data"`. Please note that the configuration file will be validated at runtime, so if there are any invalid values or incorrect types, you will be able to correct them prior to running the testbed. See the table below for details on each parameter you can modify.

### How to create your own JSON file
If you want to create a separate JSON file for a test case (to help record the inputs), first make a copy of the `config.json` file, then rename it, and finally change the values to match your desired experiment. You can tell the testbed to use this configuration file through the `--config my_config.json` command-line flag. If you want to make a copy of and edit the `configuration-schema.json`, you can also pass in the schema through the `--schema my_schema.json` command-line flag.

### Using the command line to adjust parameter values
If you want to modify a value for a single run (or see if a parameter input is valid) and do not want to modify the `.json` configuration file, then you can use the command line to modify one or more parameters. To do this, treat the desired variables as flags and use the following format: `--foo bar`, which would set the `foo` parameter's value to `bar`.

Modifying parameters in this way will not affect the contents of the configuration file.

Please note that some variables are only set through the command line. These variables are `--config` (to redirect the testbed to use a configuration file other than `config.json`) and `--check_only` (which, when set to `true`, tells the testbed to only see if the parameter values given are supported).

## Modifying the Parameters <a name="params"></a>
The default parameter values are placed in the `configuration-schema.json` file. Parameter values can be changed either directly through that file or through the command line. Parameter values modified by the command line will not change the contents of the `config.json` file.

Regardless of how the parameter values are input, the values will be validated against the types and ranges specified by the `configuration-schema.json`. Any provided file paths (e.g., a data partitions file or an output directory) must exist prior to running the testbed to avoid early termination of the testbed. If parameter errors occur, error messages and suggestions should be printed out to the terminal.

### Parameter Definitions

| Parameter | Description | Type | Limits / Allowed Values |
|---|---|---|---|
| `num_rounds` | The number of rounds of federated learning | integer | min: 1, max: 100 |
| `min_fit_clients` | The minimum number of clients that must contribute to the federated training rounds | integer | min: 1, max: 100 |
| `min_available_clients` | The minimum number of clients that must be connected to the server for federated learning to begin | integer | min: 1, max: 100 |
| `min_evaluate_clients` | The minimum number of clients that must participate in a federated evaluation round for the round to be successful | integer | min: 1, max: 100 |
| `data_partitions_file` | A path to a file that contains the data partitions | string | — |
| `partitioner_type` | The type of partitioner to use if a data partition was not provided | string | "uniform", "linear", "square", "exponential" |
| `num_partitions` | The number of data partitions. If this value is less than `min_fit_clients`, `min_available_clients`, or `min_evaluate_clients`, then those values will take the lower value. | integer | min: 1, max: 100 |
| `partition_id` | Partition ID used for the current client | integer | min: 0, max: 100 |
| `client_id` | Client ID used for the current client | integer | min: 0, max: 100 |
| `seed` | The seed used to randomize training/testing | integer | min: 1, max: 1000 |
| `epochs` | The number of model training epochs. An epoch is one full pass through a training dataset | integer | min: 1, max: 100 |
| `batch_divisor` | The divisor used to determine the number of batches (`num_batches = dataset_size / batch_divisor`) | integer | min: 1 |
| `n_models` | The number of models to train. This can be useful if simulating federated learning locally | integer | min: 1, max: 100 |
| `test_fraction` | The fraction of the dataset to set aside for testing | number | exclusive min: 0, exclusive max: 1 |
| `learning_rate` | Sets the model’s learning rate. This defines how much a model changes at each iteration | number | exclusive min: 0, exclusive max: 1 |
| `weight_decay` | Sets the model’s weight decay. This is a regularization method that penalizes high weights | number | exclusive min: 0, exclusive max: 0.1 |
| `optimizer` | The optimizer is responsible for adjusting model parameters based on the value of the loss function | string | "sgd", "adamax" |
| `opacus_secure_mode` | Turns on cryptographically secure differential privacy when set to `True` | boolean | — |
| `epsilon` | Determines the amount of privacy added to the data | number | exclusive min: 0, exclusive max: 50 |
| `delta` | Measures the chance of a data breach. It defines the probability of the noise not adding sufficient privacy | number | min: 0, max: 1 |
| `max_grad_norm` | Clips the gradients to be under this maximum before adding noise | number | min: 0, max: 100 |
| `accuracy_tolerance` | Error tolerance used to declare a prediction correct | number | min: 0, max: 1 |
| `check_only` | A flag to turn on the check-only feature, which ensures all parameter values are within the appropriate range and have the correct type. When `true`, the testbed will not run. Execution will stop after the parameters are validated. | boolean | — |
| `config` | A way to specify a different configuration file path | string | — |
| `data_dir` | The path to the intended data files to run the testbed on | string | — |

### Parameter settings
To run `centralized_training` in `run.py` (running with the `centralized_train.py` file is allowed, but note that the configuration file will not be read for it), you _should_ be able to use the following parameter values. Please note that the output will still look like federated learning training rounds, even though it is only one client, one round, and one data partition:

| Parameter | Value |
|---|---|
| `num_partitions` | 1 |
| `partition_id` | 0 |
| `client_id` | 0 |
| `min_fit_clients` | 1 |
| `min_available_clients` | 1 |
| `min_evaluate_clients` | 1 |
| `n_models` | 1 |
| `num_rounds` | 1 |

Example `config.json`:
```json
{
  "model_type": "dpcnn",
  "num_partitions": 1,
  "partition_id": 0,
  "client_id": 0,
  "min_fit_clients": 1,
  "min_available_clients": 1,
  "min_evaluate_clients": 1,
  "n_models": 1,
  "num_rounds": 1,
  "num_cpus": 12
}
```

There is currently no way to fully turn off DP. This will be added during a later beta release.

## Understanding the Output
There are two types of output files: `.npz` files and `.torch` files. Within the `.npz` files, there is metadata on the model's global and round-based performance. In the `.torch` files, there are model weights that can be loaded for further inference with the trained model. A human-readable output file is currently being worked on.

## Upcoming Features
- Batched experimentation
- Progress bars during runtime
- Improved reporting
- Tutorial notebook for meta-analysis
- Ability to run other models (besides Differential Privacy + CNN)