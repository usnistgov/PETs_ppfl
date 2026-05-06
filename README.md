# Privacy-Enhancing Technologies (PETs) Testbed

Welcome to the NIST genomics PETs testbed (beta version). This testbed aims to provide tools that help you evaluate the efficacy of different data privacy technologies on your genomics data.

# Table of Contents
1. [Currently Supported Capabilities](#current)
2. [Currently In-Progress Capabilities](#future_supported)
3. [Setting Up the Testbed](#setup)
4. [Running the Testbed](#running)
5. [Modifying the Parameters](#params)
6. [Output](#output)
7. [Running Regression Tests](#regression)
8. [References](#refs)

## Currently Supported Capabilities <a name="current"></a>

Throughout the beta development process, these capabilities are expected to expand. Feedback on additional features and discovered issues is welcome and encouraged.

### Data
- Testing using any dataset provided in the `data/` folder (or any other folder specified by the --data_dir CLI or data_dir field in the configuration file)

### Privacy and Training
- Applying differential privacy (DP) to data prior to training through the Opacus framework
- Centralized training for models (CNN models only)
- Federated training (CNN models only)

### User Interaction
- Modifying model and privacy parameters through the provided `config.json` file
- Modifying model and privacy parameters through command-line arguments
- Validation of provided parameters against the provided JSON schema
- Validation of provided file paths against the currently visible directory
- Generation of machine-readable and JSON reports, with some values printed out to the terminal, in timestamped directories

_*Please note that bring-your-own-data partition files have not been tested yet. There may be unexpected behaviors. Also note that users are expected to have already preprocessed their data prior to uploading it to the `data/` folder.*_

## Currently In-Progress Capabilities <a name="future_supported"></a>
- Generation of human-readable and machine-readable reports
- Improvement of report organization
- Support for setting the `opacus_secure_mode` parameter to true in DP

## Setting Up the Testbed <a name="setup"></a>

1. Download the zip file containing the data and code.
   1. If using the provided test data, ensure the following files exist in the `data/Oil_binned5` directory:
      1. `Oil_QTL_ho_pheno.dat`
      2. `Oil_QTL_ho_vcf.dat`
      3. `Oil_QTL_ohe_map.dat`
      4. `Oil_QTL_ohe.dat`
      5. `Oil_QTL_pheno_bins.dat`
      6. `Oil_QTL_tt_pheno.dat`
      7. `Oil_QTL_tt_vcf.dat`
   2. If using your own data:
      - Please ensure that all data files are of type `.dat`. It is also expected that the endings of the data files match the provided test data. For example, this testbed assumes the data file endings are:
         1. `_ho_pheno.dat`
         2. `_ho_vcf.dat`
         3. `_ohe.dat`
         4. `_tt_pheno.dat`
         5. `_tt_vcf.dat`
      - Please also remember to update the `data_dir` variable either through the command line or through the provided configuration file.
2. Transfer the files to the beta machine.
3. Download or validate a Python 3.10 installation (other versions may result in errors with the `torch` library) via the `python -V` or `python3 -V` commands.
   1. Run the command `python -V` to check the installed version.
   2. If the wrong version of Python is installed, or if Python is not installed, try:
      1. Running the `setup.sh` bash script, which downloads Python 3.10 from the deadsnakes repository
   3. Run the command `python3.10 -V` to confirm a successful installation.
4. `cd` into the directory where the downloaded project is located. You should see folders such as `data`, `dpcnn_opacus`, etc. Create a virtual environment with the command `python3.10 -m venv .venv`.
5. Confirm that the `.venv` directory was created by running `ls -la`.
6. Activate the virtual environment. On success, `(.venv)` should be prepended to your shell prompt.
   1. On Linux: `source ./.venv/bin/activate`
7. Install packages with the commands `python -m pip install --upgrade pip` and then `pip install -r requirements.txt` while in the project root directory.
   1. Note that if there are errors with the `torch` package, you may need to uninstall the packages and reinstall them using `pip uninstall -y -r .\requirements.txt`
8. Change directories to the `dpcnn_opacus` directory (note that this is necessary due to relative paths).
9. Run the command: 
```bash
python run.py
```
   - Note that this will use the `config.json` file for input parameters. These inputs are validated through the `configuration-schema.json` file.

## Running the Testbed <a name="running"></a>

### Hello World for the testbed
Once you have your environment set up, the most basic way to run the testbed is to navigate into the `dpcnn_opacus` folder and simply run:
```bash 
python3.10 run.py

# Or to just validate parameters:
python3.10 run.py --check_only
```
If `python3.10` is the only Python version in your environment, you may be able to run `python run.py` instead. This will load the default parameter values from the schema and run a simple federated learning workflow with differential privacy.

### Adjusting testbed parameters from the configuration file
One way to modify the parameters of the testbed is to edit the `config.json` file with new values. To do this, simply open the JSON file and add the field for the parameter you want to modify. You do not need to mirror the layered structure of the schema. Please enter parameters as key-value pairs, such as `"epochs": 20` or `"data_dir": "../data/my_data"`. Please note that the configuration file will be validated at runtime, so if there are invalid values or incorrect types, you will be able to correct them before running the testbed. See [the table below](#parameter-definitions) for details on each parameter you can modify.

### How to create your own JSON file
If you want to create a separate JSON file for a test case to help record the inputs, first make a copy of `config.json`, then rename it, and finally change the values to match your desired experiment. You can tell the testbed to use this configuration file through the `--config my_config.json` command-line flag. If you want to make a copy of and edit `configuration-schema.json`, you can also pass in the schema through the `--schema my_schema.json` command-line flag.

### Using the command line to adjust parameter values
If you want to modify a value for a single run, or see whether a parameter input is valid, and do not want to modify the `.json` configuration file, then you can use the command line to modify one or more parameters. To do this, treat the desired variables as flags and use the format `--foo bar`, which would set the `foo` parameter's value to `bar`. One exception to this is the `--check_only` flag, which does not expect any values and just turns on the parameter validation mechanism.

Modifying parameters in this way will not affect the contents of the configuration file.

Please note that some variables are only set through the command line. These variables are `--config` (to tell the testbed to use a configuration file other than `config.json`), `--schema` (to tell the testbed to use a JSON schema file other than `configuration-schema.json`), and `--check_only` (which, when passed, tells the testbed to only check whether the parameter values given are supported).

## Modifying the Parameters <a name="params"></a>

The default parameter values are placed in the `configuration-schema.json` file. Parameter values can be changed either directly through that file or through the command line. Parameter values modified through the command line will not change the contents of the `config.json` file.

Regardless of how the parameter values are input, the values will be validated against the types and ranges specified by `configuration-schema.json`. Any provided file paths, such as a data partition file or an output directory, must exist prior to running the testbed to avoid early termination. If parameter errors occur, error messages and suggestions should be printed to the terminal.

### Parameter Definitions

| Parameter | Description | Type | Limits / Allowed Values |
|---|---|---|---|
| `model_type` | The model family to run | string | `"dpcnn"`, `"cnn"`, `"xgboost"` |
| `num_cpus` | The number of CPUs available to the run | integer | min: 1, max: 100 |
| `num_gpus` | The number of GPUs available to the run | integer | min: 0, max: 100 |
| `num_rounds` | The number of rounds of federated learning | integer | min: 1, max: 100 |
| `min_fit_clients` | The minimum number of clients that must contribute to the federated training rounds | integer | min: 1, max: 100 |
| `min_available_clients` | The minimum number of clients that must be connected to the server for federated learning to begin | integer | min: 1, max: 100 |
| `min_evaluate_clients` | The minimum number of clients that must participate in a federated evaluation round for the round to be successful | integer | min: 1, max: 100 |
| `data_partitions_file` | A path to a file that contains the data partitions | string | — |
| `partitioner_type` | The type of partitioner to use if a data partition file was not provided | string | `"uniform"`, `"linear"`, `"square"`, `"exponential"` |
| `num_partitions` | The number of data partitions. If this value is less than `min_fit_clients`, `min_available_clients`, or `min_evaluate_clients`, then those values may be lowered accordingly. | integer | min: 1, max: 100 |
| `partition_id` | Partition ID used for the current client | integer | min: 0, max: 100 |
| `client_id` | Client ID used for the current client | integer | min: 0, max: 100 |
| `seed` | The seed used to randomize training and testing | integer | min: 1, max: 1000 |
| `epochs` | The number of model training epochs. An epoch is one full pass through a training dataset. | integer | min: 1, max: 100 |
| `batch_divisor` | The divisor used to determine the number of batches (`num_batches = dataset_size / batch_divisor`) | integer | min: 1 |
| `n_models` | The number of models to train. This can be useful if simulating federated learning locally. | integer | min: 1, max: 100 |
| `test_fraction` | The fraction of the dataset to set aside for testing | number | exclusive min: 0, exclusive max: 1 |
| `learning_rate` | Sets the model’s learning rate. This defines how much a model changes at each iteration. | number | exclusive min: 0, exclusive max: 1 |
| `weight_decay` | Sets the model’s weight decay. This is a regularization method that penalizes high weights. | number | exclusive min: 0, exclusive max: 0.1 |
| `optimizer` | The optimizer is responsible for adjusting model parameters based on the value of the loss function | string | `"sgd"`, `"adamax"` |
| `opacus_secure_mode` | Turns on cryptographically secure differential privacy when set to `True` | boolean | — |
| `epsilon` | Determines the amount of privacy added to the data | number | exclusive min: 0, exclusive max: 50 |
| `delta` | Measures the chance of a data breach. It defines the probability that the noise does not add sufficient privacy. | number | min: 0, max: 1 |
| `max_grad_norm` | Clips the gradients to be under this maximum before adding noise | number | min: 0, max: 100 |
| `accuracy_tolerance` | Error tolerance used to declare a prediction correct | number | min: 0, max: 1 |
| `check_only` | A flag to turn on the check-only feature, which ensures all parameter values are within the appropriate range and have the correct type. When `true`, the testbed will not run. Execution will stop after the parameters are validated. | boolean | — |
| `config` | A way to specify a different configuration file path | string | — |
| `data_dir` | The path to the intended data files to run the testbed on | string | — |
| `train_method` | The XGBoost training method | string | `"bagging"`, `"cyclic"` |
| `centralised_eval` | Whether centralized evaluation is enabled for XGBoost | boolean | — |
| `scaled_lr` | Whether scaled learning rate behavior is enabled for XGBoost | boolean | — |

### Parameter settings
To run centralized training from `run.py` (`centralized_train.py` may also be used directly, but note that the configuration file will not be read in that case), you should be able to use the following parameter values. Please note that the output may still look like federated learning training rounds, even though it is only one client, one round, and one data partition:

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

There is currently no way to fully turn off DP. This will be added in a later beta release.

## Understanding the Output <a name="output"></a>

### Files
There are three types of output files: `.npz` files, `.json` files, and `.torch` files. Within the `.npz` files, there is metadata on the model's global and client/round based performance. This is captured in the `.json` file format as well for human-readable purposes. Schemas can be found for this in the [schemas](schemas) directory, and sample outputs can be found in the [sampleReports](sampleReports) directory. Additionally in the `.json` file format is a file containing the input parameters used for a run, following the `configuration-schema.json` file, located at [dpcnn_opacus/configuration-schema.json](dpcnn_opacus/configuration-schema.json) at the time of writing. In the .torch files, there are model weights that can be loaded for further inference with the trained model.

### Metric Definitions

#### Client/Round Outputs
| Metric | Description | Type | 
|---|---|---|
| `created on` | The datetime that the report was generated | string |
| `model id` | The id of the model being developed | integer |
| `round numer` | The current round that the metrics are reporting on | integer |
| `partitions file` | The file used to partition the data for training | string |
| `train accuracy` | Training accuracy for this client and round | number |
| `test accuracy` | Test accuracy for this client and round | number |
| `train mean squared error` | Train MSE for this client and round | number |
| `test mean squared error` | Test MSE for this client and round | number |
| `train loss` | Train loss for this client and round | number |
| `test loss` | Test loss for this client and round | number |
| `train indices` | Indices used for training | integer array |
| `test indices` | Indices used for testing | integer array |
| `train accuracy per epoch` | Accuracy over epochs for training | number array |
| `test accuracy per epoch` | Accuracy over epochs for testing | number array |
| `train mse per epoch` | MSE over epochs for training | number array |
| `test mse per epoch` | MSE over epochs for testing | number array |
| `losses per epoch` | Loss values for each epoch in this client round | number array |
| `epsilon per epoch` | The epsilon values for each epoch in this client round | number array |
| `train predictions` | The predicted values for the training set | number array |
| `test predictions` | The predicted values for the test set | number array |
| `hyperparameters` | The hyperparameters the client used during its training process | object of the following properties: |
| `hyperparameters.learning rate` | Sets the model’s learning rate. This defines how much a model changes at each iteration | number | 
| `hyperparameters.weight decay` | Sets the model’s weight decay. This is a regularization method that penalizes high weights | number | 
| `hyperparameters.batch divisor` | The divisor to determine the number of batches (num_batches = dataset_size/batch_divisor) | integer |
| `hyperparameters.epochs` | The number of model training epochs. An epoch is one full pass through a training dataset | integer |
| `hyperparameters.seed` | The seed used to randomize training/testing | integer | 
| `hyperparameters.test fraction` | The fraction of the data set to set aside for testing | number |
| `hyperparameters.accuracy tolerance` | Error tolerance to declare prediction as correct | number |
| `hyperparameters.optimizer` | The optimizer is responsible for adjusting model parameters based on the value of the loss function | string |
| `hyperparameters.epsilon` | Determines the amount of privacy added to the data. | number |
| `hyperparameters.delta` | Measures the chance of a data breach. It defines the probability of the noise not adding sufficient privacy | number |
| `hyperparameters.max grad norm` | Clips the gradients to be under this maximum before adding noise | number |

#### Global Outputs
| Metric | Description | Type | 
|---|---|---|
| `created on` | The datetime that the report was generated | string |
| `loss per round` | Loss values across rounds | number array |
| `accuracy per round` | Accuracy values across rounds | number array |
| `mse per round` | MSE values across rounds | number array |

## Regression Testing <a name="regression"></a>

For regression testing, I am using a Python library called `pytest`. This helps automate the testing process. There are various test cases described in the `test_cli_config_regression.py` script. In this regression testing, it is only checking whether the parameter inputs are valid. To run the regression tests, ensure you are in the `dpcnn_opacus` folder, and if using a virtual environment, make sure it is active. Also ensure you have `pytest` installed in your environment; it is now listed in `requirements.txt`. Then, run:
```bash
python3.10 -m pytest -q
``` 
This will loop through every single test case with a progress tracker at the bottom. Any failed tests will be printed at the end.

These regression tests do have an end-to-end run, but it is skipped by default due to the time it takes. If you want to include the end-to-end run in the regression testing, run:
```bash
python3.10 -m pytest -q --run-e2e
```

To manually go through each test case using `pytest`, first gather a list of all possible tests. I recommend recording it in a `.txt` file for easy lookup by running 
```bash
python3.10 -m pytest --collect-only -q > test_list.txt
```
Then, identify the test you want to run, for example `test_cli_config_regression.py::test_run_py_regressions[t12a]`. To run that individual test, use:

```bash
python3.10 -m pytest test_cli_config_regression.py::test_run_py_regressions[t12a]
```

### Adding additional regression tests (for developers)
Between the helper definitions near the top of `test_cli_config_regression.py` and the `CASES` list, the different regression test cases are defined. Each test is defined through a `Case()` instance and added to the list via `pytest.param()`. Helper functions at the top of the file make it easier to modify the parameters used in the regression tests and compare expected versus actual output. To modify the DPCNN parameters for a regression test, use the `_base_with()` helper function. To modify the CNN parameters for a regression test, use the `_cnn_base_with()` helper function. To modify the XGBoost parameters for a regression test, use the `_xgb_base_with()` helper function.

Here is the class definition of `Case()`:

```python
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
```

The most important pieces of information when populating `Case()` are:

1. The name. This is a descriptor that allows someone looking through the regression test code, or the output of the regression tests, to know what the test is trying to accomplish.
2. The config. This is usually passed in using `_base_with()`, `_cnn_base_with()`, or `_xgb_base_with()`. `raw_config_text` should be used if you want to provide a full JSON file as a string to the test, for example if you have a very specific configuration file you want to test that is not easily expressible with the helper functions. These configuration files are typically written to a temporary pytest directory such as `/tmp/pytest-of-[username]/pytest-[num]/test_run_py_regressions_[test_id]`.
3. The command-line arguments. These can be helpful for testing overrides.
4. The acceptable exit codes. This tells the test whether you expect this case to fail or not. If you expect the regression test to fail and it does fail with one of the provided error codes, then the overall regression test passes.
5. Match or non-match output expectations. This is helpful when checking error handling. You can specify what you expect to see, or not see, in the output.

Please also note that `pytest.param()` expects a `Case()` instance and an `id` string to be provided.

Here are two examples of regression tests with the minimum information that should be provided. The first is an example of a regression test that is expected to fail, with an error message that the output must contain, and the second is an example of a regression test that is expected to pass:

```python
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
)
```

## References <a name="refs"></a>

### Soybean Trait Prediction Research

Published paper: [Machine learning models outperform deep learning models, provide interpretation and facilitate feature selection for soybean trait prediction](https://bmcplantbiol.biomedcentral.com/articles/10.1186/s12870-022-03559-z)

Jupyter notebooks for the soybean paper (GitHub): [Soybean_Trait_Prediction](https://github.com/mitchgill16/Soybean_Trait_Prediction)

#### Datasets

Dataset host site: https://data.pawsey.org.au/projects/

Unfortunately, you cannot share a link that goes directly to the folders containing the dataset CSV files, so you will have to navigate through the UI's folder structure to `/NGS Analysis Results/shortTerm/mgill/DL/holdout_and_equivalent_merged_1pcnt_removed`.

In that folder, you will see the `holdout` and `train_test` datasets named with the feature as a prefix, for example `"FlC_"` for flower color.
