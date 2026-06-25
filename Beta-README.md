# Privacy-Enhancing Technologies (PETs) Testbed

Welcome to the NIST genomics PETs testbed (beta version). This testbed aims to provide tools that help you evaluate the efficacy of different data privacy technologies on your genomics data.

# Table of Contents
1. [Currently Supported Capabilities](#current)
1. [Currently In-Progress Capabilities](#future_supported)
1. [Setting Up the Testbed](#setup)
1. [Running the Testbed](#running)
1. [Modifying the Parameters](#params)
1. [Batch Experiments](#batch)
1. [Output](#output)
1. [Results Viewer](#viewer)
1. [Data References](#refs)
1. [Flower/Ray client, model, and logging behavior](#flwr)
1. [Ray Memory Optimization](#mem)

## Currently Supported Capabilities <a name="current"></a>

Throughout the beta development process, these capabilities are expected to expand. Feedback on additional features and discovered issues is welcome and encouraged.

### Data
- Testing using any dataset provided in the `data/` folder (or any other folder specified by the --data_dir CLI or data_dir field in the configuration file)

### Privacy and Training
- Applying differential privacy (DP) during DPCNN training through the Opacus framework
- Federated training for CNN, DPCNN, and XGBoost models
- Regression and classification workflows controlled by the `problem_type` parameter

### User Interaction
- Modifying model and privacy parameters through the provided `config.json` file
- Modifying model and privacy parameters through command-line arguments
- Validation of provided parameters against the provided JSON schema
- Validation of provided file paths against the currently visible directory
- Generation of machine-readable and JSON reports, with some values printed out to the terminal, in timestamped directories

_*Please note that users are expected to have already preprocessed their data prior to uploading it to the `data/` folder.*_

## Currently In-Progress Capabilities <a name="future_supported"></a>
- Improvement of report organization
- Support for setting the `opacus_secure_mode` parameter to true in DP

## Setting Up the Testbed <a name="setup"></a>

1. Download the zip file containing the data and code.

   The testbed expects ho (holdout) and tt (train-test) datasets to be present. While this is the expectation, please note that all data will be aggregated and then split between the provided clients. This behavior is different than the names suggest; however, it was beneficial due to the original size limitations of the Oil dataset. 
   - If using the provided Oil_binned5 test data, ensure the following files exist in the `data/Oil_binned5` directory (and your data path matches):
      1. `Oil_QTL_ho_pheno.dat`
      2. `Oil_QTL_ho_vcf.dat`
      3. `Oil_QTL_ohe_map.dat`
      4. `Oil_QTL_ohe.dat`
      5. `Oil_QTL_pheno_bins.dat`
      6. `Oil_QTL_tt_pheno.dat`
      7. `Oil_QTL_tt_vcf.dat`
   - Or if using the provided SCC test data and partitions, ensure the following files exist in the `data/gpd_scc` (and your data path and partition paths matches). Only one data partition file will be used (you must specify which one):
      1. `SCC_QTL_ho_pheno.dat`
      2. `SCC_QTL_ho_vcf.dat`
      3. `SCC_QTL_ohe_map.dat`
      4. `SCC_QTL_ohe.dat`
      5. `SCC_QTL_pheno_bins.dat`
      6. `SCC_QTL_tt_pheno.dat`
      7. `SCC_QTL_tt_vcf.dat`
      8. `ppfl_SCC_c0c1_5clients_2025_01_14.npz`
      9. `ppfl_SCC_c4c5_5clients_2025_01_14.npz`
   - If using your own data:
      - Please ensure that all data files are of type `.dat` or `.npy`. It is also expected that the endings of the data files match the provided test data. For example, this testbed assumes the data file endings are:
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
| `problem_type` | Selects the prediction task | string | `"regression"`, `"classification"` |
| `class_labels` | Ordered list of allowed labels for classification tasks. Required when `problem_type` is `"classification"` | array | At least 2 unique string, number, integer, or boolean values |
| `accuracy_tolerance` | Error tolerance used to count a regression prediction as accurate. Ignored for classification runs. | number | min: 0 |
| `num_cpus` | The number of CPUs available to the run | integer | min: 1, max: 100 |
| `num_gpus` | The number of GPUs available to the run | integer | min: 0, max: 100 |
| `num_rounds` | The number of rounds of federated learning | integer | min: 1, max: 100 |
| `num_clients` | The number of clients that will be sampled. Please note that providing a data partition file will overwrite this value | integer | min: 1, max: 100 |
| `data_partitions_file` | A path to a file that contains the data partitions | string | — |
| `partitioner_type` | The type of partitioner to use if a data partition file was not provided | string | `"uniform"`, `"linear"`, `"square"`, `"exponential"` |
| `num_partitions` | The number of data partitions. If this value is less than `num_clients` then those values may be lowered accordingly. | integer | min: 1, max: 100 |
| `partition_id` | Partition ID used for the current client | integer | min: 0, max: 100 |
| `client_id` | Client ID used for the current client | integer | min: 0, max: 100 |
| `seed` | The seed used to randomize training and testing | integer | min: 1, max: 1000 |
| `epochs` | The number of model training epochs. An epoch is one full pass through a training dataset. | integer | min: 1, max: 100 |
| `batch_divisor` | The divisor used to determine the number of batches (`num_batches = dataset_size / batch_divisor`) | integer | min: 1 |
| `test_fraction` | The fraction of the dataset to set aside for testing | number | exclusive min: 0, exclusive max: 1 |
| `learning_rate` | Sets the model’s learning rate. This defines how much a model changes at each iteration. | number | exclusive min: 0, exclusive max: 1 |
| `weight_decay` | Sets the model’s weight decay. This is a regularization method that penalizes high weights. | number | exclusive min: 0, exclusive max: 0.1 |
| `optimizer` | The optimizer is responsible for adjusting model parameters based on the value of the loss function | string | `"sgd"`, `"adamax"` |
| `opacus_secure_mode` | Turns on cryptographically secure differential privacy when set to `True` | boolean | — |
| `epsilon` | Determines the amount of privacy added to the data | number | exclusive min: 0, exclusive max: 50 |
| `delta` | Measures the chance of a data breach. It defines the probability that the noise does not add sufficient privacy. | number | min: 0, max: 1 |
| `max_grad_norm` | Clips the gradients to be under this maximum before adding noise | number | min: 0, max: 100 |
| `check_only` | A flag to turn on the check-only feature, which ensures all parameter values are within the appropriate range and have the correct type. When `true`, the testbed will not run. Execution will stop after the parameters are validated. | boolean | — |
| `data_dir` | The path to the intended data files to run the testbed on | string | — |
| `output_dir` | The path to the intended report directory. | string | - |
| `objective` | XGBoost objective. If classification uses the default regression objective, the code changes it to `multi:softprob`. | string | XGBoost objective string |
| `eta` | XGBoost learning rate | number | exclusive min: 0, max: 1 |
| `max_depth` | Maximum XGBoost tree depth | integer | min: 1, max: 100 |
| `eval_metric` | XGBoost evaluation metric. If classification uses the default `rmse`, the code changes it to `mlogloss`. | string | XGBoost metric string |
| `nthread` | Number of XGBoost threads | integer | min: 1, max: 100 |
| `num_parallel_tree` | Number of parallel trees per boosting round | integer | min: 1, max: 100 |
| `subsample` | Fraction of rows sampled for each XGBoost tree | number | exclusive min: 0, max: 1 |
| `tree_method` | XGBoost tree construction method | string | `"auto"`, `"exact"`, `"approx"`, `"hist"` |
| `colsample_bylevel` | Fraction of columns sampled at each tree level | number | exclusive min: 0, max: 1 |
| `colsample_bytree` | Fraction of columns sampled for each tree | number | exclusive min: 0, max: 1 |
| `gamma` | Minimum loss reduction required for an XGBoost split | number | min: 0 |
| `max_delta_step` | Maximum delta step for XGBoost tree weights | number | min: 0 |
| `min_child_weight` | Minimum child weight for XGBoost splitting | number | min: 0 |
| `reg_alpha` | XGBoost L1 regularization term | number | min: 0 |
| `reg_lambda` | XGBoost L2 regularization term | number | min: 0 |
| `scale_pos_weight` | XGBoost class-balancing weight | number | exclusive min: 0 |
| `train_method` | The XGBoost training method | string | `"bagging"`, `"cyclic"` |
| `centralised_eval` | Whether centralized evaluation is enabled for XGBoost | boolean | — |
| `scaled_lr` | Whether scaled learning rate behavior is enabled for XGBoost | boolean | — |
| `print_warning_logs` | Set to True to have warning logs print to the terminal. Set to False to have them directed to a log file in the output directory | boolean | - |

### Switching Between Regression and Classification

Use `problem_type` to choose the task:

```json
{
  "problem_type": "regression",
  "accuracy_tolerance": 0.1
}
```

```json
{
  "problem_type": "classification",
  "class_labels": [0, 1, 2, 3]
}
```

Regression assumes the phenotype labels are continuous numeric values. Regression accuracy is the fraction of predictions within `accuracy_tolerance` of the true label. For example, with `accuracy_tolerance` set to `0.1`, a prediction is counted as accurate if `abs(prediction - label) <= 0.1`.

Classification assumes the phenotype labels belong to a fixed set of classes. `class_labels` is required for classification so labels can be encoded consistently. The order of `class_labels` defines the encoded class indices used by CNN/DPCNN and XGBoost. Classification reports use class predictions and macro precision, macro recall, and macro F1 in addition to accuracy.

For XGBoost classification, if the default regression settings are still present, the run updates `objective` from `reg:squarederror` to `multi:softprob` and `eval_metric` from `rmse` to `mlogloss`.

### Parameter settings
To approximate centralized training from `run.py`, you should be able to use the following parameter values. Please note that the output may still look like federated learning training rounds, even though it is only one client, one round, and one data partition:

| Parameter | Value |
|---|---|
| `model_type` | `"dpcnn"` |
| `num_partitions` | `1` |
| `partition_id` | `0` |
| `client_id` | `0` |
| `num_clients` | `1` |
| `num_rounds` | `1` |
| `num_cpus` | `4` |
| `seed` | `673` |
| `test_fraction` | `0.3` |
| `epochs` | `100` |
| `learning_rate` | `0.01` |
| `batch_divisor` | `5` |
| `weight_decay` | `0.0001` |
| `optimizer` | `"sgd"` |
| `problem_type` | `"regression"` |
| `accuracy_tolerance` | `0.1` |
| `opacus_secure_mode` | `false` |
| `epsilon` | `0.2` |
| `delta` | `1e-05` |
| `max_grad_norm` | `1.0` |
| `print_warning_logs` | `false` |

Example `config.json`:

```json
{
    "model_type": "dpcnn",
   "num_partitions": 1,
   "partition_id": 0,
   "client_id": 0,
   "num_clients": 1,
   "num_rounds": 1,
   "num_cpus": 4,
   "seed": 673 ,
   "test_fraction": 0.3, 
   "epochs": 100, 
   "learning_rate": 0.01, 
   "batch_divisor": 5, 
   "weight_decay": 0.0001, 
   "optimizer": "sgd", 
   "problem_type": "regression",
   "accuracy_tolerance": 0.1,
   "opacus_secure_mode": false, 
   "epsilon": 0.2, 
   "delta": 1e-05, 
   "max_grad_norm":1.0,
   "print_warning_logs": false
}
```

Use `model_type: "cnn"` for non-DP CNN training. Use `model_type: "dpcnn"` for the Opacus DP CNN workflow.

## Running Batch Experiments <a name="batch"></a>

The purpose of batch experiments is to allow users to run the PETs Testbed on a series of values for a single parameter with the goal of identifying the effect that comes from incrementally modifying that particular parameter.

To run a batch experiment, run `python run_batch.py` from the dpcnn_opacus directory.

The batch experimentation script will prompt you for the parameter type and parameter values you would like to run the experiment on. All other parameters are taken from the config.json configuration file and are kept constant in each subsequent run. Ensure config.json has all of the other parameters you would like to use. 

## Understanding the Output <a name="output"></a>

### Files
Output files include `.npz` metadata files, `.json` human-readable reports, `.torch` CNN/DPCNN model files, and `.ubj` XGBoost model files. Within the `.npz` files, there is metadata on the model's global and client/round based performance. This is captured in the `.json` file format as well for human-readable purposes. The run also writes `input_parameters.json`, which records the resolved input parameters after schema/default processing. In the `.torch` and `.ubj` files, there are model weights that can be loaded for further inference with the trained model.

### Metric Definitions

#### Client/Round Outputs
| Metric | Description | Type | 
|---|---|---|
| `created on` | The datetime that the report was generated | string |
| `model id` | The id of the model being developed | integer |
| `round number` | The current round that the metrics are reporting on | integer |
| `partitions file` | The file used to partition the data for training | string |
| `problem type` | The task type used for the run | string |
| `class labels` | The configured class label order for classification runs, or `null` for regression | array or null |
| `train accuracy` | For classification, class prediction accuracy. For regression, fraction of predictions within `accuracy_tolerance`. | number |
| `test accuracy` | For classification, class prediction accuracy. For regression, fraction of predictions within `accuracy_tolerance`. | number |
| `train mean squared error` | Train MSE for this client and round. Classification reports set this to `0`. | number |
| `test mean squared error` | Test MSE for this client and round. Classification reports set this to `0`. | number |
| `train loss` | Train loss for this client and round | number |
| `test loss` | Test loss for this client and round | number |
| `train indices` | Indices used for training | integer array |
| `test indices` | Indices used for testing | integer array |
| `train accuracy per epoch` | Per-epoch training accuracy for CNN/DPCNN runs | number array |
| `test accuracy per epoch` | Per-epoch test accuracy for CNN/DPCNN runs | number array |
| `train mse per epoch` | Per-epoch training MSE for CNN/DPCNN regression runs. Classification reports use `0` values. | number array |
| `test mse per epoch` | Per-epoch test MSE for CNN/DPCNN regression runs. Classification reports use `0` values. | number array |
| `losses per epoch` | Loss values for each epoch in this client round. Present for CNN/DPCNN reports. | number array |
| `epsilon per epoch` | The epsilon values for each epoch in this client round. Present for DPCNN reports. | number array |
| `train predictions` | The predicted values for the training set | number array |
| `test predictions` | The predicted values for the test set | number array |
| `hyperparameters` | The hyperparameters the client used during its training process | object of the following properties: |
| `hyperparameters.learning rate` | Sets the model’s learning rate. This defines how much a model changes at each iteration | number | 
| `hyperparameters.weight decay` | Sets the model’s weight decay. This is a regularization method that penalizes high weights | number | 
| `hyperparameters.batch divisor` | The divisor to determine the number of batches (num_batches = dataset_size/batch_divisor) | integer |
| `hyperparameters.epochs` | The number of model training epochs. An epoch is one full pass through a training dataset | integer |
| `hyperparameters.seed` | The seed used to randomize training/testing | integer | 
| `hyperparameters.test fraction` | The fraction of the data set to set aside for testing | number |
| `hyperparameters.problem type` | The task type used for the run | string |
| `hyperparameters.class labels` | The configured class label order for classification runs, or `null` for regression | array or null |
| `hyperparameters.accuracy tolerance` | Regression accuracy tolerance. `null` for classification runs. | number or null |
| `hyperparameters.optimizer` | The optimizer is responsible for adjusting model parameters based on the value of the loss function | string |
| `hyperparameters.epsilon` | Determines the amount of privacy added to the data. | number |
| `hyperparameters.delta` | Measures the chance of a data breach. It defines the probability of the noise not adding sufficient privacy | number |
| `hyperparameters.max grad norm` | Clips the gradients to be under this maximum before adding noise | number |

#### Global Outputs
| Metric | Description | Type | 
|---|---|---|
| `created on` | The datetime that the report was generated | string |
| `loss per round` | Loss values across rounds | number array |
| `accuracy per round` | For classification, class prediction accuracy across rounds. For regression, tolerance-based accuracy across rounds for CNN/DPCNN and rounded-prediction accuracy for current XGBoost global evaluation. | number array |
| `mse per round` | MSE values across rounds. Classification reports use `0` values. | number array |

### Accuracy Calculations and Model Implications

Accuracy is calculated differently depending on `problem_type`.

For regression, model outputs are continuous numeric predictions. Accuracy is the fraction of predictions within the configured `accuracy_tolerance`:

```python
np.mean(np.abs(predictions - labels) <= accuracy_tolerance)
```

Regression reports also include MAE, MSE, RMSE, R2, the label mean, and RMSE as a percentage of the label mean.

For classification, labels are encoded from `class_labels`, and predictions are class indices. CNN/DPCNN classification uses the largest output logit as the predicted class. XGBoost classification uses `multi:softprob` probabilities when the default classification conversion is applied, then selects the class with the largest probability. Classification reports include accuracy, macro precision, macro recall, and macro F1. Regression-only metrics such as MSE are set to `0` in classification reports.

The older behavior of treating a regression output as a rounded class prediction is no longer the default task model. Use `problem_type: "classification"` for class labels and `problem_type: "regression"` for continuous numeric targets.

## Results Viewer <a name="viewer"></a>

To make it easier for users to view the results of both singular runs and batch experiments, we have created a web-based results viewer that displays the output data in an easy-to-use user interface.

To access the results viewer, run the following command from the genomics_ppfl_base directory while replacing {PORT_NUMBER} with a desired port of your choosing. Ensure the port is not already in use by running `lsof -i :{PORT_NUMBER}`.  

`python3 -m http.server {PORT_NUMBER}`

You can then access the results viewer from your browser at `http://localhost:{PORT_NUMBER}/results_viewer.html`

When you are done using the results viewer, you can shutdown your web server by pressing CTRL + C. 
Sometimes CTRL + C will not completely kill the process. To ensure your process is killed, run `lsof -i :{PORT_NUMBER}`. Locate the process' PID from output and run `kill -9 {PID}`.

### Accessing the Results Viewer located remotely

If the machine that the PETs Testbed and test results are located on does not have a desktop interface (ex. CLI-only VM), you will need to copy your results files to your local machine and use SSH tunneling to access the results viewer.

On your remote VM:
1. Ensure the port is not already in use by running `lsof -i :{PORT_NUMBER}`
1. Start your web server by running `pyenv exec python3 -m http.server {PORT_NUMBER}`

On your local machine:
1. Ensure you have a local copy of the PETs_Testbed code 
1. Move your results files to you local machine by running `scp '{REMOTE_USER}@{REMOTE_IP}:{REMOTE_PETS_TESTBED_DIRECTORY_PATH}/reports/*' {LOCAL_PETS_TESTBED_DIRECTORY_PATH}/reports/`. If your VM does not have password authentication enabled, you may need to specify your SSH key by running `scp -i {SSH_KEY_PATH} '{REMOTE_USER}@{REMOTE_IP}:{REMOTE_PETS_TESTBED_DIRECTORY_PATH}/reports/*' {LOCAL_PETS_TESTBED_DIRECTORY_PATH}/reports/`.
1. Initiate your SSH tunnel by running `ssh -L {PORT_NUMBER}:localhost:{PORT_NUMBER} {REMOTE_USER}@{REMOTE_IP}`
1. As long as your SSH tunnel session is active, you should now be able to access and use the results viewer on your local machine's browser at `http://localhost:{PORT_NUMBER}/results_viewer.html`

## Data References <a name="refs"></a>

### Soybean Trait Prediction Research

Published paper: [Machine learning models outperform deep learning models, provide interpretation and facilitate feature selection for soybean trait prediction](https://bmcplantbiol.biomedcentral.com/articles/10.1186/s12870-022-03559-z)

Jupyter notebooks for the soybean paper (GitHub): [Soybean_Trait_Prediction](https://github.com/mitchgill16/Soybean_Trait_Prediction)

#### Datasets

Dataset host site: https://data.pawsey.org.au/projects/

Unfortunately, you cannot share a link that goes directly to the folders containing the dataset CSV files, so you will have to navigate through the UI's folder structure to `/NGS Analysis Results/shortTerm/mgill/DL/holdout_and_equivalent_merged_1pcnt_removed`.

In that folder, you will see the `holdout` and `train_test` datasets named with the feature as a prefix, for example `"FlC_"` for flower color.

## Flower/Ray client, model, and logging behavior <a name="flwr"></a>

This codebase currently follows the standard Flower simulation pattern:

- One data partition corresponds to one Flower client.
- One Flower client trains one local model update per federated round.

### Parameters that affect how many clients train

- `num_partitions` controls how many simulated client partitions are available when random partitioning is used.
- `num_clients` determines how many clients are trained when no data partition file is provided.
- When `data_partitions_file` is provided, the code reads the number of `num_client` entries in that file and uses those as the available clients and client data partitions.

### Parameters that affect concurrency

- `num_cpus` does not directly control how many models are trained.
- In Ray, `num_cpus` is a per-client resource reservation.
- Lowering `num_cpus` can allow more Flower clients to run at the same time, so logs from multiple clients may appear interleaved.
- Increasing `num_cpus` can force more sequential execution by making each client reserve more of the machine.

### Interpreting output

Current output may look like:

```
Client 0 | Epoch 1/100 | ...
Client 3 | Epoch 1/100 | ...
Client 1 | Epoch 1/100 | ...
```

It means multiple client processes are training concurrently, and Ray prints logs as each process emits them. The order is based on scheduling/runtime progress, not client id order.

### Planned cleanup

To reduce confusion in a future non-hotfix change:

- Document the relationship between partition files, Flower clients, and federated rounds directly in the configuration docs.
- Keep memory-related changes separate from naming/logging cleanup to avoid expanding the current hotfix scope.

## Memory-mapped data loading <a name="mem"></a>

The Flower simulation path uses memory-mapped `.npy` files for the DPCNN data arrays. This was added to reduce Ray out-of-memory failures caused by each client process loading and copying large genomics arrays.

### Previous behavior

The original Flower/Ray path loaded pickled `.dat` arrays inside each client process. It then built additional NumPy arrays such as:

- `vcf = np.concatenate((tt_vcf, ho_vcf), axis=0)`
- `pheno = np.concatenate((tt_pheno, ho_pheno), axis=0)`
- `combined_dataset = np.concatenate((vcf, pheno), axis=1)`
- client train/test slices such as `combined_dataset[train_indices]`

Those operations create full in-memory copies. With multiple Ray clients, the same large dataset could be loaded and copied several times at once.

### Current behavior

The federated client/server path now uses `load_npy_feature_label_data`, which:

- checks for required `.npy` files matching `_tt_vcf`, `_tt_pheno`, `_ho_vcf`, and `_ho_pheno`
- automatically converts missing `.npy` files from the matching `.dat` files
- opens the `.npy` arrays with `np.load(..., mmap_mode="r")`
- keeps the arrays file-backed instead of eagerly loading each full array into every Ray process

Using `mmap_mode="r"` means NumPy creates an array-like view over the `.npy` file instead of immediately copying the whole file into process memory. The operating system loads pages from the file only as rows are accessed. Since Ray runs clients in separate worker processes, this is important: multiple workers can map the same read-only data files without each worker eagerly owning a separate full private copy of every array.

This does not make the dataset free. Rows that are actively read still occupy memory, and PyTorch/Opacus still allocate tensors, gradients, optimizer state, and batch data during training. The benefit is that baseline dataset storage is file-backed and shared more efficiently by the OS, so memory usage is driven more by active training work and less by repeated full dataset copies in each Ray client.

The large `tt` and `ho` feature/label arrays stay physically separate:

- `tt_vcf`
- `tt_pheno`
- `ho_vcf`
- `ho_pheno`

The `IndexedArrayDataset` class treats those separate arrays as one logical `tt + ho` dataset. Global row indices keep their original meaning: rows `0..len(tt)-1` refer to `tt`, and later rows refer to `ho` after subtracting `len(tt)`.

This preserves the original partition-file behavior without building large concatenated arrays.

### Automatic conversion

If one or more required `.npy` files are missing, `dataset.py` calls `convert_dat_to_npy.convert_dat_to_npy(data_dir)` automatically. Existing `.npy` files are left in place, so conversion should only happen when needed.

The converter only converts `.dat` files that contain NumPy arrays. Non-array pickle files are skipped.

### Expected memory behavior

This change reduces memory by avoiding repeated full dataset copies across Ray client workers. It does not eliminate all memory use. Training can still use several GB of RAM because PyTorch, Opacus, Ray actors, optimizer state, gradients, and active batches all allocate memory.

A moderate RAM peak during training is expected. The important improvement is that memory should no longer scale as badly with repeated dataset copies per client. If running into OOM issues, try increasing the number of cpus allocated for each Ray/Flwr client (increase the num_cpus parameter from the command line or config.json). This will reduce the number of clients running at any given time and therefore reduce the overall RAM usage.

### Legacy path

The legacy `load_pickle_data` function remains available for older scripts such as centralized training. The federated Flower client/server path should use the mmap-backed loader instead.
