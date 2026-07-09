# Privacy-Enhancing Technologies (PETs) Testbed

The NIST Genomics PETs Testbed (beta) aims to provide a framework for evaluating the efficacy of privacy-enhancing technologies (PETs) on genomics machine learning workloads. It supports centralized and federated training using convolutional neural network (CNN), differentially private CNN, and extreme gradient boosting (XGBoost) models and provides tools for users to compare how experiment configurations and privacy settings affect model performance.

# Table of Contents
1. [Main Components](#main_components)
1. [Typical Workflow](#workflow)
1. [Capabilities](#capabilities)
1. [Prerequisites](#prerequisites)
1. [Running the Testbed](#running)
1. [Running Batch Experiments](#batch)
1. [Output](#output)
1. [Results Viewer](#viewer)
1. [Data References](#ref)
1. [Additional Documentation](#docs)
1. [License](#license)

## Main Components <a name="main_components"></a>

The PETs Testbed consists of three primary components that together support the complete experimentation workflow.

1. **Primary driver - `run.py`**

   `run.py` is the primary entry point for the testbed. It executes a single experiment using the configuration provided in config.json or through command-line arguments.

2. **Batch driver - `run_batch.py`**

   `run_batch.py` automates the execution of a series of experiments while varying a single parameter. This tool is intended for studying how changing parameters affects model performance.

3. **Results Viewer**

   The Results Viewer is a web-based user interface for viewing experiment outputs, inspecting metrics and comparing runs.

## Typical Workflow <a name="workflow"></a>

A typical workflow looks like:

1. Prepare your dataset
1. Configure experiment parameters
1. Run one or more experiments using run.py or run_batch.py
1. View results using the Results Viewer

## Capabilities <a name="capabilities"></a>

Throughout the beta development process, capabilities of the testbed are expected to expand. Feedback on additional features and discovered issues is welcome and encouraged.

### Supported Capabilities 

- Supports any dataset provided in the `data/` folder (or any other folder specified by the --data_dir CLI or data_dir field in the configuration file)
- Modifying model and privacy parameters through configuration file
- Modifying model and privacy parameters through command-line arguments
- Validation of provided parameters against the provided JSON schema
- Validation of provided file paths against the currently visible directory
- Differential privacy (DP) during DPCNN training through the Opacus framework
- Federated training for CNN, DPCNN, and XGBoost models
- Regression and classification workflows controlled by the `problem_type` parameter
- Generation of machine-readable and JSON reports in timestamped directories

### In-Progress Capabilities

- Improvement of report organization
- Support for setting the `opacus_secure_mode` parameter to true in DP

## Prerequisites <a name="prerequisites"></a>

1. **Python 3.10**

   Download or validate a Python 3.10 installation (other versions may result in errors with the `torch` library) via the `python -V` or `python3 -V` commands.
      1. Run the command `python -V` to check the installed version.
      2. If the wrong version of Python is installed, or if Python is not installed, try running the `setup.sh` bash script, which downloads Python 3.10 from the deadsnakes repository.
      3. Run the command `python3.10 -V` to confirm a successful installation.

2. **Python virtual environment**

   Using a Python virtual environment will help prevent dependency conflicts.
      1. `cd` into the repository root. You should see folders such as `data`, `PETs_Testbed`, etc.
      2. Create a virtual environment with the command `python3.10 -m venv .venv`. 
      3. Confirm that the `.venv` directory was created by running `ls -la`.
      4. Activate the virtual environment each time you use the testbed. On Linux, you would run `source ./.venv/bin/activate`. On success, `(.venv)` should be prepended to your shell prompt.

3. **Package dependencies**

   Install pip by running `python3.10 -m pip install --upgrade pip`
   Navigate to the repository root and install package dependencies by running `pip install -r requirements.txt`.
      1. Note that if there are errors with the `torch` package, you may need to uninstall the packages and reinstall them using `pip uninstall -y -r .\requirements.txt`
      
4. **Genomics dataset**

   Download and unzip your dataset into any directory, although it is recommended to place your dataset in the `/data` folder. Data should be preprocessed. 

   <details>
   <summary><strong> Dataset Details </strong></summary>

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

   </details>

## Running the Testbed <a name="running"></a>

To run the testbed, navigate to the PETs_Testbed folder and run:

```bash 
python3.10 run.py

# Or to just validate parameters:
python3.10 run.py --check_only
```

This will load the default parameter values from the schema and run a simple federated learning workflow with differential privacy. 

There are several ways to adjust the parameter values but regardless of how the parameter values are specified the values will be validated against the types and ranges specified by `configuration-schema.json`. Any provided file paths, such as a data partition file or an output directory, must exist prior to running the testbed to avoid early termination. If parameter errors occur, error messages and suggestions should be printed to the terminal.

### Adjusting testbed parameters using the configuration file

One way to modify the parameters of the testbed is to edit the `config.json` file with new values. To do this, simply open the JSON file and add the field for the parameter you want to modify. You do not need to mirror the layered structure of the schema. Please enter parameters as key-value pairs, such as `"epochs": 20` or `"data_dir": "../data/my_data"`. Please note that the configuration file will be validated at runtime, so if there are invalid values or incorrect types, you will be able to correct them before running the testbed. See [the table below](#parameter-definitions) for details on each parameter you can modify.

If you want to create a separate JSON file for a test case to help record the inputs, first make a copy of `config.json`, then rename it, and finally change the values to match your desired experiment. You can tell the testbed to use this configuration file through the `--config my_config.json` command-line flag. If you want to make a copy of and edit `configuration-schema.json`, you can also pass in the schema through the `--schema my_schema.json` command-line flag.

### Adjusting testbed parameters using the command line

If you want to modify a value for a single run, or see whether a parameter input is valid, and do not want to modify the `.json` configuration file, then you can use the command line to modify one or more parameters. To do this, treat the desired variables as flags and use the format `--foo bar`, which would set the `foo` parameter's value to `bar`. One exception to this is the `--check_only` flag, which does not expect any values and just turns on the parameter validation mechanism.

Modifying parameters in this way will not affect the contents of the configuration file.

Please note that some variables are only set through the command line. These variables are `--config` (to tell the testbed to use a configuration file other than `config.json`), `--schema` (to tell the testbed to use a JSON schema file other than `configuration-schema.json`), and `--check_only` (which, when passed, tells the testbed to only check whether the parameter values given are supported).

<details>
<summary><strong> Parameter Definitions </strong></summary>

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

</details>

<details>
<summary><strong> Switching Between Regression and Classification </strong></summary>

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

</details>

<details>
<summary><strong> Centralized Training </strong></summary>

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

</details>

## Running Batch Experiments <a name="batch"></a>

The purpose of batch experiments is to allow users to run the PETs Testbed on a series of values for a single parameter with the goal of identifying the effect that comes from incrementally modifying that particular parameter.

To run a batch experiment, run `python run_batch.py` from the dpcnn_opacus directory.

The batch experimentation script will prompt you for the parameter type and parameter values you would like to run the experiment on. All other parameters are taken from the config.json configuration file and are kept constant in each subsequent run. Ensure config.json has all of the other parameters you would like to use. 

## Understanding the Output <a name="output"></a>

### Files

Output files include `.npz` metadata files, `.json` human-readable reports, `.torch` CNN/DPCNN model files, and `.ubj` XGBoost model files. Within the `.npz` files, there is metadata on the model's global and client/round based performance. This is captured in the `.json` file format as well for human-readable purposes. The run also writes `input_parameters.json`, which records the resolved input parameters after schema/default processing. In the `.torch` and `.ubj` files, there are model weights that can be loaded for further inference with the trained model.

### Metric Definitions

<details>
<summary><strong> Client/Round Outputs </strong></summary>

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

</details>

<details>
<summary><strong> Global Outputs </strong></summary>

| Metric | Description | Type | 
|---|---|---|
| `created on` | The datetime that the report was generated | string |
| `loss per round` | Loss values across rounds | number array |
| `accuracy per round` | For classification, class prediction accuracy across rounds. For regression, tolerance-based accuracy across rounds for CNN/DPCNN and rounded-prediction accuracy for current XGBoost global evaluation. | number array |
| `mse per round` | MSE values across rounds. Classification reports use `0` values. | number array |

</details>

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

## Additional Documentation <a name="documentation"></a>

- [Flower/Ray client, model, and logging behavior](./docs/flower_behavior.md)
- [Memory-mapped data loading](./docs/memory_mapped_data_loading.md)

## License <a name="license"></a>

This Software (PETs Testbed) is being made available as a public service by the National Institute of Standards and Technology (NIST), an Agency of the United States Department of Commerce. This software was developed in part by employees of NIST and in part by NIST contractors. Copyright in portions of this software that were developed by NIST contractors has been licensed or assigned to NIST. Pursuant to Title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States. However, NIST may hold international copyright in software created by its employees and domestic copyright (or licensing rights) in portions of software that were assigned or licensed to NIST. To the extent that NIST holds copyright in this software, it is being made available under the Creative Commons Attribution 4.0 International license (CC BY 4.0). The disclaimers of the CC BY 4.0 license apply to all parts of the software developed or licensed by NIST.

ACCESS THE FULL CC BY 4.0 LICENSE HERE:
https://creativecommons.org/licenses/by/4.0/legalcode