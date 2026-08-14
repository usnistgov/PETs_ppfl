# Privacy-Enhancing Technologies (PETs) Testbed

> The NIST Genomics PETs Testbed (beta) aims to provide a framework for evaluating the efficacy of privacy-enhancing technologies (PETs) on genomics machine learning workloads. It supports centralized and federated training using convolutional neural network (CNN), differentially private CNN, and extreme gradient boosting (XGBoost) models and provides tools for users to compare how experiment configurations and privacy settings affect model performance.

# Table of Contents
1. [System Specifications](#specs)
1. [Quickstart](#quick)
1. [Background](#background)
    * [Differential Privacy](#dp)
    * [Federated Learning](#fl)
    * [Main Components](#main_components)
    * [Typical Workflow](#workflow)
    * [Testbed Capabilities](#capabilities)
1. [Installation and Setup](#setup)
1. [Datasets and Data Behavior](#data)
1. [Running the Testbed](#running)
    * [Adjusting testbed parameters using the configuration file](#conf)
    * [Adjusting testbed parameters using the command line](#cli)
1. [Running Batch Experiments](#batch)
1. [Results Viewer](#viewer)
    *  [Accessing the Results Viewer located remotely](#local)
1. [Additional Documentation](#documentation)
1. [License](#license)

## System Specs <a name="specs"></a>

Tested on:
- OS: Ubuntu 24.04.4 LTS
- Kernel: 6.8.0-136-generic
- Architecture: x86_64
- Environment: KVM/QEMU virtual machine
- CPU: 16 vCPUs, Intel Xeon Processor (Cascadelake)
- Memory: 32 GiB RAM
- Storage: 164 GB virtual disk
- GPU: None

## Quickstart <a name="quick"></a>

1. Confirm that Python 3.10 is installed:
   ```bash
   python -V
   # or
   python3 -V
   ```
2. Create and activate a virtual environment:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Change into the testbed directory:
   ```bash
   cd PETs_Testbed
   ```
5. Run the testbed:
   ```bash
   python run.py
   ```

To validate parameters without running training:
```bash
python run.py --check_only
```

## Background <a name="background"></a>
### Differential Privacy <a name="dp"></a>

Differential Privacy (DP) is a machine learning (ML) privacy mechanism that adds noise to a dataset to obscure the impact of any given data entry on the overall model. In other words, it ensures that the result from a training epoch would be statistically the same regardless of whether a given entry was present or not. This protects an individual's privacy in ML, as it prevents an actor from being able to determine whether an individual's data was used in the training process or not. Within this PETs testbed, the Opacus library was used to create a DP enabled CNN model (DPCNN). 

More information on the Opacus library can be found [here](https://opacus.ai/). 
More information on DP can be found [here](https://digitalprivacy.ieee.org/publications/topics/what-is-differential-privacy/).

### Federated Learning <a name="fl"></a>

Federated Learning (FL) allows for different groups to train a single ML model without having to share or reveal their data. FL allows each group (called a client) to train their own model, then send their model weights to a server for aggregation with other clients' models. Next, the aggregated model weight is sent back to each of the clients and another round of training can occur. FL allows for clients to keep their data private while benefiting from other clients' datasets. To simulate FL within this PETs testbed, the flwr library was used.

More information on the flwr library can be found [here](https://flower.ai/).

### Main Components of the Testbed <a name="main_components"></a>

The PETs Testbed consists of three primary components that together support the complete experimentation workflow.

1. **Primary driver - `run.py`**

   `run.py` is the primary entry point for the testbed. It executes a single experiment using the configuration provided in configs/config.json or through command-line arguments.

2. **Batch driver - `run_batch.py`**

   `run_batch.py` automates the execution of a series of experiments while varying a single parameter. This tool is intended for studying how changing parameters affects model performance.

3. **Results Viewer**

   The Results Viewer is a web-based user interface for viewing experiment outputs, inspecting metrics and comparing runs.

4. **Parameter Schema**
A JSON schema that defines the parameters, the parameter types, the parameter ranges, and the default parameter values for the PETs Testbed. This acts as the source of truth for the parameterization work.

5. **Parameter Input JSON**
A flat JSON file that can be used to supply parameter values to the PETs Testbed in an easily modifiable way.

### Typical Testbed Workflow <a name="workflow"></a>

A typical workflow looks like:

1. Prepare your dataset
1. Configure experiment parameters
1. Run one or more experiments using run.py or run_batch.py
1. View results using the Results Viewer

### Testbed Capabilities <a name="capabilities"></a>

This testbed supports:
- Differential privacy (DP) during DPCNN training through Opacus
- Federated training for CNN, DPCNN, and XGBoost models
- Regression and classification workflows controlled by the `problem_type` parameter
- Configuration through `config.json` and command-line arguments
- Validation of parameters against a JSON schema
- Validation of file paths against the currently visible directory
- Machine-readable and JSON reports, with some values printed to the terminal, in timestamped directories

> Users are expected to preprocess their data before uploading it to the `data/` folder.

Throughout the beta development process, these capabilities are expected to expand. Feedback on additional features and discovered issues is welcome and encouraged.

## Installation and Setup <a name="setup"></a>

> The testbed expects Python 3.10. Other versions may cause errors with the `torch` library.
> If Python 3.10 is not installed, try running the `setup.sh` bash script, which downloads Python 3.10 from the deadsnakes repository.

> Please note that this codebase is designed for Mac/Linux distributions. Windows distributions may behave differently.

### Instructions

1. Download the zip file or clone the git repo containing the data and code.
   - If using the provided sample SCC test data and ensure the following files exist in the `data/SCC` (and your data path and partition paths matches):
      1. `SCC_QTL_ho_pheno.dat`
      2. `SCC_QTL_ho_vcf.dat`
      3. `SCC_QTL_tt_pheno.dat`
      4. `SCC_QTL_tt_vcf.dat`
   - If using your own data:
      - Please ensure that all data files are of type `.dat` or `.npy`. It is also expected that the endings of the data files match the provided test data. For example, this testbed assumes the data file endings are:
         1. `_ho_pheno.dat`
         2. `_ho_vcf.dat`
         3. `_tt_pheno.dat`
         4. `_tt_vcf.dat`
      - You may also provide your own data partition files. Examples of old (non-usable) data partition files can be seen in the `SCC` data at `ppfl_SCC_c0c1_5clients_2025_01_14.npz` and `ppfl_SCC_c4c5_5clients_2025_01_14.npz`. However, if you attempt to use one of these, it will not work. This is because legacy behavior concatenated the tt and ho datasets, whereas the current PETs testbed does not.
      > Please also remember to update the `data_dir` variable either through the command line or through the provided configuration file.
2. Download or validate a Python 3.10 installation (other versions may result in errors with the `torch` library) via the `python -V` or `python3 -V` commands.
   1. Run the command `python -V` to check the installed version.
   2. If the wrong version of Python is installed, or if Python is not installed, try:
      -  Running the `setup.sh` bash script, which downloads Python 3.10 from the deadsnakes repository
   3. Run the command `python3.10 -V` to confirm a successful installation.
3. `cd` into the directory where the downloaded project is located. You should see folders such as `data`, `PETs_Testbed`, etc. Create a virtual environment with the command `python3.10 -m venv .venv`.
4. Confirm that the `.venv` directory was created by running `ls -la`.
5. Activate the virtual environment. On success, `(.venv)` should be prepended to your shell prompt.
   1. On Mac/Linux: `source ./.venv/bin/activate`
6. Install packages with the commands `python -m pip install --upgrade pip` and then `pip install -r requirements.txt` while in the project root directory.
   1. Note that if there are errors with the `torch` package, you may need to uninstall the packages and reinstall them using `pip uninstall -y -r .\requirements.txt`
7. Change directories to the `PETs_Testbed` directory
8. Run the command: 
```bash
python run.py
```
   > Note that this will use the `configs/config.json` file for input parameters. These inputs are validated through the `schemas/configuration-schema.json` file.
      
## Datasets and Data Behavior <a name="data"></a>
The testbed expects two datasets to be provided: the train-test (tt) dataset and the holdout (ho) dataset. The tt dataset is used to train the client(s). The ho dataset is kept separate during the training rounds and used to evaluate the final model at the end.

When there are `n` number of clients, the tt dataset is split amongst those clients. If no data partition file is provided, the tt dataset is split between the `n` clients using the partitioning method passed into `partitioner_type`. When a data partition file is provided, it is assumed that it partitions just the tt dataset. In other words, when a data partition file is provided, the length of the data partition file must match the length of the tt dataset. When provided, the tt dataset is split in accordance to the partition file.

<details>
<summary><strong> Sample Dataset: Soybean Trait Prediction Research </strong></summary>
While users are encouraged to bring and test their own datasets, a small sample dataset on soybeans may be found in the resources below. This was one of the datasets used to build and test this PETs testbed, and can run relatively quickly due to its small size. This makes it ideal for learning how to use the PETs testbed.

Published paper: [Machine learning models outperform deep learning models, provide interpretation and facilitate feature selection for soybean trait prediction](https://bmcplantbiol.biomedcentral.com/articles/10.1186/s12870-022-03559-z)

Jupyter notebooks for the soybean paper (GitHub): [Soybean_Trait_Prediction](https://github.com/mitchgill16/Soybean_Trait_Prediction)

Dataset host site: https://data.pawsey.org.au/projects/

Unfortunately, you cannot share a link that goes directly to the folders containing the dataset CSV files, so you will have to navigate through the UI's folder structure to `/NGS Analysis Results/shortTerm/mgill/DL/holdout_and_equivalent_merged_1pcnt_removed`.

In that folder, you will see the `holdout` and `train_test` datasets named with the feature as a prefix, for example `"FlC_"` for flower color.
</details>

## Running the Testbed <a name="running"></a>

To run the testbed, navigate to the PETs_Testbed folder and run:

```bash 
python3.10 run.py

# Or to just validate parameters:
python3.10 run.py --check_only
```

This will load the default parameter values from the schema and run a simple federated learning workflow with differential privacy. 

There are several ways to adjust the parameter values but regardless of how the parameter values are specified the values will be validated against the types and ranges specified by `schemas/configuration-schema.json`. Any provided file paths, such as a data partition file or an output directory, must exist prior to running the testbed to avoid early termination. If parameter errors occur, error messages and suggestions should be printed to the terminal.

### Adjusting testbed parameters using the configuration file <a name="conf"></a>

One way to modify the parameters of the testbed is to edit the `configs/config.json` file with new values. To do this, simply open the JSON file and add the field for the parameter you want to modify. You do not need to mirror the layered structure of the schema. Please enter parameters as key-value pairs, such as `"epochs": 20` or `"data_dir": "../data/my_data"`. Please note that the configuration file will be validated at runtime, so if there are invalid values or incorrect types, you will be able to correct them before running the testbed. See [the table below](#parameter-definitions) for details on each parameter you can modify.

If you want to create a separate JSON file for a test case to help record the inputs, first make a copy of `configs/config.json`, then rename it, and finally change the values to match your desired experiment. You can tell the testbed to use this configuration file through the `--config /my/path/too/my_config.json` command-line flag. If you want to make a copy of and edit `schemas/configuration-schema.json`, you can also pass in the schema through the `--schema /my/path/too/my_schema.json` command-line flag.

### Adjusting testbed parameters using the command line <a name="cli"></a>

If you want to modify a value for a single run, or see whether a parameter input is valid, and do not want to modify the `.json` configuration file, then you can use the command line to modify one or more parameters. To do this, treat the desired variables as flags and use the format `--foo bar`, which would set the `foo` parameter's value to `bar`. One exception to this is the `--check_only` flag, which does not expect any values and just turns on the parameter validation mechanism.

Modifying parameters in this way will not affect the contents of the configuration file.

Please note that some variables are only set through the command line. These variables are `--config` (to tell the testbed to use a configuration file other than `configs/config.json`), `--schema` (to tell the testbed to use a JSON schema file other than `schemas/configuration-schema.json`), and `--check_only` (which, when passed, tells the testbed to only check whether the parameter values given are supported).

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
   | `batch_size` | The size of each training batch | integer | min: 8 |
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
   | `centralized_eval` | Whether centralized evaluation is enabled for XGBoost | boolean | — |
   | `scaled_lr` | Whether scaled learning rate behavior is enabled for XGBoost | boolean | — |
   | `print_warning_logs` | Set to True to have warning logs print to the terminal. Set to False to have them directed to a log file in the output directory | boolean | - |
   | `use_public_data` | Set to True if you have public data you want to be included in the train-test dataset. | boolean | - |

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

Regression assumes the phenotype labels are continuous numeric values. Regression accuracy is the fraction of predictions within `accuracy_tolerance` of the true label. For example, with `accuracy_tolerance` set to `0.1`, a prediction is counted as accurate if `abs(prediction - label) <= 0.1`. Regression experiments report accuracy, mse, rmse, and loss.

Classification assumes the phenotype labels belong to a fixed set of classes. `class_labels` is required for classification so labels can be encoded consistently. The order of `class_labels` defines the encoded class indices used by CNN/DPCNN and XGBoost. Classification experiments report class predictions, macro precision, macro recall, and macro F1, loss, and accuracy.

For XGBoost models, the backend `objective` and `eval_metric` will be automatically updated to reflect the problem type. For example, during XGBoost classification the testbed automatically updates `objective` from `reg:squarederror` to `multi:softprob` and `eval_metric` from `rmse` to `mlogloss`.

</details>

## Running Batch Experiments <a name="batch"></a>

The purpose of batch experiments is to allow users to run the PETs Testbed on a series of values for a single parameter with the goal of identifying the effect that comes from incrementally modifying that particular parameter.

To run a batch experiment, run `python3.10 run_batch.py` from the `PETs_Testbed` directory.

The batch experimentation script will prompt you for the parameter type and parameter values you would like to run the experiment on. All other parameters are taken from the configs/config.json configuration file and are kept constant in each subsequent run. Ensure configs/config.json has all of the other parameters you would like to use. 

## Results Viewer <a name="viewer"></a>

To make it easier for users to view the results of both singular runs and batch experiments, we have created a web-based results viewer that displays the output data in an easy-to-use user interface.

To access the results viewer, run the following command from the genomics_ppfl_base directory while replacing {PORT_NUMBER} with a desired port of your choosing. Ensure the port is not already in use by running `lsof -i :{PORT_NUMBER}`.  

`python3.10 -m http.server {PORT_NUMBER}`

You can then access the results viewer from your browser at `http://localhost:{PORT_NUMBER}/results_viewer.html`

The viewer uses `fetch()` to load report schemas from `./schemas/`. This means it **must be served via HTTP** (e.g. `python -m http.server` from the repo root). Opening the HTML file directly via `file://` will fail silently due to browser CORS restrictions.

When you are done using the results viewer, you can shutdown your web server by pressing CTRL + C. 
Sometimes CTRL + C will not completely kill the process. To ensure your process is killed, run `lsof -i :{PORT_NUMBER}`. Locate the process' PID from output and run `kill -9 {PID}`.

### Accessing the Results Viewer located remotely <a name="local"></a>

If the machine that the PETs Testbed and test results are located on does not have a desktop interface (ex. CLI-only VM), you will need to copy your results files to your local machine and use SSH tunneling to access the results viewer.

On your remote VM:
1. Ensure the port is not already in use by running `lsof -i :{PORT_NUMBER}`
1. Start your web server by running `pyenv exec python3.10 -m http.server {PORT_NUMBER}`

On your local machine:
1. Ensure you have a local copy of the PETs_Testbed code 
1. Move your results files to your local machine by running `scp '{REMOTE_USER}@{REMOTE_IP}:{REMOTE_PETS_TESTBED_DIRECTORY_PATH}/reports/*' {LOCAL_PETS_TESTBED_DIRECTORY_PATH}/reports/`. If your VM does not have password authentication enabled, you may need to specify your SSH key by running `scp -i {SSH_KEY_PATH} '{REMOTE_USER}@{REMOTE_IP}:{REMOTE_PETS_TESTBED_DIRECTORY_PATH}/reports/*' {LOCAL_PETS_TESTBED_DIRECTORY_PATH}/reports/`.
1. Initiate your SSH tunnel by running `ssh -L {PORT_NUMBER}:localhost:{PORT_NUMBER} {REMOTE_USER}@{REMOTE_IP}`
1. As long as your SSH tunnel session is active, you should now be able to access and use the results viewer on your local machine's browser at `http://localhost:{PORT_NUMBER}/results_viewer.html`

## Additional Documentation <a name="documentation"></a>

Below you will find links to additional technical information for the `PETs_Testbed`. This contains in-depth information on topics such as unit/regression tests, Flower/Ray behavior, and fixes added to the codebase to prevent out of memory (OOM) errors. Casual users of the `PETs_Testbed` may skip this section, unless they are interested in learning about these details.

- [Flower/Ray client, model, and logging behavior](./docs/flower_behavior.md)
- [Memory-mapped data loading](./docs/memory_mapped_data_loading.md)
- [Understanding the raw output](./docs/understanding_output.md)
- [Unit tests](./docs/unit_tests.md)

## License <a name="license"></a>

Please refer to the licensing statement [here](https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software).