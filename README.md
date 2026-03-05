## Running instructions:

### How to run:
1. Download the data from the google drive (`PETs Testbed/PPFL/Genomics Data/Paper_Datasets/Seed_oil/seed_oil_binned5`)
2. Move the raw `.dat` files to the `data/Oil_binned5` directory. The files included will be:
   1. Oil_QTL_ho_pheno.dat
   2. Oil_QTL_ho_vcf.dat
   3. Oil_QTL_ohe_map.dat
   4. Oil_QTL_ohe.dat
   5. Oil_QTL_pheno_bins.dat
   6. Oil_QTL_tt_pheno.dat
   7. Oil_QTL_tt_vcf.dat
3. Download/validate python 3.10 installation (other installations result in errors with the `torch` library). Previous installations should not be affected.
   1. Run the command `python -V` to check the installation version
   2. If the wrong version of python (or no pythoninstallation) is installed, download the Python Install Manager found [here](https://www.python.org/downloads/) or [from the Microsoft store](https://apps.microsoft.com/detail/9NQ7512CXL7T?hl=en-us&gl=US&ocid=pdpshare)
   3. Run the install manager and follow the steps to install the correct `py` version. When this is installed, restart your terminal and run the command `py install 3.10`.
   4. Restart the terminal again, and run the command `python3.10.exe -V` to confirm a successful installation.
5. `cd` into the directory where the downloaded project is. You should see folder for `data`, `dpcnn_opacus`, etc. Create a virtual environment with the command `python3.10.exe -m venv .venv`.
6. Confirm that the `.venv` directory is created by running `ls` or `dir`.
7. Activate the virtual environment.
   1. On linux: `source ./.venv/bin/activate`
   2. On powershell `./.venv/Scripts/activate.ps1`
8. Install packages with the commands `python.exe -m pip install --upgrade pip` then `pip install -r requirements.txt` while in the home directory
   1. Note that if there are errors with the `torch` package, you may need to uninstall the packages and reinstall, using `pip uninstall -y -r .\requirements.txt`
9. Change directory to the `dpcnn_opacus` directory (note that this is necessary due to relative pathing)
10. Run the command: `python run.py` for a simple run with a single client, or `python run.py --min-fit-clients 2 --min-evaluate-clients 2 --min-available-clients 2` for 2 clients

### Regression testing
For the regression testing, I am using a python library call pytest. This helps to automate the testing process. There are various test cases described in the regression_test.py script (which are also described in the regression testing excel sheet). In this regression testing, it is only checking to see if the parameter inputs are valid. There are no running end-to-end tests (even though some tests are denoted as end-to-end). This is something that can be improved while Rebecca is out :). This testing takes roughly 7 minutes as is on the openstack VM. 

To run the regression testing, ensure you are in the dpcnn_opacus folder (with the regression_test.py file) and if using a venv it is active. Also ensure you have pytest installed in your environment (it is now listed in the requirements.txt). Then, run `python3.10 -m pytest -q`. This will loop through every single testcase with a progress tracker at the bottom. Any failed tests will be printed out at the end. 

To manually go through each test case using pytest, first gather a list of all possible test (I recommend recording it in a .txt file for easy lookup) by running `python3.10 -m pytest --collect-only -q > test_list.txt`. Then, identify the test you want to run (let's call it `regression_test.py::foo_test[test_id]`). To run that individual test, run `python3.10 -m pytest regression_test.py::foo_test[test_id]`. For exmaple, to run test 12a, ` python3.10 -m pytest regression_test.py::test_run_py_regressions[t12a]`.

To manually go through each test case without using pytest, you will need to identify the test in the excel sheet. Either update the config.json file to match what is needed, or manually create a new configuration file with the listed values (note that the base condfiguration file may differ than the original values of config.json). Once that configuration file is done, follow the run command on the excel sheet (note that if you are editing the config.json file you do _not_ need to specify the path for the configuration file with the --config).

## Rebecca's Updates
### Adding the Initial Codebase
This is a copy of our current code base.

### Adding Redteaming Analysis Code
Added Amy Hilla's red teaming analysis code, as well as other misc folders she sent Rebecca.

### Red Teaming Analysis
The red teaming analysis has been added, but it is important to note that the submission files of each contestent have been omitted. Those files can be found in the google drive folder. 

For each of the problems, the file structure of the submissions should be as follows:

{problem1_submissions_and_answers, DogData_submissions_answers_scores}
|
|---answers
|
|---submissions (YOU ADD THIS FOLDER)
        |
        |---problem{1,2}
               |
               |---{team_1}
               |      |
               |      |---submission_{n}
               |               |
               |               |---{cnn, dpcnnx}_submission_file.csv
               |---{team_2}
                      |
                      |---submission_{n}
                    
# PPFL-Framework
Privacy-preserving federated learning framework built in conjunction with @usnistgov

This framework currently works with Python v3.10. Installing v1.13 of PyTorch fails for v3.11+. Versions of Python below 3.10 may work but have not been tested.

## Dependencies
Required modules are listed in the `requirements.txt` files. It is recommended to install them with pip via `pip install -r requirements.txt` while using [pyenv](https://github.com/pyenv/pyenv) or another environment manager.

## Data Processing
The file `get_data_subset.py` in the `data_processing/genetic_plant_data` directory can be used to obtain subsets of the original soybean genome datasets. The file can be run with CLI arguments that are described with the help flag, e.g. `python data_processing/genetic_plant_data/get_data_subset.py -h` from the project root directory.

### Example
Assuming you have one of the original datasets, `FlC_Merged_filtered.csv_train_test.csv`, and have stored it in `genetic_plant_data/original_datasets/` directory, you can obtain the first 1000 rows of that dataset via running this command from the root directory:
```bash
python data_processing/genetic_plant_data/get_data_subset.py --source-directory='genetic_plant_data/original_datasets/' --source-file='FlC_Merged_filtered.csv_train_test.csv' --num-rows=1000
```

By default, this will create a new file in the `genetic_plant_data` directory named `subset_data.csv`.

(See the "Datasets" subsection in the "References" section below to obtain the original datasets)

The file `merge_csv_files.py` in the `data_processing/genetic_plant_data` directory can be used to combine the various genomic files stored as local CSVs. The file can be run with CLI arguments that are described with the help flag, e.g. `python data_processing/genetic_plant_data/merge_csv_files.py -h` from the project root directory.

The `genome_files_load_and_pickle.py` in the `data_processing/genetic_plant_data` directory can be used to read in genetic CSV files and create pickled output files used by the Flower framework. The file can be run with CLI arguments that are described with the help flag, e.g. `python data_processing/genetic_plant_data/genome_files_load_and_pickle.py -h` from the project root directory.

## XGBoost
The XGBoost model in this repository is based on the original XGBoost library: https://xgboost.readthedocs.io/en/stable/

The hyperparameters are generated from a Bayesian Hyperparameter search using scikit-optimize, as utilized in the soybean prediction paper (see [References](#references) section below): https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html

### Centralized
To run the centralized XGBoost example, run `python xgboost/centralized_train.py`. Note that this will create a pickled file of the optimized hyperparameter search values to be referenced (if present) by the federated server and client.

### Federated
To run the basic FL example using XGBoost, from the project root directory, run `python xgboost/server.py` to start the server, then `python xgboost/client.py` to run the client and execute model training. As noted above, training hyperparameters are referenced, if available, from a pickle file that stored values from a prior centralized training run. Otherwise the parameters are referenced from the `BST_PARAMS` object in `xgboost/utils.py`.

Hyperparameters such as number of clients can be passed as CLI arguments when running the above commands. Add an `-h` to the command (e.g., `python xgboost/server.py -h`) to see the available variables and their defaults.

Both server and clients of federated xgboost can also be run using the `bash xgboost/run.sh` script. For running the federated xgboost with custom data-partitions use `bash xgboost/run_custom_paritions.sh`. For this it is required to edit the `xgboost/run_custom_paritions.sh` script to provide the correct path to the custom data-partitions file on your system.

## Convolutional Neural Network (CNN)
The CNN model in this repository is [PyTorch](https://pytorch.org/)-based. The architecture parameters (e.g., layer types and sizes) of the centralized model are based on the CNN in the original soybean prediction paper (see [References](#references) section below). Note that the notebooks for the soybean paper use Tensorflow rather than PyTorch, but the architecture is the same.

### Centralized
To run the centralized CNN example, run `python cnn/centralized_train.py` from the project root directory. Hyperparameters such as the number of training epochs can be passed as arguments. Run `python cnn/centralized_train.py -h` to see the available hyperparameter arguments.

### Federated
To run the federated CNN, from the project root directory, run `python cnn/server.py` to start the server, then `python cnn/client.py` to run the client and execute model training.

Hyperparameters such as number of clients can be passed as CLI arguments when running the above commands. Add an `-h` to the command (e.g., `python cnn/server.py -h`) to see the available variables and their defaults.

## References

### Soybean Trait Prediction Research

Published Paper: [Machine learning models outperform deep learning models, provide interpretation and facilitate feature selection for soybean trait prediction](https://bmcplantbiol.biomedcentral.com/articles/10.1186/s12870-022-03559-z)

Jupyter Notebooks for soybean paper (Github): [Soybean_Trait_Prediction](https://github.com/mitchgill16/Soybean_Trait_Prediction)

#### Datasets

Datasets host site: https://data.pawsey.org.au/projects/

Unfortunately you can't share a link that goes directly to the folders containing the dataset CSV files, so you'll have to navigate though the UI's folder structure to `/NGS Analysis Results/shortTerm/mgill/DL/holdout_and_equivalent_merged_1pcnt_removed`.
In that folder you'll see the holdout and train_test datasets named with the feature as a prefix, e.g. "FlC_" = flower color.