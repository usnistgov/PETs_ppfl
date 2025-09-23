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
