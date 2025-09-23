import os
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import flwr as fl
import pickle

from utils import file_load_args_parser
from logging import INFO


def oh_feature_mappings(
        data: pd.DataFrame,
        ohe: OneHotEncoder
) -> Dict[str, Tuple[int, int]]:
    """
    Compute the index ranges for one-hot encoded columns of each original
    feature.

    Parameters:
    data (pd.DataFrame): The original dataset with features to be
        one-hot encoded.
    ohe (OneHotEncoder): The fitted one-hot encoder.

    Returns:
    dict: A mapping of original feature names to their one-hot encoded column
        index ranges.
    """
    feature_index_ranges = {}
    current_index = 0

    for i, categories in enumerate(ohe.categories_):
        start_index = current_index
        end_index = start_index + len(categories) - 1
        feature_index_ranges[data.columns[i]] = (start_index, end_index)
        current_index += len(categories)

    return feature_index_ranges


def new_prep_data(
        tt_file: str,
        ho_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Much of this code is lightly modified from the function of the
    same name in https://github.com/mitchgill16/Soybean_Trait_Prediction/
    blob/main/FlowerColour_ReducedInput_BC.ipynb

    Parameters:
    tt_file (str): Path to the train/test data file.
    ho_file (str): Path to the holdout data file.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: Tuple of
    traintest VCF data, holdout VCF data, traintest phenotypes, and
    holdout phenotypes.
    """
    def load_in_chunks(data_file: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load data in chunks and separate SNPs and phenotypes.

        Parameters:
        data_file (str): The path to the data file.

        Returns:
        tuple[pd.DataFrame, np.ndarray]: Tuple of VCF data and corresponding
        phenotypes.
        """
        my_list_chunks = []
        my_list_phenos = []
        x = 0

        for chunk in pd.read_csv(data_file,
                                 chunksize=10000,
                                 index_col="Unnamed: 0"):
            x = x+10000
            if 'Value' in chunk.columns:
                # does the selecting of pheno array for application ML
                chunk["Value"] = pd.to_numeric(chunk["Value"],
                                               downcast="float")
                labels = chunk["Value"].to_numpy()
                # reshapes it so its not a 1D array
                labels = np.reshape(labels, (len(labels), 1))
                my_list_phenos.append(labels)
                chunk = chunk.drop(columns=['Value'])
            else:
                raise ValueError(f"Phenotype array (Value column) not found \
                                 in data file: {data_file}")
            headers = chunk.columns
            row_idx = chunk.index
            # SHOULD TURN ./. into the most common for each column
            chunk = imp.fit_transform(chunk)
            # since imputing makes a numpy array have to turn back into PD
            # for label encoding
            chunk = pd.DataFrame(data=chunk, index=row_idx, columns=headers)
            my_list_chunks.append(chunk)

        vcf = pd.concat(my_list_chunks, axis=0)
        pheno = np.concatenate(my_list_phenos, axis=0)
        return vcf, pheno

    imp = SimpleImputer(missing_values='./.', strategy='most_frequent')
    tt_vcf, tt_pheno = load_in_chunks(tt_file)
    ho_vcf, ho_pheno = load_in_chunks(ho_file)

    # check final shape. Columns should be same number, rows should roughly
    # be 5x bigger for tt_vcf
    fl.common.logger.log(INFO, f"train_test shape: {tt_vcf.shape}")
    fl.common.logger.log(INFO, f"holdout shape: {ho_vcf.shape}")
    if tt_vcf.shape[1] != ho_vcf.shape[1]:
        raise ValueError("train/test and holdout datasets have \
                         mismatched column counts")
    return tt_vcf, ho_vcf, tt_pheno, ho_pheno


args = file_load_args_parser()
cur_path = os.path.dirname(__file__)
source_directory = args.source_directory
# move directory two levels up
project_root = os.path.join(cur_path, "..", "..")
data_directory = os.path.join(os.getcwd(), source_directory)
train_test_filename = f"{data_directory}/{args.train_test_filename}"
holdout_filename = f"{data_directory}/{args.holdout_filename}"
pickle_output_directory = args.pickle_output_directory
prefix_label = args.output_files_prefix_label

# breakpoint()
print(f"train_test_filename: {train_test_filename}")
print(f"holdout_filename: {holdout_filename}")
tt_vcf, ho_vcf, tt_pheno, ho_pheno = new_prep_data(
    train_test_filename, holdout_filename)

# fits one hot encoder first and then transforms both training test data
# and holdout data in the same way
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe = ohe.fit(tt_vcf)

tt_ohe_map = oh_feature_mappings(tt_vcf, ohe)
fl.common.logger.log(INFO, f"tt_vcf.shape pre-transform: {tt_vcf.shape}")
tt_vcf = ohe.transform(tt_vcf)

fl.common.logger.log(INFO, f"tt_vcf.shape post-transform: {tt_vcf.shape}")
fl.common.logger.log(INFO, f"ho_vcf.shape pre-transform: {ho_vcf.shape}")
ho_ohe_map = oh_feature_mappings(ho_vcf, ohe)
ho_vcf = ohe.transform(ho_vcf)
fl.common.logger.log(INFO, f"ho_vcf.shape post-transform: {ho_vcf.shape}")

ohe_map = {"traintest": tt_ohe_map, "holdout": ho_ohe_map}

# saves the onehot encoder and data for future use
path_substr = f"{pickle_output_directory}/{prefix_label}_QTL"
pickle.dump(ohe, open(f"{path_substr}_ohe.dat", "wb"))
pickle.dump(tt_vcf, open(f"{path_substr}_tt_vcf.dat", "wb"))
pickle.dump(ho_vcf, open(f"{path_substr}_ho_vcf.dat", "wb"))
pickle.dump(tt_pheno, open(f"{path_substr}_tt_pheno.dat", "wb"))
pickle.dump(ho_pheno, open(f"{path_substr}_ho_pheno.dat", "wb"))
pickle.dump(ohe_map, open(f"{path_substr}_ohe_map.dat", "wb"))

# if need or have new holdout data etc.
ohe = pickle.load(open(f"{path_substr}_ohe.dat", "rb"))

#
#
# def find_snp_from_header(
#         categories: List[List[str]],
#         snp_num: int, verify_snp: str
# ) -> str:
#     count = 0
#     snp = "Not found"
#     found = False
#     i = 0
#     while i < len(categories) and (found is False):
#         j = 0
#         while j < len(categories[i]):
#             if (count == snp_num):
#                 snp = categories[i][j]
#                 if snp == verify_snp:
#                     found = True
#                 else:
#                     raise ValueError("Unexpected SNP value")
#                 break
#             count = count + 1
#             j = j + 1
#         i = i + 1
#     return snp
#
#
# # TESTING IF IT WORKS eg. input data point 200 is A/G
# my_snp = find_snp_from_header(ohe.categories_, 200, "G/G")
# print(my_snp)
