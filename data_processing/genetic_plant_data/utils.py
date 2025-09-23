import argparse


def file_load_args_parser():
    """Parse arguments for loading CSV files"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-directory",
        default="./genetic_plant_data",
        type=str,
        help="Directory to source genomic data ( \
                default = 'genetic_plant_data').",
    )
    parser.add_argument(
        "--train-test-filename",
        default=('transpose_FlC_Merged_filtered.'
                 'csv_train_test_rows_2463183_2476598_with_header.csv'),
        type=str,
        help=("Output filename for combined genomic data (default = "
              "'transpose_FlC_Merged_filtered."
              "csv_train_test_rows_2463183_2476598_with_header.csv')."),
    )
    parser.add_argument(
        "--holdout-filename",
        default=('transpose_FlC_Merged_filtered.'
                 'csv_holdout_rows_2463183_2476598_with_header.csv'),
        type=str,
        help=("Output filename for combined genomic data (default = "
              "'transpose_FlC_Merged_filtered."
              "csv_holdout_rows_2463183_2476598_with_header.csv')."),
    )
    parser.add_argument(
        "--pickle-output-directory",
        default="./genetic_plant_data",
        type=str,
        help="Directory to output genomic data (default = \
            './genetic_plant_data').",
    )
    parser.add_argument(
        "--output-files-prefix-label",
        default="FC",
        type=str,
        help=("Prefix label for output files (default = 'FC' -- Flower Color)."),
    )

    args = parser.parse_args()
    return args


def file_merge_args_parser():
    """Parse arguments for merging CSV data files"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-directory",
        default="./",
        type=str,
        help="Directory to source genomic data (default = './').",
    )
    parser.add_argument(
        "--filename",
        default='combined_data.csv',
        type=str,
        help=("Output filename for combined genomic data "
              "(default = 'combined_data.csv')."),
    )
    parser.add_argument(
        "--output-directory",
        default="./genetic_plant_data/",
        type=str,
        help=("Directory to output genomic data (default = "
              "'./genetic_plant_data/')."),
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        help=("Number of rows selected for combining in new file "
              "(default = full length of file)."),
    )

    args = parser.parse_args()
    return args


def file_subset_args_parser():
    """Parse arguments for creating subset of CSV dataset file"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-directory",
        type=str,
        help="Source file directory to source genomic data (no default).",
    )
    parser.add_argument(
        "--source-file",
        type=str,
        help="Source filename to source genomic data (no default).",
    )
    parser.add_argument(
        "--filename",
        default='subset_data.csv',
        type=str,
        help=("Output filename for combined genomic data "
              "(default = 'subset_data.csv')."),
    )
    parser.add_argument(
        "--output-directory",
        default="./genetic_plant_data/",
        type=str,
        help=("Directory to output genomic data "
              "(default = './genetic_plant_data/')."),
    )
    parser.add_argument(
        "--num-rows",
        default="100",
        type=int,
        help=("Number of rows selected for combining in new file "
              "(default = 100)."),
    )

    args = parser.parse_args()
    return args
