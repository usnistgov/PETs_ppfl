import csv
import os

from utils import file_subset_args_parser

"""
This script reads in a soybean genome CSV file and grabs the first X number of
rows (default 100) and the final 'Value' row to create a subset of that CSV.
"""


def concatenate_columns(
        source_directory,
        source_file,
        new_filename,
        output_directory,
        row_limit
):
    if not source_file.endswith('.csv'):
        raise ValueError("Source file is not a CSV")

    input_file_path = os.path.join(source_directory, source_file)
    output_file_path = os.path.join(output_directory, new_filename)

    # Iterate through CSV file rows
    with open(input_file_path, 'r') as input_file, open(
              output_file_path, 'w') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)
        print('reading input file')
        # Iterate though the file rows
        for i, row in enumerate(reader):
            if i < row_limit:
                writer.writerow(row)
            elif i == row_limit:  # get last line
                print('obtaining last row')
                last_row = input_file.readlines()[-1]
                # remove newline character (strip) and convert single string
                # to list (split)
                writer.writerow(last_row.strip().split(','))
            else:
                break

    print(f'completed parsing file {source_file}')
    print(f'created subset file {new_filename}')


args = file_subset_args_parser()
source_directory = args.source_directory
source_file = args.source_file
filename = args.filename
output_directory = args.output_directory
num_rows = args.num_rows

concatenate_columns(source_directory, source_file,
                    filename, output_directory, num_rows)
