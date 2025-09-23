import csv
import os

from utils import file_merge_args_parser

"""
This script reads though CSVs in a directory and grabs the headers and
rows of the first file. For subsequent files, it sees if a row
identifier was seen in the initial file, and if so, appends the data,
and skips the row if it was not in the initial file rows.

The result should be a new CSV data file wherein each row contains
matching data from across all CSV files.

This is combined with CLI arguments defining the input file directory,
the output filename, and number of rows (it scans the entire file by default).
"""


def concatenate_columns(
        source_directory,
        new_filename,
        output_directory,
        row_limit=None
):
    # Get the list of CSV files in the directory
    csv_files = [file for file in os.listdir(source_directory)
                 if file.endswith('.csv')]
    reading_first_file = True
    seen_columns = set()
    file_data = {}

    # Iterate through each CSV file
    for file in csv_files:
        file_path = os.path.join(source_directory, file)
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            if row_limit is None:
                row_limit = sum(1 for _ in reader)
            # Iterate though the file rows
            for i, row in enumerate(reader):
                first_cell = row[0]
                if first_cell == '':
                    # remove leading empty cell for header row
                    row = row[1:]

                if i < row_limit:
                    if reading_first_file:
                        # create new row data
                        seen_columns.add(first_cell)
                        file_data[first_cell] = row[1:]
                    elif first_cell in seen_columns:
                        # append existing row data
                        file_data[first_cell] = file_data[first_cell] + row[1:]
                else:
                    break

        reading_first_file = False
        print(f'completed reading file {file_path}')

    # Write the concatenated columns to a new CSV file
    output_file_path = os.path.join(output_directory, new_filename)
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        for key, value in file_data.items():
            writer.writerow([key] + value)
    print('created concatenated file')


args = file_merge_args_parser()
source_directory = args.source_directory
filename = args.filename
output_directory = args.output_directory
num_rows = args.num_rows

concatenate_columns(source_directory, filename, output_directory, num_rows)
