# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software

import json
import numpy as np

class Report:
    """Class to handle report components."""

    def __init__(self, data: dict):
        """
        Initialize the converter with a dictionary.
        :param data: A dictionary object to be converted.
        """
        if not isinstance(data, dict):
            raise TypeError("Input must be a dictionary object.")
        self.data = data

    def save_to_file(self, filename: str):
        """
        Saves the dictionary to a .json file.
        :param filename: Name of the file (e.g., 'data.json').
        """
        try:
            with open(filename, 'w') as json_file:
                json.dump(self.data, json_file, indent=4, cls=NumpyEncoder)
            print(f"Successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving metadata to JSON file: {e}")

# Converts Numpy values into their non-numpy equivalents        
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # Turn numpy array into list
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Turn numpy integer into integer
        if isinstance(obj, np.integer):
            return int(obj)
        # Turn numpy float into float
        if isinstance(obj, np.floating):
            return float(obj)
        # Default to the JSONEncoder for other types
        try:
            val = super(NumpyEncoder, self).default(obj)
            return val
        except Exception as e:
            print(f"Error resolving type {type(obj)} for object into JSON. Error: {e}")