# This Software (PETs Testbed) is being made available as a public service by the
# National Institute of Standards and Technology (NIST), an Agency of the United
# States Department of Commerce. This software was developed in part by employees of
# NIST and in part by NIST contractors. Copyright in portions of this software that
# were developed by NIST contractors has been licensed or assigned to NIST. Pursuant
# to Title 17 United States Code Section 105, works of NIST employees are not
# subject to copyright protection in the United States. However, NIST may hold
# international copyright in software created by its employees and domestic
# copyright (or licensing rights) in portions of software that were assigned or
# licensed to NIST. To the extent that NIST holds copyright in this software, it is
# being made available under the Creative Commons Attribution 4.0 International
# license (CC BY 4.0). The disclaimers of the CC BY 4.0 license apply to all parts
# of the software developed or licensed by NIST.
#
# ACCESS THE FULL CC BY 4.0 LICENSE HERE:
# https://creativecommons.org/licenses/by/4.0/legalcode

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