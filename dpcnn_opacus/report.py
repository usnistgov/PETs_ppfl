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

    def to_json_string(self, indent: int = 4) -> str:
        """
        Converts the dictionary to a JSON formatted string.
        :param indent: Number of spaces for indentation (default 4).
        :return: JSON string.
        """
        try:
            return json.dumps(self.data, indent=indent, cls=NumpyEncoder)
        except (TypeError, ValueError) as e:
            return f"Error encoding JSON: {e}"

    def save_to_file(self, filename: str):
        """
        Saves the dictionary to a .json file.
        :param filename: Name of the file (e.g., 'data.json').
        """
        try:
            with open(filename, 'w') as json_file:
                json.dump(self.data, json_file, indent=4, cls=NumpyEncoder)
            print(f"Successfully saved to {filename}")
        except IOError as e:
            print(f"File error: {e}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)