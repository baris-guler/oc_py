import numpy as np

from ocpy.errors import LengthCheckError


class Checker:
    @staticmethod
    def length_checker(data, reference):
        if len(reference) != len(data):
            raise LengthCheckError(f"length of data is not sufficient")


class Fixer:
    @staticmethod
    def length_fixer(data, reference):
        if reference is None:
            return data

        if isinstance(data, str):
            return np.array([data] * len(reference), dtype=object)

        if hasattr(data, "__len__"):
            Checker.length_checker(data, reference)
            if isinstance(data, list):
                return np.array(data)
            return data
        else:
            return np.array([data] * len(reference))

    @staticmethod
    def none_to_nan(data_frame):
        return data_frame.replace({None: np.nan})
