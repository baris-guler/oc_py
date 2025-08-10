from ocpy.errors import LengthCheckError


class Checker:
    @staticmethod
    def length_checker(data, reference):
        if len(reference) != len(data):
            raise LengthCheckError(f"length of data is not sufficient")


class Fixer:
    @staticmethod
    def length_fixer(data, reference):
        if isinstance(data, str):
            return [data] * len(reference)

        if hasattr(data, "__len__"):
            Checker.length_checker(data, reference)
            return data
        else:
            return [data] * len(reference)
