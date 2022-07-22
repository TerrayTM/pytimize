import re

import numpy as np


# TODO ADD CONSTRAINT
class LinearParser:
    @staticmethod
    def parse(expression: str, size: int = None) -> np.ndarray:
        if LinearParser.validate_expression(expression):
            term = re.compile(r"(\+|-)?(\d+(?:\.\d+)?)x_(\d+)")
            data = {}
            length = 0

            for sign, coefficient, index in term.findall(expression):
                index = int(index) - 1

                if index in data:
                    raise Exception()
                else:
                    data[index] = float(("-" if sign == "-" else "") + coefficient)

                    if index > length:
                        length = index

            result = np.full((1, length + 1 if size is None else size), 0, float)

            for key, value in data.items():
                result[0, key] = value

            return result
        else:
            raise Exception()

    @staticmethod
    def parse_multiple(expressions: list, size: int = None):
        for expression in expressions:
            if LinearParser.validate_expression(expression):
                LinearParser.parse(expression, size)
            else:
                raise Exception()

    @staticmethod
    def validate_expression(candidate):
        validation = re.compile(r"^-?\d+(?:\.\d+)?x_\d+(?:[+|-]\d+(?:\.\d+)?x_\d+)*$")

        return validation.match(candidate) is not None
