import numpy as np

from ...programs import LinearProgram
from typing import List

class AbstractLinearProgram:
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, z: float, objective: str, inequalities: List[str]):
        self._A = A
        self._b = b
        self._c = c
        self._z = z
        self._objective = objective
        self._inequalities = inequalities
        self._negative_variables = []
        self._positive_variables = []



    def where(self, *variable_constraints):
        return self



    def fill(self, length):
        pass



    def compile(self):
        return LinearProgram(self._A, self._b, self._c, self._z, self._objective, self._inequalities, [], self._negative_variables)
