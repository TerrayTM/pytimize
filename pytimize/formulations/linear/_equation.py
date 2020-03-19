import numpy as np

from typing import Dict

class LinearEquation:
    # TODO add check for math indexing in terms
    def __init__(self, terms: Dict[int, float], constant: float=0):
        self._terms = terms
        self._constant = constant



    def add_variable(self, label, coefficient):
        previous = self._terms.setdefault(label, 1)
        previous[label] = previous + coefficient



    def __add__(self, other):
        pass



    def __radd__(self, other): 
        pass 

    def __iadd__(self, other):
        pass

    def __neg__(self):
        pass

    def __sub__(self, other):
        pass 

    def __rsub__(self, other):
        pass

    @property
    def coefficients(self) -> np.ndarray:
        coefficients = np.array([0] * max(self._terms.keys()))

        for label, coefficient in self._terms.items():
            coefficients[label - 1] = coefficient

        return coefficients

    @property
    def constant(self) -> float:
        return self._constant
