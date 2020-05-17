import numpy as np

from typing import List, Optional

class LinearConstraint:
    def __init__(self, coefficients: np.ndarray, inequality: str, constant: float=0):
        self._coefficients = coefficients
        self._inequality = inequality
        self._constant = constant



    @property
    def inequality(self):
        return self._inequality



    @property
    def coefficients(self):
        return self._coefficients



    @property    
    def constant(self):
        return self._constant



class VariableConstraint:
    def __init__(self, positive_variables: Optional[List[int]]=None, negative_variables: Optional[List[int]]=None):
        self._positive_variables = positive_variables
        self._negative_variables = negative_variables



    @property
    def positive_variables(self) -> List[int]:
        return self._positive_variables



    @property
    def negative_variables(self) -> List[int]:
        return self._negative_variables
