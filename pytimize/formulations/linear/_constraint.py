import numpy as np

class LinearConstraint:
    def __init__(self, coefficients: np.ndarray, inequality: str, constant: float=0):
        self._coefficients = coefficients
        self._inequality = inequality
        self._constant = constant



    def __repr__(self):
        pass # TODO



    def compile(self):
        pass # TODO

    

    @property
    def inequality(self):
        return self._inequality



    @property
    def coefficients(self):
        return self._coefficients



    @property    
    def constant(self):
        return self._constant
